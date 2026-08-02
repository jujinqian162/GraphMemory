from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from graph_memory.datasets.isetrace.records import (
    ISETraceAssistantMessage,
    ISETraceRecord,
    ISETraceSystemMessage,
    ISETraceToolMessage,
    ISETraceUserMessage,
)
from graph_memory.trajectories import (
    CanonicalTrajectory,
    MessageEvent,
    SourceIntent,
    SourcePosition,
    ToolCallEvent,
    ToolDefinition,
    ToolOutputEvent,
    TrajectoryEvent,
)

_RECORD_ADAPTER = TypeAdapter(ISETraceRecord)
_ERROR_PATTERN = r'(?:Traceback \(most recent call last\)|No such file or directory|"status"\s*:\s*"error"|\bcommand not found\b)'
_ERROR_TEXT = re.compile(_ERROR_PATTERN, re.IGNORECASE)


class ISETraceAdaptationError(ValueError):
    code: str
    record_id: str

    def __init__(self, code: str, record_id: str, detail: str) -> None:
        self.code = code
        self.record_id = record_id
        super().__init__(f"record={record_id} [{code}] {detail}")


@dataclass
class ISETraceIngestionSummary:
    records_seen: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    tool_calls: int = 0
    tool_outputs: int = 0
    argument_parse_failures: int = 0
    reported_success_error_text_conflicts: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.records_rejected += 1
        self.rejection_reasons[reason] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "records_seen": self.records_seen,
            "records_accepted": self.records_accepted,
            "records_rejected": self.records_rejected,
            "tool_calls": self.tool_calls,
            "tool_outputs": self.tool_outputs,
            "argument_parse_failures": self.argument_parse_failures,
            "reported_success_error_text_conflicts": (
                self.reported_success_error_text_conflicts
            ),
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


def parse_isetrace_record(value: object) -> ISETraceRecord:
    return _RECORD_ADAPTER.validate_python(value)


def iter_isetrace_records(path: str | Path) -> Iterator[ISETraceRecord]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield _RECORD_ADAPTER.validate_json(line)
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    f"invalid ISETrace record at {source}:{line_number}: {error}"
                ) from error


def adapt_isetrace_record(
    record: ISETraceRecord,
    *,
    source_revision: str,
    summary: ISETraceIngestionSummary | None = None,
) -> CanonicalTrajectory:
    active_summary = summary
    events: list[TrajectoryEvent] = []
    call_count = 0
    output_count = 0

    for message_index, message in enumerate(record.messages):
        if isinstance(
            message,
            (ISETraceSystemMessage, ISETraceUserMessage, ISETraceAssistantMessage),
        ):
            events.append(
                MessageEvent(
                    event_id=f"{record.session_id}:message:{message_index}",
                    position=SourcePosition(
                        message_index=message_index,
                        sub_index=0,
                    ),
                    role=message.role,
                    content=message.content or "",
                    reasoning_content=(
                        message.reasoning_content
                        if isinstance(message, ISETraceAssistantMessage)
                        else None
                    ),
                )
            )
        if isinstance(message, ISETraceAssistantMessage):
            for call_index, call in enumerate(message.tool_calls, start=1):
                try:
                    parsed_arguments_value = cast(
                        object, json.loads(call.function.arguments)
                    )
                except json.JSONDecodeError as error:
                    if active_summary is not None:
                        active_summary.argument_parse_failures += 1
                    raise ISETraceAdaptationError(
                        "invalid_arguments_json",
                        record.session_id,
                        f"call_id={call.id}: {error}",
                    ) from error
                if not isinstance(parsed_arguments_value, dict):
                    if active_summary is not None:
                        active_summary.argument_parse_failures += 1
                    raise ISETraceAdaptationError(
                        "arguments_not_object",
                        record.session_id,
                        f"call_id={call.id} arguments must decode to an object",
                    )
                parsed_arguments = cast(
                    dict[str, JsonValue], parsed_arguments_value
                )
                events.append(
                    ToolCallEvent(
                        event_id=f"{record.session_id}:call:{call.id}",
                        position=SourcePosition(
                            message_index=message_index,
                            sub_index=call_index,
                        ),
                        call_id=call.id,
                        tool_name=call.function.name,
                        arguments=parsed_arguments,
                        raw_arguments=call.function.arguments,
                    )
                )
                call_count += 1
        elif isinstance(message, ISETraceToolMessage):
            events.append(
                ToolOutputEvent(
                    event_id=f"{record.session_id}:output:{message.tool_call_id}",
                    position=SourcePosition(
                        message_index=message_index,
                        sub_index=0,
                    ),
                    call_id=message.tool_call_id,
                    tool_name=message.name,
                    content=message.content,
                    source_reported_success=message.success,
                )
            )
            output_count += 1
            if (
                active_summary is not None
                and message.success is True
                and _ERROR_TEXT.search(message.content)
            ):
                active_summary.reported_success_error_text_conflicts += 1

    tool_definitions: list[ToolDefinition] = []
    for tool in record.tools:
        try:
            parsed_parameters_value = cast(
                object, json.loads(tool.function.parameters)
            )
        except json.JSONDecodeError as error:
            raise ISETraceAdaptationError(
                "invalid_tool_schema_json",
                record.session_id,
                f"tool={tool.function.name}: {error}",
            ) from error
        if not isinstance(parsed_parameters_value, dict):
            raise ISETraceAdaptationError(
                "tool_schema_not_object",
                record.session_id,
                f"tool={tool.function.name} parameters must decode to an object",
            )
        parsed_parameters = cast(
            dict[str, JsonValue], parsed_parameters_value
        )
        tool_definitions.append(
            ToolDefinition(
                name=tool.function.name,
                description=tool.function.description,
                parameters=parsed_parameters,
            )
        )

    try:
        trajectory = CanonicalTrajectory(
            trajectory_id=record.session_id,
            source_dataset="valiere/ISETrace",
            source_revision=source_revision,
            source_record_id=record.session_id,
            intents=tuple(
                SourceIntent(
                    intent_id=intent.intent_id,
                    text=intent.natural_language_intent,
                    task_type=intent.task_type,
                )
                for intent in record.source_intents
            ),
            tool_definitions=tuple(tool_definitions),
            events=tuple(events),
            final_response=record.final_output,
            source_total_steps=record.total_steps,
            source_metadata={
                "status": record.status,
                "intent_id": record.intent_id,
                "session_finalized_by_intent_id": (
                    record.session_finalized_by_intent_id
                ),
                "enable_thinking": record.enable_thinking,
                "metadata": record.metadata,
            },
        )
    except ValueError as error:
        raise ISETraceAdaptationError(
            "canonical_invariant",
            record.session_id,
            str(error),
        ) from error

    if active_summary is not None:
        active_summary.tool_calls += call_count
        active_summary.tool_outputs += output_count
    return trajectory


def iter_canonical_trajectories(
    paths: str | Path | Sequence[str | Path],
    *,
    source_revision: str,
    strict: bool = True,
    summary: ISETraceIngestionSummary | None = None,
) -> Iterator[CanonicalTrajectory]:
    active_summary = summary or ISETraceIngestionSummary()
    sources = (paths,) if isinstance(paths, (str, Path)) else paths
    for source in sources:
        for path in _trajectory_source_files(Path(source)):
            yield from _iter_canonical_trajectory_file(
                path,
                source_revision=source_revision,
                strict=strict,
                summary=active_summary,
            )


def _trajectory_source_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(f"ISETrace trajectory source does not exist: {source}")
    root = source / "trajectories"
    if not root.is_dir():
        root = source
    files = tuple(sorted(path for path in root.rglob("*.jsonl") if path.is_file()))
    if not files:
        raise FileNotFoundError(
            f"ISETrace trajectory directory contains no JSONL shards: {source}"
        )
    return files


def _iter_canonical_trajectory_file(
    path: Path,
    *,
    source_revision: str,
    strict: bool,
    summary: ISETraceIngestionSummary,
) -> Iterator[CanonicalTrajectory]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            summary.records_seen += 1
            try:
                record = _RECORD_ADAPTER.validate_json(line)
                trajectory = adapt_isetrace_record(
                    record,
                    source_revision=source_revision,
                    summary=summary,
                )
            except ValidationError as error:
                reason = (
                    "invalid_json"
                    if any(issue["type"] == "json_invalid" for issue in error.errors())
                    else "raw_schema"
                )
                summary.reject(reason)
                if strict:
                    raise ValueError(
                        f"invalid ISETrace input at {path}:{line_number}: {error}"
                    ) from error
                continue
            except ISETraceAdaptationError as error:
                summary.reject(error.code)
                if strict:
                    raise
                continue
            summary.records_accepted += 1
            yield trajectory


__all__ = [
    "ISETraceAdaptationError",
    "ISETraceIngestionSummary",
    "adapt_isetrace_record",
    "iter_canonical_trajectories",
    "iter_isetrace_records",
    "parse_isetrace_record",
]
