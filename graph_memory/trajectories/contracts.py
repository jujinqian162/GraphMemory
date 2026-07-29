from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr, NonNegativeInt


class SourceIntent(DomainModel):
    intent_id: NonEmptyStr
    text: NonEmptyStr
    task_type: NonEmptyStr | None = None
    metadata: dict[str, JsonValue] | None = None


class ToolDefinition(DomainModel):
    name: NonEmptyStr
    description: str
    parameters: dict[str, JsonValue]


class SourcePosition(DomainModel):
    message_index: NonNegativeInt
    sub_index: NonNegativeInt = 0


class SourceSpan(DomainModel):
    event_id: NonEmptyStr
    char_start: NonNegativeInt | None = None
    char_end: NonNegativeInt | None = None
    json_pointer: str | None = None

    @model_validator(mode="after")
    def _validate_offsets(self) -> "SourceSpan":
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("source span offsets must be both present or both absent")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("source span char_end must be greater than char_start")
        return self


class MessageEvent(DomainModel):
    event_type: Literal["message"] = "message"
    event_id: NonEmptyStr
    position: SourcePosition
    role: Literal["system", "user", "assistant"]
    content: str
    reasoning_content: str | None = None


class ToolCallEvent(DomainModel):
    event_type: Literal["tool_call"] = "tool_call"
    event_id: NonEmptyStr
    position: SourcePosition
    call_id: NonEmptyStr
    tool_name: NonEmptyStr
    arguments: dict[str, JsonValue]
    raw_arguments: str


class ToolOutputEvent(DomainModel):
    event_type: Literal["tool_output"] = "tool_output"
    event_id: NonEmptyStr
    position: SourcePosition
    call_id: NonEmptyStr
    tool_name: NonEmptyStr
    content: str
    source_reported_success: bool | None = None


TrajectoryEvent: TypeAlias = Annotated[
    MessageEvent | ToolCallEvent | ToolOutputEvent,
    Field(discriminator="event_type"),
]


class CanonicalTrajectory(DomainModel):
    schema_version: Literal[1] = 1
    trajectory_id: NonEmptyStr
    source_dataset: NonEmptyStr
    source_revision: NonEmptyStr
    source_record_id: NonEmptyStr
    intents: tuple[SourceIntent, ...] = Field(min_length=1)
    tool_definitions: tuple[ToolDefinition, ...]
    events: tuple[TrajectoryEvent, ...] = Field(min_length=1)
    final_response: str
    source_total_steps: NonNegativeInt
    source_metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_trajectory(self) -> "CanonicalTrajectory":
        intent_ids = [intent.intent_id for intent in self.intents]
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("trajectory intent IDs must be unique")

        tool_names = [tool.name for tool in self.tool_definitions]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("trajectory tool definition names must be unique")

        positions = [
            (event.position.message_index, event.position.sub_index)
            for event in self.events
        ]
        if positions != sorted(positions) or len(positions) != len(set(positions)):
            raise ValueError("trajectory events must have unique increasing positions")

        calls: dict[str, ToolCallEvent] = {}
        outputs: dict[str, ToolOutputEvent] = {}
        for event in self.events:
            if isinstance(event, ToolCallEvent):
                if event.call_id in calls:
                    raise ValueError(f"duplicate tool call ID: {event.call_id}")
                calls[event.call_id] = event
            elif isinstance(event, ToolOutputEvent):
                if event.call_id in outputs:
                    raise ValueError(f"duplicate tool output for call ID: {event.call_id}")
                outputs[event.call_id] = event

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("trajectory event IDs must be unique")
        missing = sorted(set(calls) - set(outputs))
        orphaned = sorted(set(outputs) - set(calls))
        if missing or orphaned:
            raise ValueError(
                f"tool call/output pairing mismatch; missing={missing} orphaned={orphaned}"
            )
        if self.source_total_steps != len(calls):
            detail = (
                f"source_total_steps={self.source_total_steps} does not match tool_call_count={len(calls)}"
            )
            raise ValueError(detail)
        for call_id, call in calls.items():
            output = outputs[call_id]
            if call.tool_name != output.tool_name:
                detail = (
                    f"call_id={call_id} tool mismatch: {call.tool_name!r} != {output.tool_name!r}"
                )
                raise ValueError(detail)
            call_position = (call.position.message_index, call.position.sub_index)
            output_position = (
                output.position.message_index,
                output.position.sub_index,
            )
            if output_position <= call_position:
                raise ValueError(f"call_id={call_id} output must follow its call")
        return self

    @property
    def tool_calls(self) -> tuple[ToolCallEvent, ...]:
        return tuple(
            event for event in self.events if isinstance(event, ToolCallEvent)
        )

    @property
    def tool_outputs(self) -> tuple[ToolOutputEvent, ...]:
        return tuple(
            event for event in self.events if isinstance(event, ToolOutputEvent)
        )

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "CanonicalTrajectory",
    "MessageEvent",
    "SourceIntent",
    "SourcePosition",
    "SourceSpan",
    "ToolCallEvent",
    "ToolDefinition",
    "ToolOutputEvent",
    "TrajectoryEvent",
]
