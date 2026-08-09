from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field, StrictStr, StringConstraints, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.provenance import (
    TOOL_CALL_NODE,
    TOOL_OUTPUT_NODE,
    ProvenanceGraph,
    output_content,
)
from graph_memory.query_synthesis.provenance.contracts import QueryIntent
from graph_memory.trajectories import CanonicalTrajectory, SourceSpan

_SOURCE_HANDLE = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[AE][1-9][0-9]*$"),
]
_TASK_HEADER = re.compile(
    r"(?m)^\[(?P<handle>[IAE][1-9][0-9]*) \| "
    r"(?P<kind>user_intent|tool_call|tool_output)"
    r"(?: \| [^\]\n]+)?\]\n"
)
_EXPECTED_PREFIX = {
    "user_intent": "I",
    "tool_call": "A",
    "tool_output": "E",
}
MemoryQueryMode: TypeAlias = Literal[
    "direct_recall",
    "linked_recall",
    "multi_fact_recall",
]


@dataclass(frozen=True)
class AuthoringSource:
    handle: str
    kind: str
    event_id: str | None
    json_pointer: str | None
    text: str


@dataclass(frozen=True)
class ResolvedAuthoringGold:
    source: str
    quote: str
    span: SourceSpan


class SourceQuote(Protocol):
    source: str
    quote: str


class AuthoringGold(DomainModel):
    source: _SOURCE_HANDLE
    quote: NonEmptyStr


class AuthoringQueryRecord(DomainModel):
    id: NonEmptyStr
    text: NonEmptyStr
    query: NonEmptyStr
    gold: tuple[AuthoringGold, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_gold(self) -> "AuthoringQueryRecord":
        sources = {source.handle: source for source in parse_task_sources(self.text)}
        keys = [(item.source, item.quote) for item in self.gold]
        if len(keys) != len(set(keys)):
            raise ValueError("authoring gold entries must be unique")
        for item in self.gold:
            source = sources.get(item.source)
            if source is None:
                raise ValueError(
                    f"authoring gold references missing source={item.source!r}"
                )
            unique_quote_start(source.text, item.quote, source=item.source)
        return self


class AuthoringQueryMetadataRecord(DomainModel):
    query_id: NonEmptyStr
    task_key: NonEmptyStr
    trajectory_id: NonEmptyStr
    memory_mode: MemoryQueryMode


def authoring_metadata_path(query_source: Path) -> Path:
    return query_source.with_suffix(query_source.suffix + ".metadata.jsonl")


def memory_mode_for_query_intent(query_intent: QueryIntent) -> MemoryQueryMode:
    if query_intent in {"call_result", "downstream_result"}:
        return "direct_recall"
    if query_intent in {"upstream_source", "artifact_origin", "artifact_use"}:
        return "linked_recall"
    return "multi_fact_recall"


def parse_task_sections(text: str) -> tuple[AuthoringSource, ...]:
    """Parse the complete handle-delimited v7 task text losslessly."""

    matches = list(_TASK_HEADER.finditer(text))
    if not matches or matches[0].start() != 0:
        raise ValueError("authoring text must start with an I/A/E section header")
    sections: list[AuthoringSource] = []
    for index, match in enumerate(matches):
        kind = match.group("kind")
        handle = match.group("handle")
        if not handle.startswith(_EXPECTED_PREFIX[kind]):
            raise ValueError(
                f"authoring source handle={handle!r} does not match kind={kind!r}"
            )
        if index + 1 < len(matches):
            body = text[match.end() : matches[index + 1].start()]
            if not body.endswith("\n\n"):
                raise ValueError(
                    "authoring text sections must be separated by a blank line"
                )
            body = body[:-2]
        else:
            body = text[match.end() :]
        sections.append(
            AuthoringSource(
                handle=handle,
                kind=kind,
                event_id=None,
                json_pointer=None,
                text=body,
            )
        )
    handles = [section.handle for section in sections]
    if len(handles) != len(set(handles)):
        raise ValueError("authoring text source handles must be unique")
    return tuple(sections)


def parse_task_sources(text: str) -> tuple[AuthoringSource, ...]:
    sources = tuple(
        section
        for section in parse_task_sections(text)
        if section.kind != "user_intent"
    )
    if not sources:
        raise ValueError("authoring text must contain at least one A/E source")
    return sources


def parse_task_intents(text: str) -> tuple[str, ...]:
    intents = tuple(
        section.text
        for section in parse_task_sections(text)
        if section.kind == "user_intent"
    )
    if not intents:
        raise ValueError("authoring text must contain at least one user intent")
    return intents


def unique_quote_start(text: str, quote: str, *, source: str) -> int:
    start = text.find(quote)
    occurrences = 0 if start < 0 else 1 + int(text.find(quote, start + 1) >= 0)
    if occurrences != 1:
        raise ValueError(
            f"authoring gold quote must occur exactly once in source={source!r}; "
            f"observed={occurrences if occurrences < 2 else 'multiple'}"
        )
    return start


def resolve_gold_quotes(
    gold: Sequence[SourceQuote],
    sources: Sequence[AuthoringSource],
) -> tuple[ResolvedAuthoringGold, ...]:
    source_by_handle = {source.handle: source for source in sources}
    resolved: list[ResolvedAuthoringGold] = []
    seen_spans: set[tuple[str, str, int, int]] = set()
    for item in gold:
        source = source_by_handle.get(item.source)
        if source is None:
            raise ValueError(f"gold references missing source={item.source!r}")
        if source.event_id is None or source.json_pointer is None:
            raise ValueError(f"source={item.source!r} is not resolved to a trajectory")
        start = unique_quote_start(source.text, item.quote, source=item.source)
        key = (
            source.event_id,
            source.json_pointer,
            start,
            start + len(item.quote),
        )
        if key in seen_spans:
            raise ValueError("gold entries resolve to duplicate source spans")
        seen_spans.add(key)
        resolved.append(
            ResolvedAuthoringGold(
                source=item.source,
                quote=item.quote,
                span=SourceSpan(
                    event_id=source.event_id,
                    json_pointer=source.json_pointer,
                    char_start=start,
                    char_end=start + len(item.quote),
                ),
            )
        )
    return tuple(resolved)


def source_aliases(
    graph: ProvenanceGraph,
    output_ids: Sequence[str],
) -> dict[str, str]:
    ordered_outputs = sorted(
        set(output_ids),
        key=lambda node_id: (_message_index(graph, node_id), node_id),
    )
    aliases: dict[str, str] = {}
    for index, output_id in enumerate(ordered_outputs, start=1):
        aliases[output_id] = f"E{index}"
        aliases[_call_node_id_for_output(graph, output_id)] = f"A{index}"
    return aliases


def source_material(
    trajectory: CanonicalTrajectory,
    graph: ProvenanceGraph,
    aliases: dict[str, str],
) -> dict[str, AuthoringSource]:
    call_by_id = {call.call_id: call for call in trajectory.tool_calls}
    material: dict[str, AuthoringSource] = {}
    for node_id, alias in aliases.items():
        node = graph.node_by_id[node_id]
        event_id = node.source_spans[0].event_id
        if node.kind == TOOL_OUTPUT_NODE:
            material[alias] = AuthoringSource(
                handle=alias,
                kind="tool_output",
                event_id=event_id,
                json_pointer="/content",
                text=output_content(graph, node_id),
            )
            continue
        if node.kind != TOOL_CALL_NODE:
            raise ValueError(f"unsupported authoring source node kind={node.kind}")
        call_id = (node.attributes or {}).get("call_id")
        if not isinstance(call_id, str) or call_id not in call_by_id:
            raise ValueError(f"call node={node_id} lacks canonical call content")
        material[alias] = AuthoringSource(
            handle=alias,
            kind="tool_call",
            event_id=event_id,
            json_pointer="/raw_arguments",
            text=call_by_id[call_id].raw_arguments,
        )
    return material


def render_task_text(
    trajectory: CanonicalTrajectory,
    graph: ProvenanceGraph,
    aliases: dict[str, str],
) -> str:
    material = source_material(trajectory, graph, aliases)
    sections = [
        f"[I{index} | user_intent]\n{intent.text}"
        for index, intent in enumerate(trajectory.intents, start=1)
    ]
    ordered = sorted(
        aliases.items(),
        key=lambda item: (_message_index(graph, item[0]), item[1]),
    )
    for node_id, alias in ordered:
        node = graph.node_by_id[node_id]
        event_kind = "tool_output" if node.kind == TOOL_OUTPUT_NODE else "tool_call"
        tool_name = str((node.attributes or {}).get("tool_name", "tool"))
        sections.append(
            f"[{alias} | {event_kind} | {tool_name}]\n{material[alias].text}"
        )
    return "\n\n".join(sections)


def _call_node_id_for_output(graph: ProvenanceGraph, output_node_id: str) -> str:
    output = graph.node_by_id[output_node_id]
    call_id = (output.attributes or {}).get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"output={output_node_id} lacks call_id metadata")
    call_node_id = f"call:{call_id}"
    if call_node_id not in graph.node_by_id:
        raise ValueError(f"output={output_node_id} references missing call node")
    return call_node_id


def _message_index(graph: ProvenanceGraph, node_id: str) -> int:
    value = (graph.node_by_id[node_id].attributes or {}).get("message_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "AuthoringGold",
    "AuthoringQueryMetadataRecord",
    "AuthoringQueryRecord",
    "AuthoringSource",
    "MemoryQueryMode",
    "ResolvedAuthoringGold",
    "authoring_metadata_path",
    "memory_mode_for_query_intent",
    "parse_task_intents",
    "parse_task_sections",
    "parse_task_sources",
    "render_task_text",
    "resolve_gold_quotes",
    "source_aliases",
    "source_material",
    "unique_quote_start",
]
