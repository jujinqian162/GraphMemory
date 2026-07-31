from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from pydantic import JsonValue

from graph_memory.graphs.provenance.contracts import (
    ARGUMENT_CHUNK_NODE,
    ARTIFACT_NODE,
    FEEDS_EDGE,
    HAS_ARGUMENT_EDGE,
    HAS_CONTENT_EDGE,
    NEXT_CHUNK_EDGE,
    OUTPUT_CHUNK_NODE,
    PRECEDES_EDGE,
    READS_EDGE,
    RETURNS_EDGE,
    TOOL_CALL_NODE,
    TOOL_OUTPUT_NODE,
    WRITES_EDGE,
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
)
from graph_memory.text.chunking import TokenChunk
from graph_memory.trajectories import (
    CanonicalTrajectory,
    SourceSpan,
    ToolCallEvent,
)

BUILDER_VERSION = "provenance-core-v2-content-units"
ContentChunker = Callable[[str], tuple[TokenChunk, ...]]
_TYPED_TOKEN_PATTERN = "".join(
    (
        r"https?://[^\s\]\[\)\}\>\"']+",
        r"|(?:/[A-Za-z0-9_.@%+,:=~-]+){2,}",
        r"|\b[0-9a-fA-F]{16,64}\b",
        r"|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-",
        r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
        r"|\b[A-Za-z0-9_.-]{3,}\.(?:jsonl?|md|txt|csv|ya?ml|py|sh|html|pdf|",
        r"mp3|wav|png|jpe?g)\b",
    )
)
_TYPED_TOKEN: re.Pattern[str] = re.compile(_TYPED_TOKEN_PATTERN)
_UUID_PATTERN = "".join(
    (
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-",
        r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
    )
)

AccessMode = Literal["reads", "writes"]


@dataclass(frozen=True)
class ArtifactAccess:
    artifact_kind: str
    value: str
    mode: AccessMode
    json_pointer: str


class ArtifactAccessExtractor(Protocol):
    def extract(self, call: ToolCallEvent) -> tuple[ArtifactAccess, ...]: ...


class ExplicitArtifactAccessExtractor:
    _TOOL_ARGUMENTS: Mapping[str, tuple[tuple[str, AccessMode, str], ...]] = {
        "read": (("path", "reads", "path"),),
        "write": (("path", "writes", "path"),),
        "edit": (
            ("path", "reads", "path"),
            ("path", "writes", "path"),
        ),
        "web_fetch": (("url", "reads", "url"),),
    }

    def extract(self, call: ToolCallEvent) -> tuple[ArtifactAccess, ...]:
        accesses: list[ArtifactAccess] = []
        for key, mode, artifact_kind in self._TOOL_ARGUMENTS.get(
            call.tool_name, ()
        ):
            raw_value = call.arguments.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                accesses.append(
                    ArtifactAccess(
                        artifact_kind=artifact_kind,
                        value=raw_value.strip(),
                        mode=mode,
                        json_pointer=f"/arguments/{key}",
                    )
                )
        return tuple(accesses)


def _short_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _edge_id(relation: str, source: str, target: str) -> str:
    identity = "\0".join((relation, source, target))
    return f"edge:{_short_hash(identity)}"


def _call_node_id(call_id: str) -> str:
    return f"call:{call_id}"


def _output_node_id(call_id: str) -> str:
    return f"output:{call_id}"


def _argument_chunk_node_id(call_id: str, index: int) -> str:
    return f"argument:{call_id}:chunk:{index}"


def _output_chunk_node_id(call_id: str, index: int) -> str:
    return f"output-content:{call_id}:chunk:{index}"


def _artifact_node_id(kind: str, value: str) -> str:
    return f"artifact:{kind}:{_short_hash(value)}"


def _source_span(
    event_id: str,
    *,
    json_pointer: str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
) -> SourceSpan:
    return SourceSpan(
        event_id=event_id,
        json_pointer=json_pointer,
        char_start=char_start,
        char_end=char_end,
    )


def _whole_text_chunker(text: str) -> tuple[TokenChunk, ...]:
    if not text:
        return ()
    return (
        TokenChunk(
            text=text,
            char_start=0,
            char_end=len(text),
            token_count=0,
        ),
    )


def _append_next_chunk_edges(
    edges: list[ProvenanceEdge],
    node_ids: list[str],
    *,
    event_id: str,
    extractor: str,
) -> None:
    for source, target in zip(node_ids, node_ids[1:], strict=False):
        edges.append(
            ProvenanceEdge(
                edge_id=_edge_id(NEXT_CHUNK_EDGE, source, target),
                relation=NEXT_CHUNK_EDGE,
                source=source,
                target=target,
                derivation="deterministic",
                extractor=extractor,
                source_spans=(_source_span(event_id),),
            )
        )


def _iter_string_leaves(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        for child in mapping.values():
            yield from _iter_string_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _iter_string_leaves(child)


def _token_kind(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return "url"
    if value.startswith("/"):
        return "path"
    if re.fullmatch(_UUID_PATTERN, value):
        return "uuid"
    if re.fullmatch(r"[0-9a-fA-F]{16,64}", value):
        return "hash"
    return "filename"


def _typed_tokens(text: str) -> set[tuple[str, str]]:
    tokens: set[tuple[str, str]] = set()
    try:
        parsed = cast(object, json.loads(text))
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        for leaf in _iter_string_leaves(parsed):
            for token_match in _TYPED_TOKEN.finditer(leaf):
                normalized = token_match.group(0).rstrip(".,;:")
                tokens.add((_token_kind(normalized), normalized))
    for token_match in _TYPED_TOKEN.finditer(text):
        normalized = token_match.group(0).rstrip(".,;:")
        tokens.add((_token_kind(normalized), normalized))
    return tokens


def _argument_contains(call: ToolCallEvent, value: str) -> bool:
    if value in call.raw_arguments:
        return True
    return any(value in leaf for leaf in _iter_string_leaves(call.arguments))


def build_provenance_graph(
    trajectory: CanonicalTrajectory,
    *,
    artifact_extractors: tuple[ArtifactAccessExtractor, ...] | None = None,
    content_chunker: ContentChunker | None = None,
) -> ProvenanceGraph:
    extractors = artifact_extractors or (ExplicitArtifactAccessExtractor(),)
    chunk_content = content_chunker or _whole_text_chunker
    calls = list(trajectory.tool_calls)
    outputs = list(trajectory.tool_outputs)
    call_by_id = {call.call_id: call for call in calls}
    output_by_call_id = {output.call_id: output for output in outputs}

    nodes: list[ProvenanceNode] = []
    edges: list[ProvenanceEdge] = []
    for call in calls:
        call_node_id = _call_node_id(call.call_id)
        output_node_id = _output_node_id(call.call_id)
        nodes.append(
            ProvenanceNode(
                node_id=call_node_id,
                kind=TOOL_CALL_NODE,
                text=f"tool={call.tool_name}",
                source_spans=(_source_span(call.event_id),),
                attributes=cast(
                    dict[str, JsonValue],
                    {
                        "call_id": call.call_id,
                        "tool_name": call.tool_name,
                        "argument_keys": sorted(call.arguments),
                        "message_index": call.position.message_index,
                        "sub_index": call.position.sub_index,
                    },
                ),
            )
        )
        argument_node_ids: list[str] = []
        for index, chunk in enumerate(chunk_content(call.raw_arguments)):
            argument_node_id = _argument_chunk_node_id(call.call_id, index)
            argument_node_ids.append(argument_node_id)
            span = _source_span(
                call.event_id,
                json_pointer="/raw_arguments",
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            nodes.append(
                ProvenanceNode(
                    node_id=argument_node_id,
                    kind=ARGUMENT_CHUNK_NODE,
                    text=chunk.text,
                    source_spans=(span,),
                    attributes={
                        "call_id": call.call_id,
                        "tool_name": call.tool_name,
                        "chunk_index": index,
                        "token_count": chunk.token_count,
                        "message_index": call.position.message_index,
                        "sub_index": call.position.sub_index,
                    },
                )
            )
            edges.append(
                ProvenanceEdge(
                    edge_id=_edge_id(HAS_ARGUMENT_EDGE, call_node_id, argument_node_id),
                    relation=HAS_ARGUMENT_EDGE,
                    source=call_node_id,
                    target=argument_node_id,
                    derivation="native",
                    extractor="tool_call_arguments",
                    source_spans=(span,),
                )
            )
        _append_next_chunk_edges(
            edges,
            argument_node_ids,
            event_id=call.event_id,
            extractor="argument_chunk_order",
        )

        output = output_by_call_id[call.call_id]
        nodes.append(
            ProvenanceNode(
                node_id=output_node_id,
                kind=TOOL_OUTPUT_NODE,
                text=f"tool_output={output.tool_name}",
                source_spans=(_source_span(output.event_id),),
                attributes={
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "source_reported_success": output.source_reported_success,
                    "message_index": output.position.message_index,
                    "sub_index": output.position.sub_index,
                },
            )
        )
        output_chunk_node_ids: list[str] = []
        for index, chunk in enumerate(chunk_content(output.content)):
            output_chunk_node_id = _output_chunk_node_id(call.call_id, index)
            output_chunk_node_ids.append(output_chunk_node_id)
            span = _source_span(
                output.event_id,
                json_pointer="/content",
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            nodes.append(
                ProvenanceNode(
                    node_id=output_chunk_node_id,
                    kind=OUTPUT_CHUNK_NODE,
                    text=chunk.text,
                    source_spans=(span,),
                    attributes={
                        "call_id": call.call_id,
                        "tool_name": call.tool_name,
                        "chunk_index": index,
                        "token_count": chunk.token_count,
                        "message_index": output.position.message_index,
                        "sub_index": output.position.sub_index,
                    },
                )
            )
            edges.append(
                ProvenanceEdge(
                    edge_id=_edge_id(HAS_CONTENT_EDGE, output_node_id, output_chunk_node_id),
                    relation=HAS_CONTENT_EDGE,
                    source=output_node_id,
                    target=output_chunk_node_id,
                    derivation="native",
                    extractor="tool_output_content",
                    source_spans=(span,),
                )
            )
        _append_next_chunk_edges(
            edges,
            output_chunk_node_ids,
            event_id=output.event_id,
            extractor="output_chunk_order",
        )
        edges.append(
            ProvenanceEdge(
                edge_id=_edge_id(RETURNS_EDGE, call_node_id, output_node_id),
                relation=RETURNS_EDGE,
                source=call_node_id,
                target=output_node_id,
                derivation="native",
                extractor="tool_call_id",
                source_spans=(
                    _source_span(call.event_id),
                    _source_span(output.event_id),
                ),
            )
        )

    for earlier, later in zip(calls, calls[1:], strict=False):
        edges.append(
            ProvenanceEdge(
                edge_id=_edge_id(
                    PRECEDES_EDGE,
                    _call_node_id(earlier.call_id),
                    _call_node_id(later.call_id),
                ),
                relation=PRECEDES_EDGE,
                source=_call_node_id(earlier.call_id),
                target=_call_node_id(later.call_id),
                derivation="native",
                extractor="canonical_event_order",
                source_spans=(
                    _source_span(earlier.event_id),
                    _source_span(later.event_id),
                ),
            )
        )

    artifact_spans: dict[str, list[SourceSpan]] = defaultdict(list)
    artifact_values: dict[str, tuple[str, str]] = {}
    resource_edge_keys: set[tuple[str, str, str]] = set()
    for call in calls:
        for extractor in extractors:
            for access in extractor.extract(call):
                artifact_id = _artifact_node_id(
                    access.artifact_kind, access.value
                )
                artifact_values[artifact_id] = (
                    access.artifact_kind,
                    access.value,
                )
                span = _source_span(
                    call.event_id, json_pointer=access.json_pointer
                )
                if span not in artifact_spans[artifact_id]:
                    artifact_spans[artifact_id].append(span)
                relation = READS_EDGE if access.mode == "reads" else WRITES_EDGE
                key = (relation, _call_node_id(call.call_id), artifact_id)
                if key in resource_edge_keys:
                    continue
                resource_edge_keys.add(key)
                edges.append(
                    ProvenanceEdge(
                        edge_id=_edge_id(*key),
                        relation=relation,
                        source=key[1],
                        target=artifact_id,
                        derivation="deterministic",
                        extractor=type(extractor).__name__,
                        source_spans=(span,),
                        attributes={
                            "artifact_kind": access.artifact_kind,
                        },
                    )
                )
    for artifact_id in sorted(artifact_values):
        artifact_kind, value = artifact_values[artifact_id]
        nodes.append(
            ProvenanceNode(
                node_id=artifact_id,
                kind=ARTIFACT_NODE,
                text=value,
                source_spans=tuple(artifact_spans[artifact_id]),
                attributes={
                    "artifact_kind": artifact_kind,
                    "canonical_value": value,
                },
            )
        )

    producers: dict[str, set[str]] = defaultdict(set)
    token_kinds: dict[str, str] = {}
    output_position: dict[str, tuple[int, int]] = {}
    for output in outputs:
        output_id = _output_node_id(output.call_id)
        output_position[output_id] = (
            output.position.message_index,
            output.position.sub_index,
        )
        for token_kind, value in _typed_tokens(output.content):
            producers[value].add(output_id)
            token_kinds[value] = token_kind

    bindings_by_pair: dict[tuple[str, str], list[dict[str, JsonValue]]] = defaultdict(list)
    for value, producer_ids in producers.items():
        if len(producer_ids) != 1:
            continue
        producer_id = next(iter(producer_ids))
        for call in calls:
            call_position = (call.position.message_index, call.position.sub_index)
            if call_position <= output_position[producer_id]:
                continue
            if not _argument_contains(call, value):
                continue
            bindings_by_pair[(producer_id, _call_node_id(call.call_id))].append(
                {
                    "binding_type": token_kinds[value],
                    "value_hash": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                }
            )

    for (source, target), bindings in sorted(bindings_by_pair.items()):
        producer_call_id = source.removeprefix("output:")
        consumer_call_id = target.removeprefix("call:")
        producer_output = output_by_call_id[producer_call_id]
        consumer_call = call_by_id[consumer_call_id]
        unique_bindings = sorted(
            {
                (str(binding["binding_type"]), str(binding["value_hash"]))
                for binding in bindings
            }
        )
        edges.append(
            ProvenanceEdge(
                edge_id=_edge_id(FEEDS_EDGE, source, target),
                relation=FEEDS_EDGE,
                source=source,
                target=target,
                derivation="deterministic",
                extractor="exact_typed_binding_v1",
                source_spans=(
                    _source_span(producer_output.event_id),
                    _source_span(consumer_call.event_id),
                ),
                attributes={
                    "bindings": [
                        {"binding_type": kind, "value_hash": value_hash}
                        for kind, value_hash in unique_bindings
                    ]
                },
            )
        )

    return ProvenanceGraph(
        graph_id=trajectory.trajectory_id,
        trajectory_fingerprint=trajectory.fingerprint(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "builder_version": BUILDER_VERSION,
            "source_dataset": trajectory.source_dataset,
            "source_revision": trajectory.source_revision,
        },
    )


__all__ = [
    "ArtifactAccess",
    "ArtifactAccessExtractor",
    "BUILDER_VERSION",
    "ExplicitArtifactAccessExtractor",
    "build_provenance_graph",
]
