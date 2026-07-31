from __future__ import annotations

import hashlib
from dataclasses import dataclass

from graph_memory.graphs.provenance import (
    ARGUMENT_CHUNK_NODE,
    OUTPUT_CHUNK_NODE,
    ProvenanceGraph,
)
from graph_memory.retrieval.requests import TextCandidate
from graph_memory.text.chunking import OffsetTokenizer, token_chunks
from graph_memory.trajectories import (
    CanonicalTrajectory,
    MessageEvent,
    SourceSpan,
    ToolCallEvent,
    ToolOutputEvent,
)


@dataclass(frozen=True)
class _SourceFragment:
    document_start: int
    document_end: int
    event_id: str
    json_pointer: str
    source_start: int


class _TrajectoryDocumentBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._fragments: list[_SourceFragment] = []
        self._length = 0

    def append_literal(self, text: str) -> None:
        self._parts.append(text)
        self._length += len(text)

    def append_source(
        self,
        text: str,
        *,
        event_id: str,
        json_pointer: str,
    ) -> None:
        start = self._length
        self.append_literal(text)
        if text:
            self._fragments.append(
                _SourceFragment(
                    document_start=start,
                    document_end=self._length,
                    event_id=event_id,
                    json_pointer=json_pointer,
                    source_start=0,
                )
            )

    def build(self) -> tuple[str, tuple[_SourceFragment, ...]]:
        return "".join(self._parts), tuple(self._fragments)


def flat_trajectory_candidates(
    trajectory: CanonicalTrajectory,
    *,
    tokenizer: OffsetTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[TextCandidate, ...]:
    document, fragments = _render_trajectory(trajectory)
    chunks = token_chunks(
        document,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    candidates: list[TextCandidate] = []
    for index, chunk in enumerate(chunks):
        source_spans = _chunk_source_spans(
            chunk_start=chunk.char_start,
            chunk_end=chunk.char_end,
            fragments=fragments,
        )
        if not source_spans:
            continue
        identity = "\0".join(
            (
                trajectory.trajectory_id,
                str(index),
                str(chunk.char_start),
                str(chunk.char_end),
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        candidates.append(
            TextCandidate(
                item_id=f"flat-chunk:{digest}",
                text=chunk.text,
                metadata={
                    "graph_id": trajectory.trajectory_id,
                    "representation": "flat_trajectory_chunk",
                    "chunk_index": index,
                    "document_char_start": chunk.char_start,
                    "document_char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                },
                source_spans=source_spans,
            )
        )
    if not candidates:
        raise ValueError(
            f"trajectory={trajectory.trajectory_id} produced no flat retrieval chunks"
        )
    return tuple(candidates)


def provenance_unit_candidates(graph: ProvenanceGraph) -> tuple[TextCandidate, ...]:
    candidates = tuple(
        TextCandidate(
            item_id=node.node_id,
            text=node.text,
            metadata={
                "graph_id": graph.graph_id,
                "representation": "execution_provenance_unit",
                "node_kind": node.kind,
                **{
                    key: value
                    for key in (
                        "call_id",
                        "tool_name",
                        "chunk_index",
                        "token_count",
                        "message_index",
                        "sub_index",
                    )
                    if isinstance(
                        (value := (node.attributes or {}).get(key)),
                        (str, int, float, bool),
                    )
                },
            },
            source_spans=node.source_spans,
        )
        for node in graph.nodes
        if node.kind in {ARGUMENT_CHUNK_NODE, OUTPUT_CHUNK_NODE}
    )
    if not candidates:
        raise ValueError(f"graph={graph.graph_id} has no provenance retrieval units")
    return candidates


def _render_trajectory(
    trajectory: CanonicalTrajectory,
) -> tuple[str, tuple[_SourceFragment, ...]]:
    builder = _TrajectoryDocumentBuilder()
    for event in trajectory.events:
        if isinstance(event, MessageEvent):
            builder.append_literal(
                f"[message role={event.role} position={event.position.message_index}:"
                f"{event.position.sub_index}]\ncontent="
            )
            builder.append_source(
                event.content,
                event_id=event.event_id,
                json_pointer="/content",
            )
            if event.reasoning_content:
                builder.append_literal("\nreasoning=")
                builder.append_source(
                    event.reasoning_content,
                    event_id=event.event_id,
                    json_pointer="/reasoning_content",
                )
        elif isinstance(event, ToolCallEvent):
            builder.append_literal(
                f"[tool_call tool={event.tool_name} position="
                f"{event.position.message_index}:{event.position.sub_index}]\narguments="
            )
            builder.append_source(
                event.raw_arguments,
                event_id=event.event_id,
                json_pointer="/raw_arguments",
            )
        elif isinstance(event, ToolOutputEvent):
            builder.append_literal(
                f"[tool_output tool={event.tool_name} position="
                f"{event.position.message_index}:{event.position.sub_index}]\noutput="
            )
            builder.append_source(
                event.content,
                event_id=event.event_id,
                json_pointer="/content",
            )
        builder.append_literal("\n\n")
    return builder.build()


def _chunk_source_spans(
    *,
    chunk_start: int,
    chunk_end: int,
    fragments: tuple[_SourceFragment, ...],
) -> tuple[SourceSpan, ...]:
    spans: list[SourceSpan] = []
    for fragment in fragments:
        overlap_start = max(chunk_start, fragment.document_start)
        overlap_end = min(chunk_end, fragment.document_end)
        if overlap_end <= overlap_start:
            continue
        spans.append(
            SourceSpan(
                event_id=fragment.event_id,
                char_start=(
                    fragment.source_start + overlap_start - fragment.document_start
                ),
                char_end=(
                    fragment.source_start + overlap_end - fragment.document_start
                ),
                json_pointer=fragment.json_pointer,
            )
        )
    return tuple(spans)


__all__ = ["flat_trajectory_candidates", "provenance_unit_candidates"]
