from __future__ import annotations

from graph_memory.graphs.provenance.contracts import (
    HAS_CONTENT_EDGE,
    OUTPUT_CHUNK_NODE,
    ProvenanceGraph,
    ProvenanceNode,
)
from graph_memory.trajectories import SourceSpan


def output_source_spans(
    graph: ProvenanceGraph, output_node_ids: tuple[str, ...]
) -> tuple[SourceSpan, ...]:
    node_by_id = graph.node_by_id
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    selected = set(output_node_ids)
    for edge in graph.edges:
        if edge.relation != HAS_CONTENT_EDGE or edge.source not in selected:
            continue
        node = node_by_id[edge.target]
        if node.kind != OUTPUT_CHUNK_NODE:
            raise ValueError(f"output={edge.source} has a non-output content child")
        for span in node.source_spans:
            if (
                span.json_pointer is None
                or span.char_start is None
                or span.char_end is None
            ):
                raise ValueError("output content chunks require exact source spans")
            grouped.setdefault((span.event_id, span.json_pointer), []).append(
                (span.char_start, span.char_end)
            )
    spans: list[SourceSpan] = []
    for (event_id, json_pointer), intervals in sorted(grouped.items()):
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        spans.extend(
            SourceSpan(
                event_id=event_id,
                json_pointer=json_pointer,
                char_start=start,
                char_end=end,
            )
            for start, end in merged
        )
    if not spans:
        raise ValueError(f"outputs={sorted(selected)} have no source-backed content")
    return tuple(spans)


def output_content(graph: ProvenanceGraph, output_node_id: str) -> str:
    node_by_id = graph.node_by_id
    chunk_ids = [
        edge.target
        for edge in graph.edges
        if edge.relation == HAS_CONTENT_EDGE and edge.source == output_node_id
    ]
    chunks = [node_by_id[node_id] for node_id in chunk_ids]
    if any(node.kind != OUTPUT_CHUNK_NODE for node in chunks):
        raise ValueError(f"output={output_node_id} has a non-output content child")

    def chunk_index(node: ProvenanceNode) -> int:
        value = (node.attributes or {}).get("chunk_index")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    chunks.sort(key=chunk_index)
    if not chunks:
        raise ValueError(f"output={output_node_id} has no content chunks")
    text = chunks[0].text
    for previous, current in zip(chunks, chunks[1:], strict=False):
        previous_span = previous.source_spans[0]
        current_span = current.source_spans[0]
        if previous_span.char_end is None or current_span.char_start is None:
            raise ValueError("output content chunks require exact source spans")
        overlap = max(0, previous_span.char_end - current_span.char_start)
        text += current.text[overlap:]
    return text


__all__ = ["output_content", "output_source_spans"]
