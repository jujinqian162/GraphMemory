from __future__ import annotations

from graph_memory.graphs.contracts import GraphEdge
from graph_memory.retrieval.contracts import NativeRetrievalTrace, RankedNode
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.results import (
    RankedNodeRecord,
    RankedResult,
    RankedResultMetadata,
    RetrievedSubgraph,
)
from graph_memory.text.tokens import content_tokens


def assemble_ranked_result(
    *,
    text_request: TextRankingRequest,
    method: str,
    ranked_nodes: list[RankedNode] | tuple[RankedNode, ...],
    top_k: int,
    latency_ms: float,
    retrieved_edges: list[GraphEdge] | tuple[GraphEdge, ...],
    native_trace: NativeRetrievalTrace | None,
) -> RankedResult:
    top_node_ids = tuple(
        ranked_node.node_id for ranked_node in ranked_nodes[:top_k]
    )
    return RankedResult(
        task_id=text_request.task_id,
        method=RetrievalMethodId(method),
        ranked_nodes=tuple(
            RankedNodeRecord(node_id=item.node_id, score=item.score)
            for item in ranked_nodes
        ),
        retrieved_subgraph=RetrievedSubgraph(
            nodes=top_node_ids,
            edges=tuple(retrieved_edges),
        ),
        latency_ms=latency_ms,
        input_tokens=_approx_input_tokens(text_request),
        metadata=(
            RankedResultMetadata(native_trace=native_trace)
            if native_trace is not None
            else None
        ),
    )


def _approx_input_tokens(text_request: TextRankingRequest) -> int:
    query_tokens = content_tokens(text_request.query_text)
    memory_tokens = [
        token
        for candidate in text_request.candidates
        for token in content_tokens(candidate.text)
    ]
    return len(query_tokens) + len(memory_tokens)


__all__ = ["assemble_ranked_result"]
