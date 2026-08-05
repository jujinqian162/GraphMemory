from __future__ import annotations

from graph_memory.graphs.contracts import GraphEdge
from graph_memory.retrieval.contracts import NativeRetrievalTrace, RankedNode
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import RankingMethodRequest, TextCandidate
from graph_memory.retrieval.results import (
    RankedNodeRecord,
    RankedResult,
    RankedResultMetadata,
    RetrievedSubgraph,
)
from graph_memory.text.tokens import content_tokens


def assemble_ranked_result(
    *,
    request: RankingMethodRequest,
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
    candidate_by_id = {
        candidate.item_id: candidate for candidate in request.candidates
    }
    ranked_node_ids = tuple(item.node_id for item in ranked_nodes)
    observed = set(ranked_node_ids)
    valid = frozenset(candidate_by_id)
    if observed != valid or len(ranked_node_ids) != len(valid):
        missing = sorted(valid - observed)
        extra = sorted(observed - valid)
        raise ValueError(
            "ranking must include every candidate exactly once; "
            f"missing={missing} extra={extra}"
        )
    if native_trace is not None:
        native_trace.validate_candidate_context(valid)

    return RankedResult(
        task_id=request.task_id,
        method=RetrievalMethodId(method),
        ranked_nodes=tuple(
            RankedNodeRecord(
                node_id=item.node_id,
                score=item.score,
                source_spans=candidate_by_id[item.node_id].source_spans,
                token_count=_candidate_token_count(candidate_by_id[item.node_id]),
            )
            for item in ranked_nodes
        ),
        retrieved_subgraph=RetrievedSubgraph(
            nodes=top_node_ids,
            edges=tuple(retrieved_edges),
        ),
        latency_ms=latency_ms,
        input_tokens=_approx_input_tokens(request),
        metadata=(
            RankedResultMetadata(native_trace=native_trace)
            if native_trace is not None
            else None
        ),
    )


def _candidate_token_count(candidate: TextCandidate) -> int:
    value = candidate.metadata.get("token_count")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return len(content_tokens(candidate.text))


def _approx_input_tokens(request: RankingMethodRequest) -> int:
    query_tokens = content_tokens(request.query_text)
    memory_tokens = [
        token
        for candidate in request.candidates
        for token in content_tokens(candidate.text)
    ]
    return len(query_tokens) + len(memory_tokens)


__all__ = ["assemble_ranked_result"]
