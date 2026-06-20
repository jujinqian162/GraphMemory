from __future__ import annotations

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.text.tokens import content_tokens


def assemble_ranked_result(
    *,
    text_request: TextRankingRequest,
    method: str,
    ranked_nodes: list[RankedNode],
    top_k: int,
    latency_ms: float,
    retrieved_edges: list[GraphEdge],
) -> RankedResult:
    top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
    return {
        "task_id": text_request.task_id,
        "method": method,
        "ranked_nodes": [
            {"node_id": ranked_node.node_id, "score": ranked_node.score}
            for ranked_node in ranked_nodes
        ],
        "retrieved_subgraph": {
            "nodes": top_node_ids,
            "edges": retrieved_edges,
        },
        "latency_ms": latency_ms,
        "input_tokens": _approx_input_tokens(text_request),
    }


def _approx_input_tokens(text_request: TextRankingRequest) -> int:
    query_tokens = content_tokens(text_request.query_text)
    memory_tokens = [token for candidate in text_request.candidates for token in content_tokens(candidate.text)]
    return len(query_tokens) + len(memory_tokens)
