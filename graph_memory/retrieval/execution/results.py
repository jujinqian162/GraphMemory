from __future__ import annotations

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.contracts import (
    GraphRAGTrace,
    NativeRetrievalTrace,
    RankedNode,
)
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
    native_trace: NativeRetrievalTrace | None,
) -> RankedResult:
    top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
    result: RankedResult = {
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
    if native_trace is not None:
        result["metadata"] = {
            "native_trace": _native_trace_record(native_trace)
        }
    return result


def _approx_input_tokens(text_request: TextRankingRequest) -> int:
    query_tokens = content_tokens(text_request.query_text)
    memory_tokens = [
        token
        for candidate in text_request.candidates
        for token in content_tokens(candidate.text)
    ]
    return len(query_tokens) + len(memory_tokens)


def _native_trace_record(trace: NativeRetrievalTrace) -> dict[str, object]:
    if isinstance(trace, GraphRAGTrace):
        return {
            "trace_kind": trace.trace_kind,
            "entity_ids": list(trace.entity_ids),
            "linked_entity_ids": list(trace.linked_entity_ids),
            "seed_entity_ids": list(trace.seed_entity_ids),
            "relations": [
                {
                    "source_entity_id": relation.source_entity_id,
                    "target_entity_id": relation.target_entity_id,
                    "weight": relation.weight,
                    "candidate_ids": list(relation.candidate_ids),
                }
                for relation in trace.relations
            ],
        }
    return {
        "trace_kind": trace.trace_kind,
        "node_ids": list(trace.node_ids),
        "paths": [
            {
                "node_ids": list(path.node_ids),
                "score": path.score,
                "semantic_relevance": path.semantic_relevance,
                "binding_consistency": path.binding_consistency,
                "provenance_completeness": path.provenance_completeness,
                "explicit_grounding": path.explicit_grounding,
                "path_length_penalty": path.path_length_penalty,
                "invalidation_penalty": path.invalidation_penalty,
            }
            for path in trace.paths
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type.value,
                "weight": edge.weight,
                **(
                    {
                        "binding": {
                            "output_field": edge.binding.output_field,
                            "input_parameter": edge.binding.input_parameter,
                            "binding_kind": edge.binding.binding_kind,
                        }
                    }
                    if edge.binding is not None
                    else {}
                ),
            }
            for edge in trace.edges
        ],
    }
