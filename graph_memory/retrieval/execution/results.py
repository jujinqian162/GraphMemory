from __future__ import annotations

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.contracts import (
    CandidateEdgeTrace,
    DenseRankTrace,
    ExecutionProvenanceTrace,
    GraphRAGTrace,
    NativeRetrievalTrace,
    ProvenanceEdgeTrace,
    QueryConditionedExecutionProvenanceTrace,
    RankedNode,
    StatelessExecutionProvenanceTrace,
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
        result["metadata"] = {"native_trace": _native_trace_record(native_trace)}
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
            "linked_entity_ids": list(trace.linked_entity_ids),
            "dense_ranks": [_dense_rank_record(item) for item in trace.dense_ranks],
            "seed_candidate_ids": list(trace.seed_candidate_ids),
            "mentions": [
                {
                    "candidate_id": mention.candidate_id,
                    "entity_id": mention.entity_id,
                    "mention_type": mention.mention_type,
                    "normalized_surface": mention.normalized_surface,
                    "source_prior": mention.source_prior,
                    "alias_confidence": mention.alias_confidence,
                    "entity_document_frequency": mention.entity_document_frequency,
                    "normalized_idf": mention.normalized_idf,
                    "mention_confidence": mention.mention_confidence,
                }
                for mention in trace.mentions
            ],
            "title_groups": [
                {
                    "entity_id": group.entity_id,
                    "normalized_title_entity": group.normalized_title_entity,
                    "candidate_ids": list(group.candidate_ids),
                    "group_size": group.group_size,
                    "entity_document_frequency": group.entity_document_frequency,
                    "document_frequency_ratio": group.document_frequency_ratio,
                }
                for group in trace.title_groups
            ],
            "resolver_evidence": [
                {
                    "anchor_candidate_id": evidence.anchor_candidate_id,
                    "entity_id": evidence.entity_id,
                    "candidate_ids": list(evidence.candidate_ids),
                    "selected_candidate_id": evidence.selected_candidate_id,
                    "top1_score": evidence.top1_score,
                    "top2_score": evidence.top2_score,
                    "score_margin": evidence.score_margin,
                    "accepted": evidence.accepted,
                    "rejection_reason": evidence.rejection_reason,
                }
                for evidence in trace.resolver_evidence
            ],
            "bridges": [
                {
                    "source_candidate_id": item.bridge.source_candidate_id,
                    "target_candidate_id": item.bridge.target_candidate_id,
                    "bridge_entity_id": item.bridge.bridge_entity_id,
                    "direction": item.bridge.direction,
                    "confidence": item.bridge.confidence,
                    "resolver_score": item.bridge.resolver_score,
                    "resolver_margin": item.bridge.resolver_margin,
                    "construction_reason": item.bridge.construction_reason,
                    "accepted": item.accepted,
                    "rejection_reason": item.rejection_reason,
                    "original_partner_rank": item.original_partner_rank,
                    "final_partner_rank": item.final_partner_rank,
                }
                for item in trace.bridges
            ],
            "protected_prefix": list(trace.protected_prefix),
            "exact_dense_fallback": trace.exact_dense_fallback,
            "emitted_edges": [
                _candidate_edge_record(edge) for edge in trace.emitted_edges
            ],
        }
    if isinstance(trace, ExecutionProvenanceTrace):
        return trace.model_dump(mode="json", exclude_none=True)
    if isinstance(trace, QueryConditionedExecutionProvenanceTrace):
        return {
            "trace_kind": trace.trace_kind,
            "native_graph_node_ids": list(trace.native_graph_node_ids),
            "dense_ranks": [_dense_rank_record(item) for item in trace.dense_ranks],
            "relation_description_version": trace.relation_description_version,
            "relations": [
                {
                    "edge_type": item.edge_type,
                    "similarity": item.similarity,
                    "affinity": item.affinity,
                }
                for item in trace.relations
            ],
            "transitions": [
                {
                    "source": item.source,
                    "target": item.target,
                    "edge_type": item.edge_type,
                    "direction": item.direction,
                    "recorded_weight": item.recorded_weight,
                    "relation_affinity": item.relation_affinity,
                    "probability": item.probability,
                    "cost": item.cost,
                }
                for item in trace.transitions
            ],
            "ppr_nodes": [
                {"node_id": item.node_id, "teleport": item.teleport, "score": item.score}
                for item in trace.ppr_nodes
            ],
            "ppr_iterations": trace.ppr_iterations,
            "ppr_residual": trace.ppr_residual,
            "ppr_converged": trace.ppr_converged,
            "candidate_prizes": [
                {
                    "node_id": item.node_id,
                    "dense_component": item.dense_component,
                    "ppr_component": item.ppr_component,
                    "prize": item.prize,
                }
                for item in trace.candidate_prizes
            ],
            "selected_candidate_ids": list(trace.selected_candidate_ids),
            "connector_node_ids": list(trace.connector_node_ids),
            "selection_steps": [
                {
                    "anchor_id": item.anchor_id,
                    "target_id": item.target_id,
                    "path_node_ids": list(item.path_node_ids),
                    "transitions": [
                        {
                            "source": transition.source,
                            "target": transition.target,
                            "edge_type": transition.edge_type,
                            "direction": transition.direction,
                        }
                        for transition in item.transitions
                    ],
                    "added_candidate_ids": list(item.added_candidate_ids),
                    "displaced_candidate_ids": list(item.displaced_candidate_ids),
                    "prize_gain": item.prize_gain,
                    "edge_cost": item.edge_cost,
                    "displacement_cost": item.displacement_cost,
                    "marginal_gain": item.marginal_gain,
                }
                for item in trace.selection_steps
            ],
            "selected_native_edges": [
                _provenance_edge_record(edge) for edge in trace.selected_native_edges
            ],
            "objective": trace.objective,
            "top_k": trace.top_k,
            "exact_dense_fallback": trace.exact_dense_fallback,
            "emitted_edges": [
                _candidate_edge_record(edge) for edge in trace.emitted_edges
            ],
            "scorer_identity": trace.scorer_identity,
            "variant": trace.variant,
        }
    assert isinstance(trace, StatelessExecutionProvenanceTrace)
    return {
        "trace_kind": trace.trace_kind,
        "dense_ranks": [_dense_rank_record(item) for item in trace.dense_ranks],
        "seed_candidate_ids": list(trace.seed_candidate_ids),
        "paths": [
            {
                "anchor_id": path.anchor_id,
                "partner_id": path.partner_id,
                "node_ids": list(path.node_ids),
                "path_confidence": path.path_confidence,
                "binding_valid": path.binding_valid,
                "completeness_valid": path.completeness_valid,
                "lifecycle_valid": path.lifecycle_valid,
                "accepted": path.accepted,
                "rejection_reason": path.rejection_reason,
                "original_partner_rank": path.original_partner_rank,
                "final_partner_rank": path.final_partner_rank,
            }
            for path in trace.paths
        ],
        "edges": [_provenance_edge_record(edge) for edge in trace.edges],
        "protected_prefix": list(trace.protected_prefix),
        "exact_dense_fallback": trace.exact_dense_fallback,
        "emitted_edges": [_candidate_edge_record(edge) for edge in trace.emitted_edges],
        "scorer_identity": trace.scorer_identity,
        "variant": trace.variant,
    }


def _dense_rank_record(item: DenseRankTrace) -> dict[str, object]:
    return {
        "node_id": item.node_id,
        "dense_rank": item.dense_rank,
        "dense_score": item.dense_score,
        "final_rank": item.final_rank,
    }


def _candidate_edge_record(edge: CandidateEdgeTrace) -> dict[str, object]:
    return {
        "source": edge.source,
        "target": edge.target,
        "edge_type": edge.edge_type,
        "confidence": edge.confidence,
    }


def _provenance_edge_record(edge: ProvenanceEdgeTrace) -> dict[str, object]:
    record: dict[str, object] = {
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
    if edge.semantic_rank is not None:
        record["semantic_rank"] = edge.semantic_rank
    if edge.semantic_score is not None:
        record["semantic_score"] = edge.semantic_score
    return record
