from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ProvenanceNodeType,
)
from graph_memory.retrieval.contracts import (
    ExecutionProvenanceTrace,
    ProvenanceBindingTrace,
    ProvenanceEdgeTrace,
    ProvenancePathTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.methods.execution_provenance.config import (
    ExecutionProvenanceConfig,
)
from graph_memory.retrieval.methods.execution_provenance.search import (
    ProvenancePath,
    ProvenancePathScore,
    beam_search_provenance_paths,
    invalidated_node_ids,
    score_provenance_path,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class ExecutionProvenanceRetriever:
    dense_ranker: DenseTaskRetriever
    config: ExecutionProvenanceConfig = ExecutionProvenanceConfig()
    name: str = "execution_provenance_retriever"

    def rank_task(
        self,
        request: RankingMethodRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        if not isinstance(request, ExecutionProvenanceRankingRequest):
            raise TypeError(
                f"{self.name} requires ExecutionProvenanceRankingRequest, "
                f"got {type(request).__name__}."
            )
        semantic_scores = _semantic_scores(self.dense_ranker, request)
        candidate_ids = [candidate.item_id for candidate in request.candidates]
        candidate_by_id = {
            candidate.item_id: candidate for candidate in request.candidates
        }
        node_by_id = {node.node_id: node for node in request.graph.nodes}
        eligible_seed_ids = [
            node_id
            for node_id in candidate_ids
            if node_by_id[node_id].node_type is ProvenanceNodeType.TOOL_OUTPUT
            or (
                node_by_id[node_id].node_type is ProvenanceNodeType.TOOL_CALL
                and bool(candidate_by_id[node_id].text.strip())
            )
        ]
        if not eligible_seed_ids:
            eligible_seed_ids = candidate_ids
        seed_ids = tuple(
            sorted(
                eligible_seed_ids,
                key=lambda node_id: (-semantic_scores[node_id], node_id),
            )[
                : min(
                    max(self.config.seed_top_s, top_k),
                    len(eligible_seed_ids),
                )
            ]
        )
        invalidated = invalidated_node_ids(request)
        path_records = [
            (
                score_provenance_path(
                    path,
                    semantic_scores=semantic_scores,
                    invalidated_node_ids=invalidated,
                    node_by_id=node_by_id,
                    config=self.config,
                ),
                path,
            )
            for path in beam_search_provenance_paths(
                request,
                seed_ids,
                semantic_scores=semantic_scores,
                invalidated_node_ids=invalidated,
                config=self.config,
            )
        ]
        path_records.sort(
            key=lambda record: (
                -record[0].total,
                len(record[1].edges),
                record[1].node_ids,
                tuple(_edge_key(edge) for edge in record[1].edges),
            )
        )
        selected = _select_unique_paths(path_records, self.config.top_paths)
        best_path_score_by_node: dict[str, float] = {}
        for score, path in selected:
            for node_id in path.node_ids:
                if node_id not in candidate_by_id:
                    continue
                best_path_score_by_node[node_id] = max(
                    best_path_score_by_node.get(node_id, float("-inf")),
                    score.total,
                )
        ranked_nodes = []
        for node_id in candidate_ids:
            semantic_score = self.config.semantic_weight * semantic_scores[
                node_id
            ] - self.config.invalidation_penalty * float(node_id in invalidated)
            path_score = best_path_score_by_node.get(node_id)
            ranked_nodes.append(
                RankedNode(
                    node_id,
                    semantic_score
                    + (max(0.0, path_score) if path_score is not None else 0.0),
                )
            )
        ranked_nodes.sort(key=lambda node: (-node.score, node.node_id))
        selected_edges = _selected_edges(selected)
        top_candidate_ids = {node.node_id for node in ranked_nodes[:top_k]}
        logical_edges = _logical_candidate_edges(
            selected,
            candidate_ids={
                node_id
                for node_id in candidate_ids
                if node_by_id[node_id].node_type is ProvenanceNodeType.TOOL_OUTPUT
            },
            selected_candidate_ids=top_candidate_ids,
        )
        trace_node_ids = tuple(
            sorted({node_id for _score, path in selected for node_id in path.node_ids})
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=logical_edges,
                native_trace=ExecutionProvenanceTrace(
                    node_ids=trace_node_ids,
                    paths=tuple(
                        ProvenancePathTrace(
                            node_ids=path.node_ids,
                            score=score.total,
                            semantic_relevance=score.semantic_relevance,
                            binding_consistency=score.binding_consistency,
                            provenance_completeness=score.provenance_completeness,
                            explicit_grounding=score.explicit_grounding,
                            path_length_penalty=score.path_length_penalty,
                            invalidation_penalty=score.invalidation_penalty,
                        )
                        for score, path in selected
                    ),
                    edges=tuple(_edge_trace(edge) for edge in selected_edges),
                ),
            ),
        )


def _semantic_scores(
    ranker: DenseTaskRetriever,
    request: ExecutionProvenanceRankingRequest,
) -> dict[str, float]:
    ranked = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    return {node.node_id: node.score for node in ranked}


def _selected_edges(
    selected: list[tuple[ProvenancePathScore, ProvenancePath]],
) -> list[ExecutionProvenanceEdge]:
    by_key = {_edge_key(edge): edge for _score, path in selected for edge in path.edges}
    return [by_key[key] for key in sorted(by_key)]


def _select_unique_paths(
    path_records: list[tuple[ProvenancePathScore, ProvenancePath]],
    limit: int,
) -> list[tuple[ProvenancePathScore, ProvenancePath]]:
    selected: list[tuple[ProvenancePathScore, ProvenancePath]] = []
    seen_node_paths: set[tuple[str, ...]] = set()
    for record in path_records:
        if record[1].node_ids in seen_node_paths:
            continue
        seen_node_paths.add(record[1].node_ids)
        selected.append(record)
        if len(selected) == limit:
            break
    return selected


def _logical_candidate_edges(
    selected: list[tuple[ProvenancePathScore, ProvenancePath]],
    *,
    candidate_ids: set[str],
    selected_candidate_ids: set[str],
) -> list[GraphEdge]:
    logical: dict[tuple[str, str], GraphEdge] = {}
    for _score, path in selected:
        for source_index, source in enumerate(path.node_ids):
            if source not in candidate_ids:
                continue
            for target in path.node_ids[source_index + 1 :]:
                if target not in candidate_ids:
                    continue
                if (
                    source in selected_candidate_ids
                    and target in selected_candidate_ids
                ):
                    logical[(source, target)] = {
                        "source": source,
                        "target": target,
                        "edge_type": "sequential",
                        "weight": 1.0,
                        "directed": True,
                    }
                break
    return [logical[key] for key in sorted(logical)]


def _edge_key(edge: ExecutionProvenanceEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.edge_type.value)


def _edge_trace(edge: ExecutionProvenanceEdge) -> ProvenanceEdgeTrace:
    return ProvenanceEdgeTrace(
        source=edge.source,
        target=edge.target,
        edge_type=edge.edge_type,
        weight=edge.weight,
        binding=(
            ProvenanceBindingTrace(
                output_field=edge.binding.output_field,
                input_parameter=edge.binding.input_parameter,
                binding_kind=edge.binding.binding_kind,
            )
            if edge.binding is not None
            else None
        ),
    )


__all__ = ["ExecutionProvenanceRetriever"]
