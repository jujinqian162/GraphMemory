from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.traject_bench.records import (
    TrajectBenchLabelRecord,
    TrajectBenchRankingRecord,
)
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import (
    EvidenceGraphBuildNode,
    EvidenceGraphBuildRequest,
    GraphBuildEdge,
)
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)


class TrajectBenchToTextRankingRequest:
    def project(self, record: TrajectBenchRankingRecord) -> TextRankingRequest:
        return TextRankingRequest(
            task_id=record["task_id"],
            query_text=record["query"],
            candidates=tuple(
                TextCandidate(
                    item_id=tool["tool_id"],
                    text=tool["text"],
                    metadata={
                        "tool_name": tool["tool_name"],
                        "parent_tool_name": tool["parent_tool_name"],
                        "api_name": tool["api_name"],
                        "domain_name": tool["domain_name"],
                        "position": tool["catalog_position"],
                    },
                )
                for tool in record["candidate_tools"]
            ),
        )


class TrajectBenchToEvidenceGraphBuildRequest:
    def project(self, record: TrajectBenchRankingRecord) -> EvidenceGraphBuildRequest:
        candidate_ids = {tool["tool_id"] for tool in record["candidate_tools"]}
        return EvidenceGraphBuildRequest(
            task_id=record["task_id"],
            query_text=record["query"],
            nodes=tuple(
                EvidenceGraphBuildNode(
                    node_id=tool["tool_id"],
                    text=tool["text"],
                    node_kind="tool_api",
                    source_ref=tool["tool_name"],
                    group_key=f"provider:{tool['parent_tool_name']}",
                    sequence_index=None,
                    metadata={
                        "tool_name": tool["tool_name"],
                        "api_name": tool["api_name"],
                        "domain_name": tool["domain_name"],
                        "catalog_position": tool["catalog_position"],
                    },
                )
                for tool in record["candidate_tools"]
            ),
            input_visible_edges=tuple(
                _catalog_edges(record, candidate_ids=candidate_ids)
            ),
        )


class TrajectBenchToEvidenceGraphRankingRequest:
    def project(
        self,
        record: TrajectBenchRankingRecord,
        graph: EvidenceGraph,
        initial_scores: Mapping[str, float],
    ) -> EvidenceGraphRankingRequest:
        text_request = TrajectBenchToTextRankingRequest().project(record)
        return EvidenceGraphRankingRequest(
            task_id=record["task_id"],
            query_text=record["query"],
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=initial_scores,
        )


class TrajectBenchToExecutionProvenanceRankingRequest:
    def project(
        self,
        record: TrajectBenchRankingRecord,
    ) -> ExecutionProvenanceRankingRequest:
        text_request = TrajectBenchToTextRankingRequest().project(record)
        candidate_ids = {candidate.item_id for candidate in text_request.candidates}
        graph = ExecutionProvenanceGraph(
            task_id=record["task_id"],
            nodes=tuple(
                ExecutionProvenanceNode(
                    node_id=candidate.item_id,
                    node_type=ProvenanceNodeType.TOOL_CALL,
                    text=candidate.text,
                    metadata={
                        **candidate.metadata,
                        "prospective": True,
                        "source": "traject_bench_catalog",
                    },
                )
                for candidate in text_request.candidates
            ),
            edges=tuple(
                _catalog_provenance_edges(record, candidate_ids=candidate_ids)
            ),
        )
        return ExecutionProvenanceRankingRequest(
            task_id=record["task_id"],
            query_text=record["query"],
            candidates=text_request.candidates,
            graph=graph,
        )


class TrajectBenchToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[TrajectBenchLabelRecord],
        graphs: Sequence[EvidenceGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_tool_ids"]),
                    gold_dependency_edges=tuple(
                        _dependency_edge(edge)
                        for edge in label["gold_dependency_edges"]
                    ),
                )
                for label in labels
            ),
            graphs=graphs,
        )


def _catalog_edges(
    record: TrajectBenchRankingRecord,
    *,
    candidate_ids: set[str],
) -> list[GraphBuildEdge]:
    edges: list[GraphBuildEdge] = []
    seen: set[tuple[str, str]] = set()
    for tool in record["candidate_tools"]:
        source = tool["tool_id"]
        for target in tool["connected_tool_ids"]:
            key = (source, target)
            if target not in candidate_ids or source == target or key in seen:
                continue
            seen.add(key)
            edges.append(
                GraphBuildEdge(
                    source=source,
                    target=target,
                    edge_type="sequential",
                    weight=1.0,
                    directed=True,
                    metadata={"source": "traject_bench_catalog_connection"},
                )
            )
    return edges


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(
            f"Gold TRAJECT-Bench dependency edge must have two IDs, got {len(edge)}."
        )
    return edge[0], edge[1]


def _catalog_provenance_edges(
    record: TrajectBenchRankingRecord,
    *,
    candidate_ids: set[str],
) -> list[ExecutionProvenanceEdge]:
    edges: list[ExecutionProvenanceEdge] = []
    seen: set[tuple[str, str]] = set()
    for tool in record["candidate_tools"]:
        source = tool["tool_id"]
        for target in tool["connected_tool_ids"]:
            key = (source, target)
            if target not in candidate_ids or source == target or key in seen:
                continue
            seen.add(key)
            edges.append(
                ExecutionProvenanceEdge(
                    source=source,
                    target=target,
                    edge_type=ProvenanceEdgeType.DEPENDS_ON,
                    metadata={"source": "traject_bench_catalog_connection"},
                )
            )
    return edges


__all__ = [
    "TrajectBenchToEvidenceEvaluationRequest",
    "TrajectBenchToEvidenceGraphBuildRequest",
    "TrajectBenchToEvidenceGraphRankingRequest",
    "TrajectBenchToExecutionProvenanceRankingRequest",
    "TrajectBenchToTextRankingRequest",
]
