from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.musique.records import MuSiQueLabelRecord, MuSiQueRankingRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import GraphBuildNode, GraphBuildRequest
from graph_memory.retrieval.requests import (
    GraphRankingRequest,
    TemporalMemoryRankingRequest,
    TextCandidate,
    TextRankingRequest,
)


class MuSiQueToTextRankingRequest:
    def project(self, record: MuSiQueRankingRecord) -> TextRankingRequest:
        return TextRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=paragraph["paragraph_id"],
                    text=f'{paragraph["title"]}. {paragraph["text"]}',
                    metadata={
                        "title": paragraph["title"],
                        "source_ref": paragraph["title"],
                        "sequence_index": paragraph["paragraph_index"],
                        "position": paragraph["position"],
                    },
                )
                for paragraph in record["candidate_paragraphs"]
            ),
        )


class MuSiQueToGraphBuildRequest:
    def project(self, record: MuSiQueRankingRecord) -> GraphBuildRequest:
        return GraphBuildRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            nodes=tuple(
                GraphBuildNode(
                    node_id=paragraph["paragraph_id"],
                    text=paragraph["text"],
                    node_kind="document_paragraph",
                    source_ref=paragraph["title"],
                    group_key=f'document:{paragraph["title"]}',
                    sequence_index=paragraph["paragraph_index"],
                    metadata={
                        "title": paragraph["title"],
                        "position": paragraph["position"],
                    },
                )
                for paragraph in record["candidate_paragraphs"]
            ),
            input_visible_edges=(),
        )


class MuSiQueToGraphRankingRequest:
    def project(
        self,
        record: MuSiQueRankingRecord,
        graph: MemoryGraph,
        initial_scores: Mapping[str, float],
    ) -> GraphRankingRequest:
        text_request = MuSiQueToTextRankingRequest().project(record)
        return GraphRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=initial_scores,
        )


class MuSiQueToTemporalMemoryRankingRequest:
    def project(
        self,
        record: MuSiQueRankingRecord,
        importance_by_item_id: Mapping[str, float],
    ) -> TemporalMemoryRankingRequest:
        return TemporalMemoryRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=paragraph["paragraph_id"],
                    text=paragraph["text"],
                    metadata={
                        "title": paragraph["title"],
                        "source_ref": paragraph["title"],
                        "sequence_index": paragraph["paragraph_index"],
                        "position": paragraph["position"],
                    },
                )
                for paragraph in record["candidate_paragraphs"]
            ),
            importance_by_item_id=importance_by_item_id,
            metadata={
                "position_by_item_id": {
                    paragraph["paragraph_id"]: paragraph["position"]
                    for paragraph in record["candidate_paragraphs"]
                }
            },
        )


class MuSiQueToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[MuSiQueLabelRecord],
        graphs: Sequence[MemoryGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_evidence_paragraph_ids"]),
                    gold_dependency_edges=tuple(_dependency_edge(edge) for edge in label["gold_dependency_edges"]),
                )
                for label in labels
            ),
            graphs=graphs,
        )


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(f"Gold dependency edge must contain exactly two node IDs, got {len(edge)}.")
    return edge[0], edge[1]


__all__ = [
    "MuSiQueToEvidenceEvaluationRequest",
    "MuSiQueToGraphBuildRequest",
    "MuSiQueToGraphRankingRequest",
    "MuSiQueToTemporalMemoryRankingRequest",
    "MuSiQueToTextRankingRequest",
]
