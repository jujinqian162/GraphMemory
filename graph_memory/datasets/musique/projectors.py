from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.musique.records import (
    MuSiQueLabelRecord,
    MuSiQueRankingRecord,
)
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import (
    EvidenceGraphBuildNode,
    EvidenceGraphBuildRequest,
)
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
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
                    text=f"{paragraph['title']}. {paragraph['text']}",
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


class MuSiQueToEvidenceGraphBuildRequest:
    def project(self, record: MuSiQueRankingRecord) -> EvidenceGraphBuildRequest:
        return EvidenceGraphBuildRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            nodes=tuple(
                EvidenceGraphBuildNode(
                    node_id=paragraph["paragraph_id"],
                    text=paragraph["text"],
                    node_kind="document_paragraph",
                    source_ref=paragraph["title"],
                    group_key=f"document:{paragraph['title']}",
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


class MuSiQueToEvidenceGraphRankingRequest:
    def project(
        self,
        record: MuSiQueRankingRecord,
        graph: EvidenceGraph,
        initial_scores: Mapping[str, float],
    ) -> EvidenceGraphRankingRequest:
        text_request = MuSiQueToTextRankingRequest().project(record)
        return EvidenceGraphRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=initial_scores,
        )


class MuSiQueToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[MuSiQueLabelRecord],
        graphs: Sequence[EvidenceGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_evidence_paragraph_ids"]),
                    gold_dependency_edges=tuple(
                        _dependency_edge(edge)
                        for edge in label["gold_dependency_edges"]
                    ),
                )
                for label in labels
            ),
            graphs=graphs,
        )


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(
            f"Gold dependency edge must contain exactly two node IDs, got {len(edge)}."
        )
    return edge[0], edge[1]


__all__ = [
    "MuSiQueToEvidenceEvaluationRequest",
    "MuSiQueToEvidenceGraphBuildRequest",
    "MuSiQueToEvidenceGraphRankingRequest",
    "MuSiQueToTextRankingRequest",
]
