from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.twowiki.records import TwoWikiLabelRecord, TwoWikiRankingRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import GraphBuildNode, GraphBuildRequest
from graph_memory.retrieval.requests import (
    GraphRankingRequest,
    PositionRecencySpec,
    TemporalMemoryRankingRequest,
    TextCandidate,
    TextRankingRequest,
)


class TwoWikiToTextRankingRequest:
    def project(self, record: TwoWikiRankingRecord) -> TextRankingRequest:
        return TextRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=sentence["sentence_id"],
                    text=f'{sentence["title"]}. {sentence["text"]}',
                    metadata={
                        "title": sentence["title"],
                        "source_ref": sentence["title"],
                        "sequence_index": sentence["sentence_index"],
                        "position": sentence["position"],
                        "question_type": record["question_type"],
                    },
                )
                for sentence in record["candidate_sentences"]
            ),
        )


class TwoWikiToGraphBuildRequest:
    def project(self, record: TwoWikiRankingRecord) -> GraphBuildRequest:
        return GraphBuildRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            nodes=tuple(
                GraphBuildNode(
                    node_id=sentence["sentence_id"],
                    text=sentence["text"],
                    node_kind="document_sentence",
                    source_ref=sentence["title"],
                    group_key=f'document:{sentence["title"]}',
                    sequence_index=sentence["sentence_index"],
                    metadata={
                        "title": sentence["title"],
                        "position": sentence["position"],
                        "question_type": record["question_type"],
                    },
                )
                for sentence in record["candidate_sentences"]
            ),
            input_visible_edges=(),
        )


class TwoWikiToGraphRankingRequest:
    def project(
        self,
        record: TwoWikiRankingRecord,
        graph: MemoryGraph,
        initial_scores: Mapping[str, float],
    ) -> GraphRankingRequest:
        text_request = TwoWikiToTextRankingRequest().project(record)
        return GraphRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=initial_scores,
        )


class TwoWikiToTemporalMemoryRankingRequest:
    def project(
        self,
        record: TwoWikiRankingRecord,
        importance_by_item_id: Mapping[str, float],
    ) -> TemporalMemoryRankingRequest:
        return TemporalMemoryRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=sentence["sentence_id"],
                    text=sentence["text"],
                    metadata={
                        "title": sentence["title"],
                        "source_ref": sentence["title"],
                        "sequence_index": sentence["sentence_index"],
                        "position": sentence["position"],
                        "question_type": record["question_type"],
                    },
                )
                for sentence in record["candidate_sentences"]
            ),
            importance_by_item_id=importance_by_item_id,
            metadata={
                "position_by_item_id": {
                    sentence["sentence_id"]: sentence["position"]
                    for sentence in record["candidate_sentences"]
                }
            },
            recency=PositionRecencySpec(
                position_by_item_id={
                    sentence["sentence_id"]: sentence["position"]
                    for sentence in record["candidate_sentences"]
                }
            ),
        )


class TwoWikiToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[TwoWikiLabelRecord],
        graphs: Sequence[MemoryGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
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
    "TwoWikiToEvidenceEvaluationRequest",
    "TwoWikiToGraphBuildRequest",
    "TwoWikiToGraphRankingRequest",
    "TwoWikiToTemporalMemoryRankingRequest",
    "TwoWikiToTextRankingRequest",
]
