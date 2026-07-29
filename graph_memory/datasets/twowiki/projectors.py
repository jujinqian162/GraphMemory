from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.results import RankedResult
from graph_memory.datasets.twowiki.records import (
    TwoWikiLabelRecord,
    TwoWikiRankingRecord,
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


class TwoWikiToTextRankingRequest:
    def project(self, record: TwoWikiRankingRecord | object) -> TextRankingRequest:
        record = TwoWikiRankingRecord.model_validate(record)
        return TextRankingRequest(
            task_id=record.task_id,
            query_text=record.question,
            candidates=tuple(
                TextCandidate(
                    item_id=sentence.sentence_id,
                    text=f"{sentence.title}. {sentence.text}",
                    metadata={
                        "title": sentence.title,
                        "source_ref": sentence.title,
                        "sequence_index": sentence.sentence_index,
                        "position": sentence.position,
                        "question_type": record.question_type,
                    },
                )
                for sentence in record.candidate_sentences
            ),
        )


class TwoWikiToEvidenceGraphBuildRequest:
    def project(
        self, record: TwoWikiRankingRecord | object
    ) -> EvidenceGraphBuildRequest:
        record = TwoWikiRankingRecord.model_validate(record)
        return EvidenceGraphBuildRequest(
            task_id=record.task_id,
            query_text=record.question,
            nodes=tuple(
                EvidenceGraphBuildNode(
                    node_id=sentence.sentence_id,
                    text=sentence.text,
                    node_kind="document_sentence",
                    source_ref=sentence.title,
                    group_key=f"document:{sentence.title}",
                    sequence_index=sentence.sentence_index,
                    metadata={
                        "title": sentence.title,
                        "position": sentence.position,
                        "question_type": record.question_type,
                    },
                )
                for sentence in record.candidate_sentences
            ),
            input_visible_edges=(),
        )


class TwoWikiToEvidenceGraphRankingRequest:
    def project(
        self,
        record: TwoWikiRankingRecord,
        graph: EvidenceGraph,
        initial_scores: Mapping[str, float],
    ) -> EvidenceGraphRankingRequest:
        record = TwoWikiRankingRecord.model_validate(record)
        text_request = TwoWikiToTextRankingRequest().project(record)
        return EvidenceGraphRankingRequest(
            task_id=record.task_id,
            query_text=record.question,
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=dict(initial_scores),
        )


class TwoWikiToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[TwoWikiLabelRecord],
        graphs: Sequence[EvidenceGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=tuple(predictions),
            labels=tuple(
                EvidenceLabel(
                    task_id=label.task_id,
                    gold_answer=label.gold_answer,
                    gold_evidence_item_ids=tuple(label.gold_evidence_sentence_ids),
                    gold_dependency_edges=tuple(
                        _dependency_edge(edge)
                        for edge in label.gold_dependency_edges
                    ),
                )
                for label in labels
            ),
            graphs=tuple(graphs),
        )


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(
            f"Gold dependency edge must contain exactly two node IDs, got {len(edge)}."
        )
    return edge[0], edge[1]


__all__ = [
    "TwoWikiToEvidenceEvaluationRequest",
    "TwoWikiToEvidenceGraphBuildRequest",
    "TwoWikiToEvidenceGraphRankingRequest",
    "TwoWikiToTextRankingRequest",
]
