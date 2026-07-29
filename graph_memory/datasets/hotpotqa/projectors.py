from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.results import RankedResult
from graph_memory.datasets.hotpotqa.records import (
    HotpotQALabelRecord,
    HotpotQARankingRecord,
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


class HotpotQAToTextRankingRequest:
    def project(self, record: HotpotQARankingRecord | object) -> TextRankingRequest:
        record = HotpotQARankingRecord.model_validate(record)
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
                    },
                )
                for sentence in record.candidate_sentences
            ),
        )


class HotpotQAToEvidenceGraphBuildRequest:
    def project(
        self, record: HotpotQARankingRecord | object
    ) -> EvidenceGraphBuildRequest:
        record = HotpotQARankingRecord.model_validate(record)
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
                    },
                )
                for sentence in record.candidate_sentences
            ),
            input_visible_edges=(),
        )


class HotpotQAToEvidenceGraphRankingRequest:
    def project(
        self,
        record: HotpotQARankingRecord,
        graph: EvidenceGraph,
        initial_scores: Mapping[str, float],
    ) -> EvidenceGraphRankingRequest:
        record = HotpotQARankingRecord.model_validate(record)
        text_request = HotpotQAToTextRankingRequest().project(record)
        return EvidenceGraphRankingRequest(
            task_id=record.task_id,
            query_text=record.question,
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=dict(initial_scores),
        )


class HotpotQAToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[HotpotQALabelRecord],
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
    "HotpotQAToEvidenceEvaluationRequest",
    "HotpotQAToEvidenceGraphBuildRequest",
    "HotpotQAToEvidenceGraphRankingRequest",
    "HotpotQAToTextRankingRequest",
]
