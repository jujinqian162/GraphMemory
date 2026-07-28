from __future__ import annotations

from collections.abc import Sequence

from graph_memory.datasets.twowiki_provenance.records import (
    TwoWikiProvenanceLabelRecord,
    TwoWikiProvenanceRankingRecord,
)
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.retrieval.results import RankedResult


class TwoWikiProvenanceToTextRankingRequest:
    def project(
        self, record: TwoWikiProvenanceRankingRecord | object
    ) -> TextRankingRequest:
        record = TwoWikiProvenanceRankingRecord.model_validate(record)
        return TextRankingRequest(
            task_id=record.task_id,
            query_text=record.question,
            candidates=tuple(
                TextCandidate(
                    item_id=candidate.output_id,
                    text=f"{candidate.title}. {candidate.text}",
                    metadata={
                        "title": candidate.title,
                        "source_ref": candidate.title,
                        "sequence_index": candidate.sentence_index,
                        "position": candidate.position,
                        "question_type": record.question_type,
                    },
                )
                for candidate in record.candidates
            ),
        )


class TwoWikiProvenanceToExecutionProvenanceRankingRequest:
    def project(
        self, record: TwoWikiProvenanceRankingRecord | object
    ) -> ExecutionProvenanceRankingRequest:
        record = TwoWikiProvenanceRankingRecord.model_validate(record)
        text_request = TwoWikiProvenanceToTextRankingRequest().project(record)
        return ExecutionProvenanceRankingRequest(
            task_id=record.task_id,
            query_text=record.question,
            candidates=text_request.candidates,
            graph=record.graph,
        )


class TwoWikiProvenanceToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[TwoWikiProvenanceLabelRecord],
        graphs: Sequence[EvidenceGraph],
    ) -> EvidenceEvaluationRequest:
        validated_labels = tuple(
            TwoWikiProvenanceLabelRecord.model_validate(label) for label in labels
        )
        return EvidenceEvaluationRequest(
            predictions=tuple(predictions),
            labels=tuple(
                EvidenceLabel(
                    task_id=label.task_id,
                    gold_answer=label.gold_answer,
                    gold_evidence_item_ids=label.gold_evidence_output_ids,
                    gold_dependency_edges=label.gold_dependency_edges,
                )
                for label in validated_labels
            ),
            graphs=tuple(graphs),
        )


__all__ = [
    "TwoWikiProvenanceToEvidenceEvaluationRequest",
    "TwoWikiProvenanceToExecutionProvenanceRankingRequest",
    "TwoWikiProvenanceToTextRankingRequest",
]
