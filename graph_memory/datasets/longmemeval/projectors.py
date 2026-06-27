from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.longmemeval.records import LongMemEvalLabelRecord, LongMemEvalRankingRecord, LongMemEvalTurnItem
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import GraphBuildNode, GraphBuildRequest
from graph_memory.retrieval.requests import (
    GraphRankingRequest,
    TemporalMemoryRankingRequest,
    TextCandidate,
    TextRankingRequest,
)


class LongMemEvalToTextRankingRequest:
    def project(self, record: LongMemEvalRankingRecord) -> TextRankingRequest:
        return TextRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=item["item_id"],
                    text=item["text"],
                    metadata=_candidate_metadata(record, item),
                )
                for item in record["candidate_items"]
            ),
        )


class LongMemEvalToGraphBuildRequest:
    def project(self, record: LongMemEvalRankingRecord) -> GraphBuildRequest:
        return GraphBuildRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            nodes=tuple(
                GraphBuildNode(
                    node_id=item["item_id"],
                    text=item["text"],
                    node_kind="conversation_turn",
                    source_ref=item["session_id"],
                    group_key=f'session:{item["session_id"]}',
                    sequence_index=item["turn_index"],
                    metadata={
                        "session_id": item["session_id"],
                        "session_order": item["session_order"],
                        "turn_index": item["turn_index"],
                        "global_position": item["global_position"],
                        "role": item["role"],
                        "datetime": item["datetime"],
                        "question_type": _question_type(record),
                    },
                )
                for item in record["candidate_items"]
            ),
            input_visible_edges=(),
        )


class LongMemEvalToGraphRankingRequest:
    def project(
        self,
        record: LongMemEvalRankingRecord,
        graph: MemoryGraph,
        initial_scores: Mapping[str, float],
    ) -> GraphRankingRequest:
        text_request = LongMemEvalToTextRankingRequest().project(record)
        return GraphRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=text_request.candidates,
            graph=graph,
            initial_scores=initial_scores,
        )


class LongMemEvalToTemporalMemoryRankingRequest:
    def project(
        self,
        record: LongMemEvalRankingRecord,
        importance_by_item_id: Mapping[str, float] | None = None,
    ) -> TemporalMemoryRankingRequest:
        external_importance = importance_by_item_id or {}
        return TemporalMemoryRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=item["item_id"],
                    text=item["text"],
                    metadata=_candidate_metadata(record, item),
                )
                for item in record["candidate_items"]
            ),
            importance_by_item_id={
                item["item_id"]: float(external_importance.get(item["item_id"], 0.0))
                for item in record["candidate_items"]
            },
            metadata={
                "position_by_item_id": {
                    item["item_id"]: item["global_position"]
                    for item in record["candidate_items"]
                },
                "session_order_by_item_id": {
                    item["item_id"]: item["session_order"]
                    for item in record["candidate_items"]
                },
                "turn_index_by_item_id": {
                    item["item_id"]: item["turn_index"]
                    for item in record["candidate_items"]
                },
                "datetime_by_item_id": {
                    item["item_id"]: item["datetime"]
                    for item in record["candidate_items"]
                },
            },
        )


class LongMemEvalToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[LongMemEvalLabelRecord],
        graphs: Sequence[MemoryGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_support_item_ids"]),
                    gold_dependency_edges=tuple(_dependency_edge(edge) for edge in label["gold_dependency_edges"]),
                    gold_session_ids=tuple(label["gold_support_session_ids"]),
                )
                for label in labels
            ),
            graphs=graphs,
        )


def _candidate_metadata(
    record: LongMemEvalRankingRecord,
    item: LongMemEvalTurnItem,
) -> Mapping[str, str | int | float | bool | None]:
    return {
        "session_id": str(item["session_id"]),
        "source_ref": str(item["session_id"]),
        "session_order": int(item["session_order"]),
        "turn_index": int(item["turn_index"]),
        "sequence_index": int(item["turn_index"]),
        "position": int(item["global_position"]),
        "global_position": int(item["global_position"]),
        "role": str(item["role"]),
        "datetime": str(item["datetime"]),
        "question_type": _question_type(record),
    }


def _question_type(record: LongMemEvalRankingRecord) -> str:
    value = record["metadata"].get("question_type")
    return value if isinstance(value, str) else ""


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(f"Gold dependency edge must contain exactly two node IDs, got {len(edge)}.")
    return edge[0], edge[1]


__all__ = [
    "LongMemEvalToEvidenceEvaluationRequest",
    "LongMemEvalToGraphBuildRequest",
    "LongMemEvalToGraphRankingRequest",
    "LongMemEvalToTemporalMemoryRankingRequest",
    "LongMemEvalToTextRankingRequest",
]
