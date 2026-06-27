from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.contracts.errors import ContractValidationError
from graph_memory.datasets.longmemeval.records import (
    CombinedLongMemEvalRecord,
    LongMemEvalLabelRecord,
    LongMemEvalRankingRecord,
)


def coerce_longmemeval_ranking_records(records: object) -> list[LongMemEvalRankingRecord]:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid LongMemEval ranking records: artifact must be a list.")
    return [_coerce_longmemeval_ranking_record(record) for record in records]


def coerce_longmemeval_label_records(records: object) -> list[LongMemEvalLabelRecord]:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid LongMemEval label records: artifact must be a list.")
    return [_coerce_longmemeval_label_record(record) for record in records]


def combined_longmemeval_records(
    ranking_records: Sequence[LongMemEvalRankingRecord],
    label_records: Sequence[LongMemEvalLabelRecord],
) -> list[CombinedLongMemEvalRecord]:
    labels_by_task_id = {record["task_id"]: record for record in label_records}
    combined: list[CombinedLongMemEvalRecord] = []
    for record in ranking_records:
        task_id = record["task_id"]
        matching_label = labels_by_task_id.get(task_id)
        if matching_label is None:
            raise ValueError(f"Cannot combine task_id={task_id}: matching labels are missing.")
        combined.append(cast(CombinedLongMemEvalRecord, cast(object, {**record, **matching_label})))
    return combined


def _coerce_longmemeval_ranking_record(record: object) -> LongMemEvalRankingRecord:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Invalid LongMemEval ranking record: record is not an object.")
    return cast(LongMemEvalRankingRecord, cast(object, dict(record)))


def _coerce_longmemeval_label_record(record: object) -> LongMemEvalLabelRecord:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Invalid LongMemEval label record: record is not an object.")
    return cast(LongMemEvalLabelRecord, cast(object, dict(record)))
