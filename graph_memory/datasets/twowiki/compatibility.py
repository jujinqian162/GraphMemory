from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.contracts.errors import ContractValidationError
from graph_memory.datasets.twowiki.records import CombinedTwoWikiRecord, TwoWikiLabelRecord, TwoWikiRankingRecord


def coerce_twowiki_ranking_records(records: object) -> list[TwoWikiRankingRecord]:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid 2Wiki canonical ranking records: artifact must be a list.")
    return [_coerce_twowiki_ranking_record(record) for record in records]


def coerce_twowiki_label_records(records: object) -> list[TwoWikiLabelRecord]:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid 2Wiki canonical label records: artifact must be a list.")
    return [_coerce_twowiki_label_record(record) for record in records]


def combined_twowiki_records(
    ranking_records: Sequence[TwoWikiRankingRecord],
    label_records: Sequence[TwoWikiLabelRecord],
) -> list[CombinedTwoWikiRecord]:
    labels_by_task_id = {record["task_id"]: record for record in label_records}
    combined: list[CombinedTwoWikiRecord] = []
    for record in ranking_records:
        task_id = record["task_id"]
        matching_label = labels_by_task_id.get(task_id)
        if matching_label is None:
            raise ValueError(f"Cannot combine task_id={task_id}: matching labels are missing.")
        combined.append(cast(CombinedTwoWikiRecord, cast(object, {**record, **matching_label})))
    return combined


def _coerce_twowiki_ranking_record(record: object) -> TwoWikiRankingRecord:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Invalid 2Wiki canonical ranking record: record is not an object.")
    return cast(TwoWikiRankingRecord, cast(object, dict(record)))


def _coerce_twowiki_label_record(record: object) -> TwoWikiLabelRecord:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Invalid 2Wiki canonical label record: record is not an object.")
    return cast(TwoWikiLabelRecord, cast(object, dict(record)))
