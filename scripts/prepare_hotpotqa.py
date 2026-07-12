from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.contracts.common import JsonObject
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToTemporalMemoryRankingRequest,
)
from graph_memory.datasets.hotpotqa.records import (
    HotpotQARankingRecord,
    HotpotQALabelRecord,
)
from graph_memory.datasets.hotpotqa import (
    combined_hotpotqa_records,
    coerce_hotpotqa_label_records,
    coerce_hotpotqa_ranking_records,
    convert_hotpotqa_example,
    convert_hotpotqa_examples,
    parse_hotpotqa_example,
    parse_hotpotqa_examples,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest
from graph_memory.io import read_json, write_json
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import (
    ImportancePrepareStageConfig,
    PrepareStageConfig,
    RawPrepareStageConfig,
)
from graph_memory.experiment.state import stage_lifecycle
from pydantic import TypeAdapter
from graph_memory.validation import (
    validate_task_importance_record,
    validate_hotpotqa_ranking_records,
    validate_hotpotqa_label_records,
)

LOGGER = logging.getLogger("prepare_hotpotqa")


@dataclass(frozen=True)
class ValidRawExamples:
    records: list[object]
    invalid_reason_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedHotpotQARecords:
    task_inputs: list[HotpotQARankingRecord]
    task_labels: list[HotpotQALabelRecord]
    counts: dict[str, object]


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        TypeAdapter(PrepareStageConfig),
        description="Prepare a typed HotpotQA experiment split.",
        script=Path(__file__),
    )
    config = execution.config
    if config.dataset != "hotpotqa":
        raise ValueError(
            f"prepare_hotpotqa.py requires dataset=hotpotqa, got {config.dataset}"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        if isinstance(config, RawPrepareStageConfig):
            prepared = prepare_from_raw(config)
        else:
            prepared = prepare_from_importance(config)

        task_inputs = prepared.task_inputs
        task_labels = prepared.task_labels
        inputs_by_task_id = {
            task_input["task_id"]: task_input for task_input in task_inputs
        }
        validate_hotpotqa_ranking_records(task_inputs)
        validate_hotpotqa_label_records(task_labels, inputs_by_task_id)

        write_json(config.outputs.input, task_inputs)
        write_json(config.outputs.labels, task_labels)
        write_json(
            config.outputs.combined,
            combined_hotpotqa_records(task_inputs, task_labels),
        )
        LOGGER.info(
            "wrote compatibility combined artifact: %s", config.outputs.combined
        )

        LOGGER.info("wrote inputs: %s", config.outputs.input)
        LOGGER.info("wrote labels: %s", config.outputs.labels)
        counts = cast(
            JsonObject,
            {
                **prepared.counts,
                "task_inputs": len(task_inputs),
                "task_labels": len(task_labels),
            },
        )
        for key, value in counts.items():
            observations.count(key, value)
        observations.timing("total_seconds", time.perf_counter() - start_time)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


def select_valid_raw_examples(
    raw_records: Sequence[object], *, strict: bool
) -> ValidRawExamples:
    valid_records: list[object] = []
    invalid_reason_counts: Counter[str] = Counter()
    for record_index, raw_record in enumerate(raw_records):
        try:
            parsed_example = parse_hotpotqa_example(
                raw_record, record_index=record_index
            )
            converted_example = convert_hotpotqa_example(parsed_example)
            input_by_task_id = {
                converted_example.ranking_record[
                    "task_id"
                ]: converted_example.ranking_record
            }
            validate_hotpotqa_ranking_records([converted_example.ranking_record])
            validate_hotpotqa_label_records(
                [converted_example.label_record],
                input_by_task_id,
            )
        except ValueError as error:
            if strict:
                raise ValueError(
                    f"Invalid HotpotQA raw example index={record_index}: {error}"
                ) from error
            invalid_reason_counts[str(error)] += 1
            continue
        valid_records.append(raw_record)
    return ValidRawExamples(
        records=valid_records, invalid_reason_counts=dict(invalid_reason_counts)
    )


def prepare_from_raw(config: RawPrepareStageConfig) -> PreparedHotpotQARecords:
    raw_records = read_json(config.source)
    if not isinstance(raw_records, list):
        raise ValueError("HotpotQA raw input must be a JSON list.")
    LOGGER.info("read raw examples: %s", len(raw_records))

    valid_raw_examples = select_valid_raw_examples(
        raw_records, strict=config.strict_invalid_examples
    )
    invalid_examples_dropped = len(raw_records) - len(valid_raw_examples.records)
    if invalid_examples_dropped:
        LOGGER.info("dropped invalid raw examples: %s", invalid_examples_dropped)

    selected_records = select_examples(
        valid_raw_examples.records,
        count=config.count,
        seed=config.seed,
        offset=config.offset,
    )
    LOGGER.info(
        "selected examples: count=%s seed=%s offset=%s",
        len(selected_records),
        config.seed,
        config.offset,
    )

    parsed_examples = parse_hotpotqa_examples(selected_records)
    conversion = convert_hotpotqa_examples(parsed_examples)
    return PreparedHotpotQARecords(
        task_inputs=conversion.ranking_records,
        task_labels=conversion.label_records,
        counts={
            "raw_examples": len(raw_records),
            "valid_examples": len(valid_raw_examples.records),
            "invalid_examples_dropped": invalid_examples_dropped,
            "invalid_example_reasons": valid_raw_examples.invalid_reason_counts,
            "selected_examples": len(selected_records),
            "parsed_examples": len(parsed_examples),
        },
    )


def prepare_from_importance(
    config: ImportancePrepareStageConfig,
) -> PreparedHotpotQARecords:
    canonical_inputs = read_json(config.canonical_inputs)
    canonical_labels = read_json(config.canonical_labels)
    importance_artifact = read_json(config.importance)
    if not isinstance(canonical_inputs, list):
        raise ValueError("Canonical input artifact must be a JSON list.")
    if not isinstance(canonical_labels, list):
        raise ValueError("Canonical label artifact must be a JSON list.")
    if not isinstance(importance_artifact, dict):
        raise ValueError("Importance artifact must be a JSON object.")
    if (
        importance_artifact.get("schema_version") != 1
        or importance_artifact.get("method") != "memory_stream"
    ):
        raise ValueError(
            "Importance artifact must use schema_version=1 and method=memory_stream."
        )
    task_records = importance_artifact.get("tasks")
    if not isinstance(task_records, list):
        raise ValueError("Importance artifact tasks must be a list.")

    typed_inputs = coerce_hotpotqa_ranking_records(canonical_inputs)
    typed_labels = coerce_hotpotqa_label_records(canonical_labels)
    input_by_task_id = {
        task_input["task_id"]: task_input for task_input in typed_inputs
    }
    label_by_task_id = {
        task_label["task_id"]: task_label for task_label in typed_labels
    }
    validate_hotpotqa_ranking_records(typed_inputs)
    validate_hotpotqa_label_records(typed_labels, input_by_task_id)
    temporal_by_task_id = {
        request.task_id: request for request in _temporal_requests(typed_inputs)
    }

    selected_records = select_ordered_records(
        task_records,
        count=config.count,
        offset=config.offset,
    )
    selected_inputs: list[HotpotQARankingRecord] = []
    selected_labels: list[HotpotQALabelRecord] = []
    for task_record in selected_records:
        if not isinstance(task_record, dict):
            raise ValueError("Importance task record must be a JSON object.")
        task_id = task_record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(
                "Importance task record task_id must be a non-empty string."
            )
        task_input = input_by_task_id.get(task_id)
        if task_input is None:
            raise ValueError(f"Canonical input missing importance task_id={task_id}.")
        task_label = label_by_task_id.get(task_id)
        if task_label is None:
            raise ValueError(f"Canonical labels missing importance task_id={task_id}.")
        validate_task_importance_record(task_record, temporal_by_task_id[task_id])
        selected_inputs.append(task_input)
        selected_labels.append(task_label)

    LOGGER.info(
        "selected importance-backed examples: count=%s offset=%s",
        len(selected_inputs),
        config.offset,
    )
    return PreparedHotpotQARecords(
        task_inputs=selected_inputs,
        task_labels=selected_labels,
        counts={
            "canonical_task_inputs": len(typed_inputs),
            "canonical_task_labels": len(typed_labels),
            "importance_tasks": len(task_records),
            "selected_examples": len(selected_inputs),
            "parsed_examples": len(selected_inputs),
            "invalid_examples_dropped": 0,
            "invalid_example_reasons": {},
        },
    )


def _temporal_requests(
    records: Sequence[HotpotQARankingRecord],
) -> list[TemporalMemoryRankingRequest]:
    projector = HotpotQAToTemporalMemoryRankingRequest()
    return [projector.project(record, {}) for record in records]


def select_examples(
    raw_records: Sequence[object], *, count: int, seed: int, offset: int
) -> list[object]:
    return sample_split(raw_records, count=count, seed=seed, offset=offset)


def select_ordered_records(
    records: Sequence[object], *, count: int, offset: int
) -> list[object]:
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if count < 0:
        raise ValueError("count must be non-negative.")
    end = offset + count
    if end > len(records):
        raise ValueError(
            f"Requested split offset+count={end} exceeds available examples={len(records)}."
        )
    return list(records[offset:end])


if __name__ == "__main__":
    raise SystemExit(main())
