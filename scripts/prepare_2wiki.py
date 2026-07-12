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
from graph_memory.datasets.splits import sample_split
from graph_memory.datasets.twowiki import (
    TwoWikiLabelRecord,
    TwoWikiRankingRecord,
    combined_twowiki_records,
    convert_twowiki_example,
    convert_twowiki_examples,
    parse_twowiki_example,
    parse_twowiki_examples,
)
from graph_memory.io import read_json, write_json
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import RawPrepareStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.validation import (
    validate_twowiki_label_records,
    validate_twowiki_ranking_records,
)

LOGGER = logging.getLogger("prepare_2wiki")


@dataclass(frozen=True)
class ValidRawExamples:
    records: list[object]
    invalid_reason_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedTwoWikiRecords:
    task_inputs: list[TwoWikiRankingRecord]
    task_labels: list[TwoWikiLabelRecord]
    counts: dict[str, object]


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        RawPrepareStageConfig,
        description="Prepare a typed 2WikiMultiHopQA experiment split.",
        script=Path(__file__),
    )
    config = execution.config
    if config.dataset != "twowiki":
        raise ValueError(
            f"prepare_2wiki.py requires dataset=twowiki, got {config.dataset}"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        prepared = prepare_from_raw(config)
        task_inputs = prepared.task_inputs
        task_labels = prepared.task_labels
        inputs_by_task_id = {
            task_input["task_id"]: task_input for task_input in task_inputs
        }
        validate_twowiki_ranking_records(task_inputs)
        validate_twowiki_label_records(task_labels, inputs_by_task_id)

        write_json(config.outputs.input, task_inputs)
        write_json(config.outputs.labels, task_labels)
        write_json(
            config.outputs.combined,
            combined_twowiki_records(task_inputs, task_labels),
        )
        LOGGER.info("wrote combined inspection artifact: %s", config.outputs.combined)

        counts = cast(
            JsonObject,
            {
                **prepared.counts,
                "task_inputs": len(task_inputs),
                "task_labels": len(task_labels),
                "path_supported_tasks": sum(
                    1
                    for label in task_labels
                    if bool(label.get("gold_dependency_edges"))
                ),
            },
        )
        for key, value in counts.items():
            observations.count(key, value)
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("wrote inputs: %s", config.outputs.input)
        LOGGER.info("wrote labels: %s", config.outputs.labels)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


def select_valid_raw_examples(
    raw_records: Sequence[object], *, strict: bool
) -> ValidRawExamples:
    valid_records: list[object] = []
    invalid_reason_counts: Counter[str] = Counter()
    for record_index, raw_record in enumerate(raw_records):
        try:
            parsed_example = parse_twowiki_example(
                raw_record, record_index=record_index
            )
            converted_example = convert_twowiki_example(parsed_example)
            inputs_by_task_id = {
                converted_example.ranking_record[
                    "task_id"
                ]: converted_example.ranking_record
            }
            validate_twowiki_ranking_records([converted_example.ranking_record])
            validate_twowiki_label_records(
                [converted_example.label_record], inputs_by_task_id
            )
        except ValueError as error:
            if strict:
                raise ValueError(
                    f"Invalid 2Wiki raw example index={record_index}: {error}"
                ) from error
            invalid_reason_counts[str(error)] += 1
            continue
        valid_records.append(raw_record)
    return ValidRawExamples(
        records=valid_records, invalid_reason_counts=dict(invalid_reason_counts)
    )


def prepare_from_raw(config: RawPrepareStageConfig) -> PreparedTwoWikiRecords:
    raw_records = read_json(config.source)
    if not isinstance(raw_records, list):
        raise ValueError("2Wiki raw input must be a JSON list.")
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

    parsed_examples = parse_twowiki_examples(selected_records)
    conversion = convert_twowiki_examples(parsed_examples)
    return PreparedTwoWikiRecords(
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


def select_examples(
    raw_records: Sequence[object], *, count: int, seed: int, offset: int
) -> list[object]:
    return sample_split(raw_records, count=count, seed=seed, offset=offset)


if __name__ == "__main__":
    raise SystemExit(main())
