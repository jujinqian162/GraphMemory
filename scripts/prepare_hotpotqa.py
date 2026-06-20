from __future__ import annotations

import argparse
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
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTemporalMemoryRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord, HotpotQALabelRecord
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
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.validation import (
    validate_task_importance_record,
    validate_hotpotqa_ranking_records,
    validate_hotpotqa_label_records,
)

LOGGER = logging.getLogger("prepare_hotpotqa")


@dataclass(frozen=True)
class PrepareHotpotQAArgs:
    source: str
    input: str
    input_labels: str | None
    importance: str | None
    output_input: str
    output_labels: str
    output_combined: str | None
    max_examples: int | None
    seed: int
    offset: int
    strict_invalid_examples: bool


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
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_input_path = Path(args.output_input)
    summary_path = output_input_path.with_name(f"{output_input_path.stem}.run_summary.json")
    effective_config = {
        "source": args.source,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "offset": args.offset,
        "write_combined": args.output_combined is not None,
        "drop_invalid_examples": not args.strict_invalid_examples,
        "strict_invalid_examples": args.strict_invalid_examples,
    }
    inputs = {"raw": args.input} if args.source == "raw" else {"canonical_inputs": args.input}
    if args.input_labels is not None:
        inputs["canonical_labels"] = args.input_labels
    if args.importance is not None:
        inputs["importance"] = args.importance
    outputs = {
        "inputs": args.output_input,
        "labels": args.output_labels,
        "run_summary": str(summary_path),
    }
    if args.output_combined is not None:
        outputs["combined"] = args.output_combined

    try:
        if args.source == "raw":
            prepared = prepare_from_raw(args)
        elif args.source == "importance":
            prepared = prepare_from_importance(args)
        else:
            raise ValueError(f"Unsupported HotpotQA prepare source: {args.source}")

        task_inputs = prepared.task_inputs
        task_labels = prepared.task_labels
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_hotpotqa_ranking_records(task_inputs)
        validate_hotpotqa_label_records(task_labels, inputs_by_task_id)

        write_json(args.output_input, task_inputs)
        write_json(args.output_labels, task_labels)
        if args.output_combined is not None:
            write_json(args.output_combined, combined_hotpotqa_records(task_inputs, task_labels))
            LOGGER.info("wrote compatibility combined artifact: %s", args.output_combined)

        LOGGER.info("wrote inputs: %s", args.output_input)
        LOGGER.info("wrote labels: %s", args.output_labels)
        counts = cast(JsonObject, {
            **prepared.counts,
            "task_inputs": len(task_inputs),
            "task_labels": len(task_labels),
        })
        summary = build_run_summary(
            script="prepare_hotpotqa.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts=counts,
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[] if args.output_combined is not None else ["compatibility output was not requested"],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote run summary: %s", summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="prepare_hotpotqa.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
            error=str(error),
        )
        write_run_summary(summary_path, summary)
        raise


def select_valid_raw_examples(raw_records: Sequence[object], *, strict: bool) -> ValidRawExamples:
    valid_records: list[object] = []
    invalid_reason_counts: Counter[str] = Counter()
    for record_index, raw_record in enumerate(raw_records):
        try:
            parsed_example = parse_hotpotqa_example(raw_record, record_index=record_index)
            converted_example = convert_hotpotqa_example(parsed_example)
            input_by_task_id = {converted_example.ranking_record["task_id"]: converted_example.ranking_record}
            validate_hotpotqa_ranking_records([converted_example.ranking_record])
            validate_hotpotqa_label_records(
                [converted_example.label_record],
                input_by_task_id,
            )
        except ValueError as error:
            if strict:
                raise ValueError(f"Invalid HotpotQA raw example index={record_index}: {error}") from error
            invalid_reason_counts[str(error)] += 1
            continue
        valid_records.append(raw_record)
    return ValidRawExamples(records=valid_records, invalid_reason_counts=dict(invalid_reason_counts))


def prepare_from_raw(args: PrepareHotpotQAArgs) -> PreparedHotpotQARecords:
    raw_records = read_json(args.input)
    if not isinstance(raw_records, list):
        raise ValueError("HotpotQA raw input must be a JSON list.")
    LOGGER.info("read raw examples: %s", len(raw_records))

    valid_raw_examples = select_valid_raw_examples(raw_records, strict=args.strict_invalid_examples)
    invalid_examples_dropped = len(raw_records) - len(valid_raw_examples.records)
    if invalid_examples_dropped:
        LOGGER.info("dropped invalid raw examples: %s", invalid_examples_dropped)

    selected_records = select_examples(
        valid_raw_examples.records,
        max_examples=args.max_examples,
        seed=args.seed,
        offset=args.offset,
    )
    LOGGER.info("selected examples: count=%s seed=%s offset=%s", len(selected_records), args.seed, args.offset)

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


def prepare_from_importance(args: PrepareHotpotQAArgs) -> PreparedHotpotQARecords:
    if args.input_labels is None:
        raise ValueError("--input_labels is required when --source importance.")
    if args.importance is None:
        raise ValueError("--importance is required when --source importance.")

    canonical_inputs = read_json(args.input)
    canonical_labels = read_json(args.input_labels)
    importance_artifact = read_json(args.importance)
    if not isinstance(canonical_inputs, list):
        raise ValueError("Canonical input artifact must be a JSON list.")
    if not isinstance(canonical_labels, list):
        raise ValueError("Canonical label artifact must be a JSON list.")
    if not isinstance(importance_artifact, dict):
        raise ValueError("Importance artifact must be a JSON object.")
    if importance_artifact.get("schema_version") != 1 or importance_artifact.get("method") != "memory_stream":
        raise ValueError("Importance artifact must use schema_version=1 and method=memory_stream.")
    task_records = importance_artifact.get("tasks")
    if not isinstance(task_records, list):
        raise ValueError("Importance artifact tasks must be a list.")

    typed_inputs = coerce_hotpotqa_ranking_records(canonical_inputs)
    typed_labels = coerce_hotpotqa_label_records(canonical_labels)
    input_by_task_id = {task_input["task_id"]: task_input for task_input in typed_inputs}
    label_by_task_id = {task_label["task_id"]: task_label for task_label in typed_labels}
    validate_hotpotqa_ranking_records(typed_inputs)
    validate_hotpotqa_label_records(typed_labels, input_by_task_id)
    temporal_by_task_id = {request.task_id: request for request in _temporal_requests(typed_inputs)}

    selected_records = select_ordered_records(
        task_records,
        max_examples=args.max_examples,
        offset=args.offset,
    )
    selected_inputs: list[HotpotQARankingRecord] = []
    selected_labels: list[HotpotQALabelRecord] = []
    for task_record in selected_records:
        if not isinstance(task_record, dict):
            raise ValueError("Importance task record must be a JSON object.")
        task_id = task_record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Importance task record task_id must be a non-empty string.")
        task_input = input_by_task_id.get(task_id)
        if task_input is None:
            raise ValueError(f"Canonical input missing importance task_id={task_id}.")
        task_label = label_by_task_id.get(task_id)
        if task_label is None:
            raise ValueError(f"Canonical labels missing importance task_id={task_id}.")
        validate_task_importance_record(task_record, temporal_by_task_id[task_id])
        selected_inputs.append(task_input)
        selected_labels.append(task_label)

    LOGGER.info("selected importance-backed examples: count=%s offset=%s", len(selected_inputs), args.offset)
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


def _temporal_requests(records: Sequence[HotpotQARankingRecord]) -> list[TemporalMemoryRankingRequest]:
    projector = HotpotQAToTemporalMemoryRankingRequest()
    return [projector.project(record, {}) for record in records]


def select_examples(raw_records: Sequence[object], *, max_examples: int | None, seed: int, offset: int) -> list[object]:
    if max_examples is None:
        if offset != 0:
            raise ValueError("--offset requires --max_examples so the split size is explicit.")
        return list(raw_records)
    return sample_split(raw_records, count=max_examples, seed=seed, offset=offset)


def select_ordered_records(records: Sequence[object], *, max_examples: int | None, offset: int) -> list[object]:
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if max_examples is None:
        if offset != 0:
            raise ValueError("--offset requires --max_examples so the split size is explicit.")
        return list(records)
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative.")
    end = offset + max_examples
    if end > len(records):
        raise ValueError(f"Requested split offset+count={end} exceeds available examples={len(records)}.")
    return list(records[offset:end])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labeled HotpotQA examples into leakage-safe ranking and label artifacts.")
    parser.add_argument("--source", choices=("raw", "importance"), default="raw", help="Source mode for materializing the split.")
    parser.add_argument("--input", required=True, help="Path to labeled HotpotQA raw JSON file.")
    parser.add_argument("--input_labels", default=None, help="Path to canonical labels when --source importance is used.")
    parser.add_argument("--importance", default=None, help="Path to compact Memory Stream importance artifact.")
    parser.add_argument("--output_input", required=True, help="Path to write HotpotQA ranking record JSON.")
    parser.add_argument("--output_labels", required=True, help="Path to write HotpotQA label record JSON.")
    parser.add_argument("--output_combined", default=None, help="Optional compatibility output with input and label fields combined.")
    parser.add_argument("--max_examples", type=int, default=None, help="Number of examples to sample after deterministic shuffling.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for deterministic split sampling.")
    parser.add_argument("--offset", type=int, default=0, help="Offset into the deterministic shuffled example order.")
    parser.add_argument(
        "--strict_invalid_examples",
        action="store_true",
        help="Fail on the first invalid raw HotpotQA example instead of dropping invalid examples.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PrepareHotpotQAArgs:
    namespace = build_parser().parse_args(argv)
    return PrepareHotpotQAArgs(
        source=namespace.source,
        input=namespace.input,
        input_labels=namespace.input_labels,
        importance=namespace.importance,
        output_input=namespace.output_input,
        output_labels=namespace.output_labels,
        output_combined=namespace.output_combined,
        max_examples=namespace.max_examples,
        seed=namespace.seed,
        offset=namespace.offset,
        strict_invalid_examples=namespace.strict_invalid_examples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
