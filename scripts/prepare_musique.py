from __future__ import annotations

import argparse
import json
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
from graph_memory.datasets.musique import (
    MuSiQueLabelRecord,
    MuSiQueRankingRecord,
    combined_musique_records,
    convert_musique_example,
    convert_musique_examples,
    parse_musique_example,
    parse_musique_examples,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.io import write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.validation import validate_musique_label_records, validate_musique_ranking_records

LOGGER = logging.getLogger("prepare_musique")


@dataclass(frozen=True)
class PrepareMuSiQueArgs:
    input: str
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
class PreparedMuSiQueRecords:
    task_inputs: list[MuSiQueRankingRecord]
    task_labels: list[MuSiQueLabelRecord]
    counts: dict[str, object]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_input_path = Path(args.output_input)
    summary_path = output_input_path.with_name(f"{output_input_path.stem}.run_summary.json")
    effective_config = {
        "max_examples": args.max_examples,
        "seed": args.seed,
        "offset": args.offset,
        "write_combined": args.output_combined is not None,
        "drop_invalid_examples": not args.strict_invalid_examples,
        "strict_invalid_examples": args.strict_invalid_examples,
    }
    inputs = {"raw": args.input}
    outputs = {
        "inputs": args.output_input,
        "labels": args.output_labels,
        "run_summary": str(summary_path),
    }
    if args.output_combined is not None:
        outputs["combined"] = args.output_combined

    try:
        prepared = prepare_from_raw(args)
        task_inputs = prepared.task_inputs
        task_labels = prepared.task_labels
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_musique_ranking_records(task_inputs)
        validate_musique_label_records(task_labels, inputs_by_task_id)

        write_json(args.output_input, task_inputs)
        write_json(args.output_labels, task_labels)
        if args.output_combined is not None:
            write_json(args.output_combined, combined_musique_records(task_inputs, task_labels))
            LOGGER.info("wrote combined inspection artifact: %s", args.output_combined)

        counts = cast(JsonObject, {
            **prepared.counts,
            "task_inputs": len(task_inputs),
            "task_labels": len(task_labels),
            "path_supported_tasks": sum(1 for label in task_labels if bool(label.get("gold_dependency_edges"))),
        })
        summary = build_run_summary(
            script="prepare_musique.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts=counts,
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[] if args.output_combined is not None else ["combined inspection output was not requested"],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote inputs: %s", args.output_input)
        LOGGER.info("wrote labels: %s", args.output_labels)
        LOGGER.info("wrote run summary: %s", summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="prepare_musique.py",
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
            parsed_example = parse_musique_example(raw_record, record_index=record_index)
            converted_example = convert_musique_example(parsed_example)
            inputs_by_task_id = {converted_example.ranking_record["task_id"]: converted_example.ranking_record}
            validate_musique_ranking_records([converted_example.ranking_record])
            validate_musique_label_records([converted_example.label_record], inputs_by_task_id)
        except ValueError as error:
            if strict:
                raise ValueError(f"Invalid MuSiQue raw example index={record_index}: {error}") from error
            invalid_reason_counts[str(error)] += 1
            continue
        valid_records.append(raw_record)
    return ValidRawExamples(records=valid_records, invalid_reason_counts=dict(invalid_reason_counts))


def prepare_from_raw(args: PrepareMuSiQueArgs) -> PreparedMuSiQueRecords:
    raw_records = read_jsonl(args.input)
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

    parsed_examples = parse_musique_examples(selected_records)
    conversion = convert_musique_examples(parsed_examples)
    return PreparedMuSiQueRecords(
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


def read_jsonl(path: str) -> list[object]:
    records: list[object] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(cast(object, json.loads(stripped)))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def select_examples(raw_records: Sequence[object], *, max_examples: int | None, seed: int, offset: int) -> list[object]:
    if max_examples is None:
        if offset != 0:
            raise ValueError("--offset requires --max_examples so the split size is explicit.")
        return list(raw_records)
    return sample_split(raw_records, count=max_examples, seed=seed, offset=offset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labeled MuSiQue-Ans JSONL examples into leakage-safe ranking and label artifacts.")
    _ = parser.add_argument("--input", required=True, help="Path to official MuSiQue-Ans JSONL file.")
    _ = parser.add_argument("--output_input", required=True, help="Path to write MuSiQue ranking record JSON.")
    _ = parser.add_argument("--output_labels", required=True, help="Path to write MuSiQue label record JSON.")
    _ = parser.add_argument("--output_combined", default=None, help="Optional inspection output with input and label fields combined.")
    _ = parser.add_argument("--max_examples", type=int, default=None, help="Number of examples to sample after deterministic shuffling.")
    _ = parser.add_argument("--seed", type=int, default=13, help="Random seed for deterministic split sampling.")
    _ = parser.add_argument("--offset", type=int, default=0, help="Offset into the deterministic shuffled example order.")
    _ = parser.add_argument(
        "--strict_invalid_examples",
        action="store_true",
        help="Fail on the first invalid raw MuSiQue example instead of dropping invalid examples.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PrepareMuSiQueArgs:
    namespace = build_parser().parse_args(argv)
    output_combined = cast(str | None, getattr(namespace, "output_combined"))
    return PrepareMuSiQueArgs(
        input=cast(str, getattr(namespace, "input")),
        output_input=cast(str, getattr(namespace, "output_input")),
        output_labels=cast(str, getattr(namespace, "output_labels")),
        output_combined=output_combined,
        max_examples=cast(int | None, getattr(namespace, "max_examples")),
        seed=cast(int, getattr(namespace, "seed")),
        offset=cast(int, getattr(namespace, "offset")),
        strict_invalid_examples=cast(bool, getattr(namespace, "strict_invalid_examples")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
