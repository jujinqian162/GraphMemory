from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.datasets.hotpotqa import (
    combined_memory_tasks,
    convert_hotpotqa_example,
    convert_hotpotqa_examples,
    parse_hotpotqa_example,
    parse_hotpotqa_examples,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.validation import (
    validate_memory_task_inputs,
    validate_memory_task_labels,
)

LOGGER = logging.getLogger("prepare_hotpotqa")


@dataclass(frozen=True)
class PrepareHotpotQAArgs:
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
        task_inputs = conversion.task_inputs
        task_labels = conversion.task_labels
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_memory_task_inputs(task_inputs)
        validate_memory_task_labels(task_labels, inputs_by_task_id)

        write_json(args.output_input, task_inputs)
        write_json(args.output_labels, task_labels)
        if args.output_combined is not None:
            write_json(args.output_combined, combined_memory_tasks(task_inputs, task_labels))
            LOGGER.info("wrote compatibility combined artifact: %s", args.output_combined)

        LOGGER.info("wrote inputs: %s", args.output_input)
        LOGGER.info("wrote labels: %s", args.output_labels)
        summary = build_run_summary(
            script="prepare_hotpotqa.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={
                "raw_examples": len(raw_records),
                "valid_examples": len(valid_raw_examples.records),
                "invalid_examples_dropped": invalid_examples_dropped,
                "invalid_example_reasons": valid_raw_examples.invalid_reason_counts,
                "selected_examples": len(selected_records),
                "parsed_examples": len(parsed_examples),
                "task_inputs": len(task_inputs),
                "task_labels": len(task_labels),
            },
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
            input_by_task_id = {converted_example.task_input["task_id"]: converted_example.task_input}
            validate_memory_task_inputs([converted_example.task_input])
            validate_memory_task_labels(
                [converted_example.task_labels],
                input_by_task_id,
            )
        except ValueError as error:
            if strict:
                raise ValueError(f"Invalid HotpotQA raw example index={record_index}: {error}") from error
            invalid_reason_counts[str(error)] += 1
            continue
        valid_records.append(raw_record)
    return ValidRawExamples(records=valid_records, invalid_reason_counts=dict(invalid_reason_counts))


def select_examples(raw_records: Sequence[object], *, max_examples: int | None, seed: int, offset: int) -> list[object]:
    if max_examples is None:
        if offset != 0:
            raise ValueError("--offset requires --max_examples so the split size is explicit.")
        return list(raw_records)
    return sample_split(raw_records, count=max_examples, seed=seed, offset=offset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labeled HotpotQA examples into leakage-safe memory task artifacts.")
    parser.add_argument("--input", required=True, help="Path to labeled HotpotQA raw JSON file.")
    parser.add_argument("--output_input", required=True, help="Path to write input-visible memory task JSON.")
    parser.add_argument("--output_labels", required=True, help="Path to write label-only memory task JSON.")
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
        input=namespace.input,
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
