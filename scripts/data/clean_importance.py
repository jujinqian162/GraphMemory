from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTemporalMemoryRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
from graph_memory.infrastructure.io import write_json_atomic
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceArtifact,
    TaskImportanceRecord,
)
from graph_memory.validation import (
    ContractValidationError,
    validate_importance_artifact,
    validate_hotpotqa_ranking_records,
    validate_task_importance_record,
)

DEFAULT_TASKS = Path("data/hotpotqa/processed/dev_memory_tasks.input.json")
DEFAULT_INPUT = Path("data/hotpotqa/processed/memory_stream/dev.first_1000.gpt-5.4-mini.importance.json")
DEFAULT_OUTPUT = Path(
    "data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json"
)
DEFAULT_SUMMARY = Path("data/hotpotqa/processed/memory_stream/dev.first_1000.importance.cleaning_summary.json")

LEGACY_ARTIFACT_FIELDS = {"method", "model", "prompt_version", "generation", "tasks"}
LEGACY_TASK_FIELDS = {"task_id", "content_digest", "scores"}


@dataclass(frozen=True)
class CleanImportanceArgs:
    tasks: Path
    input: Path
    output: Path
    summary: Path
    limit: int


def read_json_strict(path: str | Path) -> object:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise ContractValidationError(f"Invalid JSON: duplicate key={key}.")
            record[key] = value
        return record

    with Path(path).open("r", encoding="utf-8") as file:
        loaded: object = json.load(  # pyright: ignore[reportAny]
            file,
            object_pairs_hook=reject_duplicate_keys,
        )
    return loaded


def normalize_task_scores(scores: Mapping[str, int]) -> dict[str, int]:
    levels = sorted(set(scores.values()))
    if len(levels) == 1:
        return {node_id: 5 for node_id in scores}
    normalized_by_level = {
        level: 1 + _round_half_up(9 * index / (len(levels) - 1))
        for index, level in enumerate(levels)
    }
    return {node_id: normalized_by_level[value] for node_id, value in scores.items()}


def clean_legacy_artifact(
    legacy_artifact: object,
    task_inputs: Sequence[HotpotQARankingRecord],
    *,
    source_path: Path,
    source_sha256: str,
) -> tuple[ImportanceArtifact, dict[str, object]]:
    legacy = _require_mapping(legacy_artifact, "legacy importance artifact")
    _reject_unknown(legacy, LEGACY_ARTIFACT_FIELDS, "legacy importance artifact")
    if legacy.get("method") != "memory_stream":
        raise ContractValidationError(
            "Invalid legacy importance artifact: method must be memory_stream."
        )
    raw_legacy_tasks = legacy.get("tasks")
    if not isinstance(raw_legacy_tasks, list):
        raise ContractValidationError(
            "Invalid legacy importance artifact: tasks must be a list."
        )
    legacy_tasks = cast(list[object], raw_legacy_tasks)
    if len(legacy_tasks) != len(task_inputs):
        raise ContractValidationError(
            f"Invalid legacy importance artifact: task count mismatch expected={len(task_inputs)} observed={len(legacy_tasks)}."
        )

    temporal_requests = _temporal_requests(task_inputs)
    output_records: list[TaskImportanceRecord] = []
    before_scores: list[int] = []
    after_scores: list[int] = []
    constant_tasks: list[str] = []
    narrow_range_tasks: list[str] = []
    score_vectors: dict[tuple[int, ...], list[str]] = {}

    for index, (legacy_record, task_input, temporal_request) in enumerate(
        zip(legacy_tasks, task_inputs, temporal_requests, strict=True)
    ):
        record = _require_mapping(
            legacy_record,
            f"legacy task importance record index={index}",
        )
        _reject_unknown(record, LEGACY_TASK_FIELDS, "legacy task importance record")
        if record.get("task_id") != task_input["task_id"]:
            raise ContractValidationError(
                f"Invalid legacy importance artifact: task order mismatch index={index} expected={task_input['task_id']} observed={record.get('task_id')}."
            )
        validate_task_importance_record(record, temporal_request)
        typed_record = cast(TaskImportanceRecord, record)
        raw_scores = typed_record["scores"]
        normalized_scores = normalize_task_scores(raw_scores)
        values = list(raw_scores.values())
        before_scores.extend(values)
        after_scores.extend(normalized_scores.values())
        if len(set(values)) == 1:
            constant_tasks.append(task_input["task_id"])
        if max(values) - min(values) <= 1:
            narrow_range_tasks.append(task_input["task_id"])
        vector = tuple(raw_scores[node_id] for node_id in sorted(raw_scores, key=_node_sort_key))
        score_vectors.setdefault(vector, []).append(task_input["task_id"])
        _validate_rank_preservation(raw_scores, normalized_scores, task_input["task_id"])
        output_records.append(
            {
                "task_id": typed_record["task_id"],
                "content_digest": typed_record["content_digest"],
                "scores": normalized_scores,
            }
        )

    artifact: ImportanceArtifact = {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": output_records,
    }
    validate_importance_artifact(artifact, temporal_requests)
    duplicate_vectors = [
        task_ids for task_ids in score_vectors.values() if len(task_ids) > 1
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "operation": "memory_stream_importance_task_rank_normalization",
        "normalization": {
            "scope": "task",
            "method": "unique_level_rank",
            "output_range": [1, 10],
            "rounding": "half_up",
            "constant_value": 5,
        },
        "source": {
            "path": source_path.as_posix(),
            "sha256": source_sha256,
            "legacy_metadata": {
                key: deepcopy(legacy[key])
                for key in ("model", "prompt_version", "generation")
                if key in legacy
            },
        },
        "counts": {
            "tasks": len(task_inputs),
            "candidate_sentences": len(before_scores),
            "constant_tasks": len(constant_tasks),
        },
        "validation": {
            "task_order": True,
            "content_digests": True,
            "node_coverage": True,
            "score_range": True,
            "rank_and_ties_preserved": True,
        },
        "distributions": {
            "before": _distribution(before_scores),
            "after": _distribution(after_scores),
            "source_shards_40_tasks": _shard_distributions(legacy_tasks, shard_size=40),
        },
        "anomalies": {
            "constant_task_ids": constant_tasks,
            "range_le_1_task_ids": narrow_range_tasks,
            "duplicate_score_vector_task_ids": duplicate_vectors,
        },
    }
    return artifact, summary


def _temporal_requests(records: Sequence[HotpotQARankingRecord]) -> list[TemporalMemoryRankingRequest]:
    projector = HotpotQAToTemporalMemoryRankingRequest()
    return [projector.project(record, {}) for record in records]


def parse_args(argv: Sequence[str] | None = None) -> CleanImportanceArgs:
    parser = argparse.ArgumentParser(
        description="Validate and task-rank normalize a legacy Memory Stream importance artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    _ = parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    _ = parser.add_argument("--limit", type=int, default=1000)
    namespace = parser.parse_args(argv)
    return CleanImportanceArgs(
        tasks=cast(Path, namespace.tasks),
        input=cast(Path, namespace.input),
        output=cast(Path, namespace.output),
        summary=cast(Path, namespace.summary),
        limit=cast(int, namespace.limit),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")

    loaded_tasks = read_json_strict(args.tasks)
    validate_hotpotqa_ranking_records(loaded_tasks)
    all_tasks = cast(list[HotpotQARankingRecord], loaded_tasks)
    task_inputs = all_tasks[: args.limit]
    legacy_artifact = read_json_strict(args.input)
    artifact, summary = clean_legacy_artifact(
        legacy_artifact,
        task_inputs,
        source_path=args.input,
        source_sha256=_file_sha256(args.input),
    )
    write_json_atomic(args.output, artifact)
    summary["output"] = {
        "path": args.output.as_posix(),
        "sha256": _file_sha256(args.output),
    }
    summary["tasks"] = {
        "path": args.tasks.as_posix(),
        "selected_prefix": len(task_inputs),
        "sha256": _file_sha256(args.tasks),
    }
    write_json_atomic(args.summary, summary)
    return 0


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _validate_rank_preservation(
    before: Mapping[str, int],
    after: Mapping[str, int],
    task_id: str,
) -> None:
    node_ids = list(before)
    for left in node_ids:
        for right in node_ids:
            if (before[left] == before[right]) != (after[left] == after[right]):
                raise ContractValidationError(
                    f"Invalid normalized importance: task_id={task_id} ties changed."
                )
            if (before[left] < before[right]) != (after[left] < after[right]):
                raise ContractValidationError(
                    f"Invalid normalized importance: task_id={task_id} rank changed."
                )


def _distribution(values: list[int]) -> dict[str, object]:
    counts = Counter(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_stdev": statistics.pstdev(values),
        "counts": {str(score): counts.get(score, 0) for score in range(1, 11)},
    }


def _shard_distributions(
    records: list[object],
    *,
    shard_size: int,
) -> list[dict[str, object]]:
    shards: list[dict[str, object]] = []
    for start in range(0, len(records), shard_size):
        values: list[int] = []
        for record in records[start : start + shard_size]:
            mapping = _require_mapping(record, "legacy task importance record")
            scores = _require_mapping(mapping.get("scores"), "legacy task scores")
            values.extend(cast(int, score) for score in scores.values())
        shards.append(
            {
                "start_task_index": start,
                "end_task_index": start + len(records[start : start + shard_size]) - 1,
                **_distribution(values),
            }
        )
    return shards


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_sort_key(node_id: str) -> tuple[int, str]:
    suffix = node_id[1:]
    return (int(suffix), node_id) if node_id.startswith("m") and suffix.isdigit() else (0, node_id)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"Invalid {name}: must be an object.")
    return cast(Mapping[str, object], value)


def _reject_unknown(
    record: Mapping[str, object],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ContractValidationError(f"Invalid {name}: unknown fields={unknown}.")


if __name__ == "__main__":
    raise SystemExit(main())
