from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from pydantic import JsonValue

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_dataset import ISETRACE_REVISION

SplitName: TypeAlias = Literal["train", "dev", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "dev", "test")
SPLIT_POLICY_VERSION = "isetrace-intent-components-v1"
DEFAULT_RATIOS: dict[SplitName, float] = {
    "train": 0.8,
    "dev": 0.1,
    "test": 0.1,
}
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TrajectoryIdentity:
    trajectory_id: str
    source_intent_ids: tuple[str, ...]
    intent_text_hashes: tuple[str, ...]
    source_file: str
    source_line: int


@dataclass(frozen=True)
class SplitAssignment:
    trajectory: TrajectoryIdentity
    group_id: str
    split: SplitName


class _UnionFind:
    def __init__(self) -> None:
        self._parents: dict[str, str] = {}
        self._ranks: dict[str, int] = {}

    def find(self, item: str) -> str:
        parent = self._parents.setdefault(item, item)
        if parent != item:
            self._parents[item] = self.find(parent)
        return self._parents[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self._ranks.get(left_root, 0)
        right_rank = self._ranks.get(right_root, 0)
        if left_rank < right_rank:
            left_root, right_root = right_root, left_root
            left_rank, right_rank = right_rank, left_rank
        self._parents[right_root] = left_root
        if left_rank == right_rank:
            self._ranks[left_root] = left_rank + 1


def normalized_intent_text_hash(text: str) -> str:
    normalized = _WHITESPACE.sub(" ", text.strip().casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_trajectory_identities(
    source_paths: Iterable[Path],
) -> tuple[list[TrajectoryIdentity], list[dict[str, JsonValue]]]:
    trajectories: list[TrajectoryIdentity] = []
    source_files: list[dict[str, JsonValue]] = []
    seen_trajectory_ids: set[str] = set()

    for source_path in sorted(source_paths):
        digest = hashlib.sha256()
        record_count = 0
        with source_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if not line.strip():
                    continue
                record_count += 1
                value = cast(object, json.loads(line))
                if not isinstance(value, dict):
                    raise ValueError(
                        f"ISETrace record must be an object at {source_path}:{line_number}"
                    )
                record = cast(dict[str, object], value)
                trajectory_id = _required_string(
                    record, "session_id", source_path, line_number
                )
                if trajectory_id in seen_trajectory_ids:
                    raise ValueError(f"duplicate ISETrace session_id: {trajectory_id}")
                seen_trajectory_ids.add(trajectory_id)
                source_intent_ids = _required_string_tuple(
                    record, "source_intent_ids", source_path, line_number
                )
                source_intents_value = record.get("source_intents")
                if not isinstance(source_intents_value, list) or not source_intents_value:
                    raise ValueError(
                        f"source_intents must be a non-empty list at "
                        f"{source_path}:{line_number}"
                    )
                embedded_ids: list[str] = []
                text_hashes: list[str] = []
                for intent_value in source_intents_value:
                    if not isinstance(intent_value, dict):
                        raise ValueError(
                            f"source intent must be an object at "
                            f"{source_path}:{line_number}"
                        )
                    intent = cast(dict[str, object], intent_value)
                    embedded_ids.append(
                        _required_string(intent, "intent_id", source_path, line_number)
                    )
                    text = _required_string(
                        intent, "natural_language_intent", source_path, line_number
                    )
                    text_hashes.append(normalized_intent_text_hash(text))
                if tuple(embedded_ids) != source_intent_ids:
                    raise ValueError(
                        f"source intent IDs do not match embedded intents at "
                        f"{source_path}:{line_number}"
                    )
                trajectories.append(
                    TrajectoryIdentity(
                        trajectory_id=trajectory_id,
                        source_intent_ids=source_intent_ids,
                        intent_text_hashes=tuple(text_hashes),
                        source_file=source_path.name,
                        source_line=line_number,
                    )
                )
        source_files.append(
            {
                "path": source_path.as_posix(),
                "bytes": source_path.stat().st_size,
                "records": record_count,
                "sha256": digest.hexdigest(),
            }
        )
    return trajectories, source_files


def build_split_assignments(
    trajectories: Sequence[TrajectoryIdentity],
    *,
    seed: int,
    ratios: dict[SplitName, float] | None = None,
) -> tuple[list[SplitAssignment], dict[str, object]]:
    active_ratios = dict(ratios or DEFAULT_RATIOS)
    _validate_ratios(active_ratios)
    if not trajectories:
        raise ValueError("cannot split an empty ISETrace trajectory collection")

    union_find = _UnionFind()
    for trajectory in trajectories:
        identity_nodes = [
            *(f"intent:{intent_id}" for intent_id in trajectory.source_intent_ids),
            *(f"text:{text_hash}" for text_hash in trajectory.intent_text_hashes),
        ]
        anchor = identity_nodes[0]
        for node in identity_nodes[1:]:
            union_find.union(anchor, node)

    components: dict[str, list[TrajectoryIdentity]] = defaultdict(list)
    for trajectory in trajectories:
        root = union_find.find(f"intent:{trajectory.source_intent_ids[0]}")
        components[root].append(trajectory)

    groups: list[tuple[str, list[TrajectoryIdentity]]] = []
    for component in components.values():
        intent_ids = sorted(
            {
                intent_id
                for trajectory in component
                for intent_id in trajectory.source_intent_ids
            }
        )
        group_digest = hashlib.sha256("\n".join(intent_ids).encode("utf-8"))
        groups.append((f"intent-component:{group_digest.hexdigest()[:20]}", component))

    targets = _largest_remainder_targets(len(trajectories), active_ratios)
    split_counts: dict[SplitName, int] = {
        split: 0 for split in SPLIT_NAMES
    }
    group_assignments: dict[str, SplitName] = {}
    ordered_groups = sorted(
        groups,
        key=lambda item: (
            -len(item[1]),
            _seeded_digest(seed, item[0]),
            item[0],
        ),
    )
    for group_id, component in ordered_groups:
        group_size = len(component)
        fitting: list[SplitName] = [
            split
            for split in SPLIT_NAMES
            if split_counts[split] + group_size <= targets[split]
        ]
        candidates: Sequence[SplitName] = fitting or SPLIT_NAMES
        selected: SplitName = max(
            candidates,
            key=lambda split: (
                (targets[split] - split_counts[split]) / targets[split],
                _seeded_digest(seed, f"{group_id}\0{split}"),
            ),
        )
        group_assignments[group_id] = selected
        split_counts[selected] += group_size

    group_by_trajectory_id = {
        trajectory.trajectory_id: group_id
        for group_id, component in groups
        for trajectory in component
    }
    assignments = [
        SplitAssignment(
            trajectory=trajectory,
            group_id=group_by_trajectory_id[trajectory.trajectory_id],
            split=group_assignments[group_by_trajectory_id[trajectory.trajectory_id]],
        )
        for trajectory in trajectories
    ]
    _validate_assignments(assignments)

    group_counts = Counter(group_assignments.values())
    component_sizes = [len(component) for _, component in groups]
    summary: dict[str, object] = {
        "trajectory_counts": dict(split_counts),
        "target_trajectory_counts": dict(targets),
        "group_counts": {
            split: group_counts[split] for split in SPLIT_NAMES
        },
        "total_groups": len(groups),
        "largest_group_trajectories": max(component_sizes),
        "shared_intent_overlap_across_splits": 0,
        "exact_normalized_intent_text_overlap_across_splits": 0,
    }
    return assignments, summary


def write_split_artifacts(
    output_dir: Path,
    assignments: Sequence[SplitAssignment],
    *,
    source_files: Sequence[dict[str, JsonValue]],
    source_revision: str,
    seed: int,
    ratios: dict[SplitName, float],
    summary: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_files: dict[str, dict[str, JsonValue]] = {}
    for split in SPLIT_NAMES:
        path = output_dir / f"{split}.jsonl"
        rows = sorted(
            (item for item in assignments if item.split == split),
            key=lambda item: item.trajectory.trajectory_id,
        )
        digest = hashlib.sha256()
        with path.open("wb") as handle:
            for item in rows:
                payload = {
                    "schema_version": 1,
                    "trajectory_id": item.trajectory.trajectory_id,
                    "group_id": item.group_id,
                    "source_intent_ids": item.trajectory.source_intent_ids,
                    "source_file": item.trajectory.source_file,
                    "source_line": item.trajectory.source_line,
                }
                line = (
                    json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
                handle.write(line)
                digest.update(line)
        assignment_files[split] = {
            "path": path.as_posix(),
            "records": len(rows),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }

    manifest = {
        "schema_version": 1,
        "dataset": "valiere/ISETrace",
        "source_revision": source_revision,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "seed": seed,
        "ratios": ratios,
        "grouping_policy": [
            "connected components over shared source_intent_ids",
            "connected components over exact normalized intent text hashes",
        ],
        "assignment_policy": (
            "whole components, descending component size, seeded deterministic "
            "deficit balancing"
        ),
        "source_files": list(source_files),
        "assignment_files": assignment_files,
        "summary": summary,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _required_string(
    mapping: dict[str, object], key: str, source: Path, line_number: int
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string at {source}:{line_number}")
    return value


def _required_string_tuple(
    mapping: dict[str, object], key: str, source: Path, line_number: int
) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list at {source}:{line_number}")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(
            f"{key} values must be non-empty strings at {source}:{line_number}"
        )
    return tuple(cast(list[str], value))


def _validate_ratios(ratios: dict[SplitName, float]) -> None:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"split ratios must define exactly {SPLIT_NAMES}")
    if any(value <= 0.0 for value in ratios.values()):
        raise ValueError("split ratios must be positive")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to one")


def _largest_remainder_targets(
    total: int, ratios: dict[SplitName, float]
) -> dict[SplitName, int]:
    exact = {split: total * ratios[split] for split in SPLIT_NAMES}
    targets: dict[SplitName, int] = {
        split: math.floor(exact[split]) for split in SPLIT_NAMES
    }
    remainder = total - sum(targets.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(exact[split] - targets[split]), SPLIT_NAMES.index(split)),
    )
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def _seeded_digest(seed: int, value: str) -> str:
    return hashlib.sha256(
        f"{SPLIT_POLICY_VERSION}\0seed={seed}\0{value}".encode("utf-8")
    ).hexdigest()


def _validate_assignments(assignments: Sequence[SplitAssignment]) -> None:
    trajectory_ids = [item.trajectory.trajectory_id for item in assignments]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("split assignments contain duplicate trajectory IDs")
    split_by_intent: dict[str, SplitName] = {}
    split_by_text_hash: dict[str, SplitName] = {}
    for item in assignments:
        for intent_id in item.trajectory.source_intent_ids:
            previous = split_by_intent.setdefault(intent_id, item.split)
            if previous != item.split:
                raise ValueError(f"source intent crosses splits: {intent_id}")
        for text_hash in item.trajectory.intent_text_hashes:
            previous = split_by_text_hash.setdefault(text_hash, item.split)
            if previous != item.split:
                raise ValueError(f"normalized intent text crosses splits: {text_hash}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic leakage-safe trajectory-level split for ISETrace."
        )
    )
    parser.add_argument(
        "--source_dir",
        default="data/isetrace/raw/trajectories",
        help="Directory containing trajectories-*.jsonl shards.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/isetrace/splits/v1",
        help="Directory for manifest.json and train/dev/test JSONL assignments.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--source_revision", default=ISETRACE_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_dir = Path(args.source_dir)
    source_paths = sorted(source_dir.glob("trajectories-*.jsonl"))
    if not source_paths:
        raise FileNotFoundError(f"no ISETrace trajectory shards found in {source_dir}")
    trajectories, source_files = load_trajectory_identities(source_paths)
    assignments, summary = build_split_assignments(
        trajectories,
        seed=args.seed,
    )
    manifest_path = write_split_artifacts(
        Path(args.output_dir),
        assignments,
        source_files=source_files,
        source_revision=args.source_revision,
        seed=args.seed,
        ratios=DEFAULT_RATIOS,
        summary=summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
