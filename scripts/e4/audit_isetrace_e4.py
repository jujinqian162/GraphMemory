from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringQueryRecord,
    parse_task_sections,
)

MemoryMode: TypeAlias = str
MODES: tuple[MemoryMode, ...] = (
    "direct_recall",
    "linked_recall",
    "multi_fact_recall",
)
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SAMPLE_SEED = 104_729

REVIEW_FIELDS = (
    "valid_query",
    "answerable",
    "gold_accuracy",
    "gold_completeness",
    "reviewed_query_type",
    "notes",
)

RUBRIC = """# E4 Review Rubric

This packet supports a blinded, two-reviewer semantic audit of the frozen
ISETrace test artifact. It does not alter the source authoring data or the
prepared dataset.

For each item, review the query and its complete authoring packet. Do not
inspect sample_identity.jsonl until both independent reviews are complete.

- valid_query: yes only if the question is natural, unambiguous, and a
  plausible later memory request rather than a benchmark-oriented prompt.
- answerable: yes only if the requested answer can be recovered from the
  supplied task-local evidence.
- gold_accuracy: yes only if every marked quote contributes answer-bearing
  evidence required by the query.
- gold_completeness: yes only if the marked quotes cover every fact needed
  to answer all parts of the query. Extra context alone is not gold.
- reviewed_query_type: independently classify as direct_recall,
  linked_recall, multi_fact_recall, or unclear.

Definitions for the blind type decision:

- direct_recall: one substantive fact or outcome from one localized event.
- linked_recall: a remembered clue asks for a related earlier source, later
  correction, reuse, decision, or outcome.
- multi_fact_recall: answering completely requires distinct facts from
  multiple events.

Use unclear when the query does not support a reliable type decision. Record
short evidence-grounded notes for every no or unclear judgment. A final
adjudicator should resolve only disagreements after both CSVs are frozen.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock a blinded E4 audit sample and structurally validate its "
            "frozen ISETrace test artifact."
        )
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path(
            "data/processed/datasets/isetrace/a729232bf650444a89572949522d8941"
        ),
    )
    parser.add_argument(
        "--query-source",
        type=Path,
        default=Path("data/isetrace/query-authoring/isetrace-v7-raw.jsonl"),
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=Path(
            "data/isetrace/query-authoring/isetrace-v7-raw.jsonl.metadata.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/isetrace/e4-audit"),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_size <= 0:
        raise ValueError("sample size must be positive")

    prepared_dir = args.prepared_dir
    manifest = _read_mapping(prepared_dir / "manifest.json")
    labels = _read_list(prepared_dir / "labels.json")
    prepared_metadata = _read_list(prepared_dir / "query_metadata.json")
    artifact_digest = _string(manifest.get("digest"), "manifest.digest")

    labels_by_id = _index_records(labels, id_key="task_id", label="labels")
    prepared_by_id = _index_records(
        prepared_metadata, id_key="task_id", label="prepared query metadata"
    )
    test_ids = set(labels_by_id)
    if set(prepared_by_id) != test_ids:
        _raise_set_difference(
            "labels", set(labels_by_id), "prepared metadata", set(prepared_by_id)
        )

    authored_metadata = _read_jsonl_index(args.query_metadata, id_key="query_id")
    missing_source_metadata = sorted(test_ids - set(authored_metadata))
    if missing_source_metadata:
        raise ValueError(
            f"test tasks missing authoring metadata: {missing_source_metadata[:5]}"
        )

    modes_by_id: dict[str, MemoryMode] = {}
    for task_id in sorted(test_ids):
        mode = _string(
            authored_metadata[task_id].get("memory_mode"), f"{task_id}.memory_mode"
        )
        if mode not in MODES:
            raise ValueError(f"task={task_id!r} has unsupported memory mode={mode!r}")
        modes_by_id[task_id] = mode
        source_graph_id = _string(
            authored_metadata[task_id].get("trajectory_id"), f"{task_id}.trajectory_id"
        )
        label_graph_id = _string(
            labels_by_id[task_id].get("graph_id"), f"{task_id}.graph_id"
        )
        prepared_graph_id = _string(
            prepared_by_id[task_id].get("graph_id"), f"{task_id}.prepared graph_id"
        )
        if source_graph_id != label_graph_id or label_graph_id != prepared_graph_id:
            raise ValueError(
                f"task={task_id!r} has graph identity drift: "
                f"source={source_graph_id!r}, label={label_graph_id!r}, "
                f"prepared={prepared_graph_id!r}"
            )

    sample_ids = _stratified_sample(
        modes_by_id=modes_by_id,
        sample_size=args.sample_size,
        seed=args.sample_seed,
    )
    source_records, structural = _validate_source_records(
        query_source=args.query_source,
        test_ids=test_ids,
        labels_by_id=labels_by_id,
        sample_ids=sample_ids,
    )
    if set(source_records) != sample_ids:
        _raise_set_difference(
            "sample", sample_ids, "source records", set(source_records)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(args.output_dir / "rubric.md", RUBRIC)
    _write_jsonl(
        args.output_dir / "review_packet.jsonl",
        _review_packet_rows(sample_ids=sample_ids, source_records=source_records),
    )
    _write_jsonl(
        args.output_dir / "sample_identity.jsonl",
        _sample_identity_rows(
            sample_ids=sample_ids,
            labels_by_id=labels_by_id,
            modes_by_id=modes_by_id,
        ),
    )
    _write_reviewer_csv(args.output_dir / "reviewer_a.csv", sample_ids)
    _write_reviewer_csv(args.output_dir / "reviewer_b.csv", sample_ids)

    sample_mode_counts = Counter(modes_by_id[task_id] for task_id in sample_ids)
    structural_report = {
        "schema_version": 1,
        "purpose": "pre-semantic structural validation for the E4 human audit",
        "test_artifact": {
            "artifact_dir": str(prepared_dir),
            "digest": artifact_digest,
            "task_count": len(test_ids),
            "trajectory_count": len(
                {
                    _string(record.get("graph_id"), "label.graph_id")
                    for record in labels_by_id.values()
                }
            ),
            "input_sha256": {
                "manifest.json": _sha256(prepared_dir / "manifest.json"),
                "labels.json": _sha256(prepared_dir / "labels.json"),
                "query_metadata.json": _sha256(prepared_dir / "query_metadata.json"),
                "authoring.jsonl": _sha256(args.query_source),
                "authoring_metadata.jsonl": _sha256(args.query_metadata),
            },
        },
        "checks": structural,
    }
    _write_json(args.output_dir / "structural_validation.json", structural_report)
    _write_json(
        args.output_dir / "sample_manifest.json",
        {
            "schema_version": 1,
            "purpose": "frozen stratified random sample for the E4 semantic audit",
            "test_artifact_digest": artifact_digest,
            "sample_size": len(sample_ids),
            "sample_seed": args.sample_seed,
            "strata": {mode: sample_mode_counts[mode] for mode in MODES},
            "reviewer_files": ["reviewer_a.csv", "reviewer_b.csv"],
            "blind_packet": "review_packet.jsonl",
            "identity_file": "sample_identity.jsonl",
            "rubric": "rubric.md",
        },
    )
    return 0


def _stratified_sample(
    *, modes_by_id: Mapping[str, MemoryMode], sample_size: int, seed: int
) -> set[str]:
    by_mode = {
        mode: sorted(task_id for task_id, value in modes_by_id.items() if value == mode)
        for mode in MODES
    }
    total = len(modes_by_id)
    if sample_size > total:
        raise ValueError(
            f"sample size exceeds test population: {sample_size} > {total}"
        )

    exact = {mode: sample_size * len(ids) / total for mode, ids in by_mode.items()}
    quotas = {mode: int(exact[mode]) for mode in MODES}
    remaining = sample_size - sum(quotas.values())
    for mode in sorted(
        MODES,
        key=lambda value: (exact[value] - quotas[value], value),
        reverse=True,
    )[:remaining]:
        quotas[mode] += 1

    rng = random.Random(seed)
    selected: set[str] = set()
    for mode in MODES:
        selected.update(rng.sample(by_mode[mode], quotas[mode]))
    if len(selected) != sample_size:
        raise AssertionError("stratified sample contains duplicate task IDs")
    return selected


def _validate_source_records(
    *,
    query_source: Path,
    test_ids: set[str],
    labels_by_id: Mapping[str, Mapping[str, object]],
    sample_ids: set[str],
) -> tuple[dict[str, AuthoringQueryRecord], dict[str, object]]:
    seen: set[str] = set()
    sample_records: dict[str, AuthoringQueryRecord] = {}
    failures: list[dict[str, str]] = []
    gold_count_mismatches = 0
    for line_number, value in _iter_jsonl(query_source):
        task_id = _string(value.get("id"), f"authoring line {line_number}.id")
        if task_id not in test_ids:
            continue
        if task_id in seen:
            failures.append({"task_id": task_id, "reason": "duplicate source record"})
            continue
        seen.add(task_id)
        try:
            record = AuthoringQueryRecord.model_validate(value)
            sections = {
                section.handle: section for section in parse_task_sections(record.text)
            }
            for gold in record.gold:
                source = sections[gold.source]
                if source.text.count(gold.quote) != 1:
                    raise ValueError(
                        f"gold quote is not unique in source={gold.source!r}"
                    )
            label_gold = labels_by_id[task_id].get("gold_evidence_spans")
            if not isinstance(label_gold, list):
                raise ValueError("label gold_evidence_spans must be a list")
            if len(label_gold) != len(record.gold):
                gold_count_mismatches += 1
                raise ValueError(
                    f"authoring/label gold count differs: {len(record.gold)} != {len(label_gold)}"
                )
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"task_id": task_id, "reason": str(error)})
            continue
        if task_id in sample_ids:
            sample_records[task_id] = record

    missing = sorted(test_ids - seen)
    if missing:
        failures.extend(
            {"task_id": task_id, "reason": "missing source record"}
            for task_id in missing
        )
    if failures:
        raise ValueError(f"source structural validation failed: {failures[:5]}")
    return sample_records, {
        "test_task_ids_in_labels": len(test_ids),
        "test_task_ids_in_prepared_metadata": len(test_ids),
        "test_task_ids_with_authoring_metadata": len(test_ids),
        "test_task_ids_found_in_authoring_source": len(seen),
        "authoring_records_schema_valid": len(seen),
        "gold_quote_unique_within_declared_source": len(seen),
        "authoring_gold_count_matches_compiled_label_count": len(seen)
        - gold_count_mismatches,
        "failed_records": failures,
    }


def _review_packet_rows(
    *, sample_ids: set[str], source_records: Mapping[str, AuthoringQueryRecord]
) -> Iterable[dict[str, object]]:
    for index, task_id in enumerate(sorted(sample_ids), start=1):
        record = source_records[task_id]
        gold_by_source: dict[str, list[str]] = {}
        for item in record.gold:
            gold_by_source.setdefault(item.source, []).append(item.quote)
        sections = parse_task_sections(record.text)
        yield {
            "audit_id": f"E4-{index:04d}",
            "task_id": task_id,
            "query": record.query,
            "context": [
                {
                    "handle": section.handle,
                    "kind": section.kind,
                    "text": section.text,
                    "gold_quotes": gold_by_source.get(section.handle, []),
                }
                for section in sections
            ],
        }


def _sample_identity_rows(
    *,
    sample_ids: set[str],
    labels_by_id: Mapping[str, Mapping[str, object]],
    modes_by_id: Mapping[str, MemoryMode],
) -> Iterable[dict[str, object]]:
    for index, task_id in enumerate(sorted(sample_ids), start=1):
        yield {
            "audit_id": f"E4-{index:04d}",
            "task_id": task_id,
            "graph_id": _string(
                labels_by_id[task_id].get("graph_id"), f"{task_id}.graph_id"
            ),
            "authored_memory_mode": modes_by_id[task_id],
        }


def _write_reviewer_csv(path: Path, sample_ids: set[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "task_id", *REVIEW_FIELDS)
        )
        writer.writeheader()
        for index, task_id in enumerate(sorted(sample_ids), start=1):
            writer.writerow({"audit_id": f"E4-{index:04d}", "task_id": task_id})


def _read_mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(Mapping[str, object], value)


def _read_list(path: Path) -> list[Mapping[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected JSON object array: {path}")
    return cast(list[Mapping[str, object]], value)


def _read_jsonl_index(path: Path, *, id_key: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for line_number, value in _iter_jsonl(path):
        record_id = _string(value.get(id_key), f"{path}:{line_number}.{id_key}")
        if record_id in result:
            raise ValueError(f"duplicate {id_key}={record_id!r} in {path}")
        result[record_id] = value
    return result


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield line_number, cast(Mapping[str, object], value)


def _index_records(
    records: Sequence[Mapping[str, object]], *, id_key: str, label: str
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, record in enumerate(records):
        record_id = _string(record.get(id_key), f"{label}[{index}].{id_key}")
        if record_id in result:
            raise ValueError(f"duplicate {id_key}={record_id!r} in {label}")
        result[record_id] = record
    return result


def _raise_set_difference(
    left_label: str, left: set[str], right_label: str, right: set[str]
) -> None:
    missing = sorted(left - right)
    extra = sorted(right - left)
    raise ValueError(
        f"{left_label}/{right_label} task IDs differ: "
        f"missing={missing[:5]}, extra={extra[:5]}"
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
