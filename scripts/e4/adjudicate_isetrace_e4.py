from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

AUDIT_DIR = Path("results/isetrace/e4-audit")
BINARY_FIELDS = (
    "valid_query",
    "answerable",
    "gold_accuracy",
    "gold_completeness",
)
TYPE_FIELD = "reviewed_query_type"
NOTES_FIELD = "notes"
REVIEW_FIELDS = (*BINARY_FIELDS, TYPE_FIELD, NOTES_FIELD)
VALID_BINARY = {"yes", "no"}
VALID_TYPES = {
    "direct_recall",
    "linked_recall",
    "multi_fact_recall",
    "unclear",
}

ReviewRow: TypeAlias = dict[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate independent E4 reviews, initialize adjudication, and derive "
            "the fully verified task subset after final adjudication."
        )
    )
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--initialize", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument("--apply-decisions", type=Path)
    parser.add_argument(
        "--minimum-verified",
        type=int,
        default=300,
        help="Target minimum for the paper's fully verified subset.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.minimum_verified < 1:
        raise ValueError("minimum verified count must be positive")
    if args.initialize:
        _initialize(args.audit_dir)
    elif args.apply_decisions is not None:
        _apply_decisions(args.audit_dir, args.apply_decisions)
    else:
        _finalize(args.audit_dir, minimum_verified=args.minimum_verified)
    return 0


def _initialize(audit_dir: Path) -> None:
    identity = _load_identity(audit_dir / "sample_identity.jsonl")
    reviewer_a = _load_reviewer(audit_dir / "reviewer_a.csv", identity)
    reviewer_b = _load_reviewer(audit_dir / "reviewer_b.csv", identity)

    rows: list[ReviewRow] = []
    disagreement_counts: Counter[str] = Counter()
    full_agreement_count = 0
    for audit_id in identity:
        left = reviewer_a[audit_id]
        right = reviewer_b[audit_id]
        agreement = all(left[field] == right[field] for field in REVIEW_FIELDS)
        if agreement:
            full_agreement_count += 1
        row = {
            "audit_id": audit_id,
            "task_id": identity[audit_id]["task_id"],
            **{f"reviewer_a_{field}": left[field] for field in REVIEW_FIELDS},
            **{f"reviewer_b_{field}": right[field] for field in REVIEW_FIELDS},
            "adjudication_required": "no" if agreement else "yes",
            **{
                f"final_{field}": left[field] if left[field] == right[field] else ""
                for field in REVIEW_FIELDS
            },
            "adjudicator_notes": "",
        }
        rows.append(row)
        for field in REVIEW_FIELDS:
            if left[field] != right[field]:
                disagreement_counts[field] += 1

    _write_csv(
        audit_dir / "adjudication.csv", rows, fieldnames=_adjudication_fieldnames()
    )
    packets = _load_review_packets(audit_dir / "review_packet.jsonl", identity)
    disagreement_rows = [row for row in rows if row["adjudication_required"] == "yes"]
    _write_jsonl(
        audit_dir / "disagreement_packet.jsonl",
        (
            _disagreement_packet(row, packets[row["audit_id"]])
            for row in disagreement_rows
        ),
    )
    _write_jsonl(
        audit_dir / "disagreement_index.jsonl",
        (
            _disagreement_index(row, packets[row["audit_id"]])
            for row in disagreement_rows
        ),
    )
    _write_jsonl(
        audit_dir / "type_adjudication_queue.jsonl",
        (
            _type_adjudication_item(row, packets[row["audit_id"]])
            for row in disagreement_rows
            if row[f"reviewer_a_{TYPE_FIELD}"] != row[f"reviewer_b_{TYPE_FIELD}"]
        ),
    )
    _write_jsonl(
        audit_dir / "quality_adjudication_queue.jsonl",
        (
            _disagreement_index(row, packets[row["audit_id"]])
            for row in disagreement_rows
            if any(
                row[f"reviewer_a_{field}"] != row[f"reviewer_b_{field}"]
                for field in (*BINARY_FIELDS,)
            )
        ),
    )
    _write_json(
        audit_dir / "initial_adjudication_summary.json",
        {
            "schema_version": 1,
            "reviewer_kind": "independent model-agent review; not human annotation",
            "sample_count": len(identity),
            "full_record_agreement_count": full_agreement_count,
            "full_record_agreement_rate": full_agreement_count / len(identity),
            "field_disagreement_counts": dict(sorted(disagreement_counts.items())),
            "adjudication_csv": "adjudication.csv",
            "disagreement_packet": "disagreement_packet.jsonl",
            "disagreement_index": "disagreement_index.jsonl",
            "type_adjudication_queue": "type_adjudication_queue.jsonl",
            "quality_adjudication_queue": "quality_adjudication_queue.jsonl",
            "disagreement_count": len(disagreement_rows),
            "next_step": (
                "An adjudicator must fill every blank final_* field and leave the "
                "pre-filled consensus fields unchanged unless correcting an evident error."
            ),
        },
    )


def _apply_decisions(audit_dir: Path, decisions_path: Path) -> None:
    identity = _load_identity(audit_dir / "sample_identity.jsonl")
    loaded = _read_csv(audit_dir / "adjudication.csv")
    if tuple(loaded.fieldnames) != _adjudication_fieldnames():
        raise ValueError("adjudication CSV has wrong columns")
    rows = [dict(row) for row in loaded.records]
    if [row.get("audit_id") for row in rows] != list(identity):
        raise ValueError("adjudication CSV must preserve the frozen sample order")

    value = json.loads(decisions_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("adjudication decisions must be a JSON object")
    type_reviewer = value.get("type_reviewer")
    binary_final = value.get("binary_final")
    if not isinstance(type_reviewer, dict) or not isinstance(binary_final, dict):
        raise ValueError(
            "adjudication decisions require type_reviewer and binary_final objects"
        )
    if not all(
        isinstance(audit_id, str) and reviewer in {"reviewer_a", "reviewer_b"}
        for audit_id, reviewer in type_reviewer.items()
    ):
        raise ValueError("type_reviewer values must be reviewer_a or reviewer_b")
    if not all(
        isinstance(audit_id, str)
        and isinstance(fields, dict)
        and all(
            isinstance(field, str) and isinstance(result, str)
            for field, result in fields.items()
        )
        for audit_id, fields in binary_final.items()
    ):
        raise ValueError("binary_final must map audit IDs to string field decisions")

    expected_type_ids = {
        row["audit_id"]
        for row in rows
        if row[f"reviewer_a_{TYPE_FIELD}"] != row[f"reviewer_b_{TYPE_FIELD}"]
    }
    if set(type_reviewer) != expected_type_ids:
        raise ValueError("type_reviewer must cover exactly the type disagreements")
    expected_binary_fields = {
        (row["audit_id"], field)
        for row in rows
        for field in BINARY_FIELDS
        if row[f"reviewer_a_{field}"] != row[f"reviewer_b_{field}"]
    }
    provided_binary_fields = {
        (audit_id, field)
        for audit_id, fields in binary_final.items()
        for field in fields
    }
    if provided_binary_fields != expected_binary_fields:
        raise ValueError("binary_final must cover exactly the binary disagreements")

    for row in rows:
        audit_id = _required(row, "audit_id", path=audit_dir / "adjudication.csv")
        for field in REVIEW_FIELDS:
            current = row.get(f"final_{field}", "")
            expected = (
                row[f"reviewer_a_{field}"]
                if row[f"reviewer_a_{field}"] == row[f"reviewer_b_{field}"]
                else ""
            )
            if current != expected:
                raise ValueError(
                    "adjudication CSV must be freshly initialized before applying decisions"
                )

        decided_fields: list[str] = []
        for field in BINARY_FIELDS:
            reviewer_a_value = row[f"reviewer_a_{field}"]
            reviewer_b_value = row[f"reviewer_b_{field}"]
            if reviewer_a_value == reviewer_b_value:
                row[f"final_{field}"] = reviewer_a_value
                continue
            result = binary_final[audit_id][field]
            if result not in {reviewer_a_value, reviewer_b_value}:
                raise ValueError(
                    f"binary decision must select a reviewer value: {audit_id}/{field}"
                )
            row[f"final_{field}"] = result
            decided_fields.append(f"{field}={result}")

        reviewer_a_type = row[f"reviewer_a_{TYPE_FIELD}"]
        reviewer_b_type = row[f"reviewer_b_{TYPE_FIELD}"]
        if reviewer_a_type == reviewer_b_type:
            row[f"final_{TYPE_FIELD}"] = reviewer_a_type
        else:
            reviewer = type_reviewer[audit_id]
            row[f"final_{TYPE_FIELD}"] = row[f"{reviewer}_{TYPE_FIELD}"]
            decided_fields.append(
                f"{TYPE_FIELD}={row[f'final_{TYPE_FIELD}']} ({reviewer})"
            )

        row[f"final_{NOTES_FIELD}"] = _select_final_notes(row)
        if decided_fields:
            row["adjudicator_notes"] = (
                "Blind evidence-first adjudication: " + "; ".join(decided_fields)
            )
        elif row[f"reviewer_a_{NOTES_FIELD}"] != row[f"reviewer_b_{NOTES_FIELD}"]:
            row["adjudicator_notes"] = (
                "Reviewer notes differ; substantive labels agree."
            )
        else:
            row["adjudicator_notes"] = ""

    _write_csv(
        audit_dir / "adjudication.csv", rows, fieldnames=_adjudication_fieldnames()
    )
    _write_json(
        audit_dir / "adjudication_decision_log.json",
        {
            "schema_version": 1,
            "method": "blind evidence-first adjudication against frozen packets",
            "decisions_file": decisions_path.name,
            "decisions_sha256": _sha256(decisions_path),
            "type_decision_count": len(expected_type_ids),
            "binary_decision_count": len(expected_binary_fields),
        },
    )


def _select_final_notes(row: Mapping[str, str]) -> str:
    for field in (*BINARY_FIELDS, TYPE_FIELD):
        reviewer_a_value = row[f"reviewer_a_{field}"]
        reviewer_b_value = row[f"reviewer_b_{field}"]
        if reviewer_a_value == reviewer_b_value:
            continue
        final_value = row[f"final_{field}"]
        for reviewer, value in (
            ("reviewer_a", reviewer_a_value),
            ("reviewer_b", reviewer_b_value),
        ):
            if value == final_value and row[f"{reviewer}_{NOTES_FIELD}"]:
                return row[f"{reviewer}_{NOTES_FIELD}"]
    return row[f"reviewer_a_{NOTES_FIELD}"] or row[f"reviewer_b_{NOTES_FIELD}"]


def _finalize(audit_dir: Path, *, minimum_verified: int) -> None:
    identity = _load_identity(audit_dir / "sample_identity.jsonl")
    rows = _load_adjudication(audit_dir / "adjudication.csv", identity)

    verified_task_ids: list[str] = []
    final_rows: list[dict[str, object]] = []
    quality_yes_counts: Counter[str] = Counter()
    type_correct_count = 0
    status_counts: Counter[str] = Counter()
    for row in rows:
        audit_id = row["audit_id"]
        authored_type = identity[audit_id]["authored_memory_mode"]
        final_type = row[f"final_{TYPE_FIELD}"]
        quality_passes = {
            field: row[f"final_{field}"] == "yes" for field in BINARY_FIELDS
        }
        for field, passed in quality_passes.items():
            quality_yes_counts[field] += int(passed)
        type_correct = final_type == authored_type
        type_correct_count += int(type_correct)
        verified = all(quality_passes.values()) and type_correct
        status = "verified" if verified else "not_verified"
        status_counts[status] += 1
        if verified:
            verified_task_ids.append(row["task_id"])
        final_rows.append(
            {
                "audit_id": audit_id,
                "task_id": row["task_id"],
                "authored_memory_mode": authored_type,
                "reviewer_a": {
                    field: row[f"reviewer_a_{field}"] for field in REVIEW_FIELDS
                },
                "reviewer_b": {
                    field: row[f"reviewer_b_{field}"] for field in REVIEW_FIELDS
                },
                "final": {field: row[f"final_{field}"] for field in REVIEW_FIELDS},
                "adjudicator_notes": row["adjudicator_notes"],
                "status": status,
            }
        )

    _write_jsonl(audit_dir / "adjudication.jsonl", final_rows)
    _write_json(audit_dir / "verified_task_ids.json", sorted(verified_task_ids))
    sample_count = len(rows)
    _write_json(
        audit_dir / "adjudication_summary.json",
        {
            "schema_version": 1,
            "reviewer_kind": "independent model-agent review; not human annotation",
            "sample_count": sample_count,
            "quality_rates": {
                field: quality_yes_counts[field] / sample_count
                for field in BINARY_FIELDS
            },
            "query_type_accuracy": type_correct_count / sample_count,
            "status_counts": dict(sorted(status_counts.items())),
            "verified_task_count": len(verified_task_ids),
            "minimum_verified_target": minimum_verified,
            "meets_minimum_verified_target": len(verified_task_ids) >= minimum_verified,
            "verified_task_ids": "verified_task_ids.json",
            "adjudication_records": "adjudication.jsonl",
            "input_sha256": {
                "reviewer_a.csv": _sha256(audit_dir / "reviewer_a.csv"),
                "reviewer_b.csv": _sha256(audit_dir / "reviewer_b.csv"),
                "adjudication.csv": _sha256(audit_dir / "adjudication.csv"),
                "sample_identity.jsonl": _sha256(audit_dir / "sample_identity.jsonl"),
            },
        },
    )


def _disagreement_packet(
    row: ReviewRow, packet: Mapping[str, object]
) -> dict[str, object]:
    return {
        "audit_id": row["audit_id"],
        "task_id": row["task_id"],
        "query": packet["query"],
        "context": packet["context"],
        "reviewer_a": {field: row[f"reviewer_a_{field}"] for field in REVIEW_FIELDS},
        "reviewer_b": {field: row[f"reviewer_b_{field}"] for field in REVIEW_FIELDS},
        "prefilled_final": {field: row[f"final_{field}"] for field in REVIEW_FIELDS},
    }


def _disagreement_index(
    row: ReviewRow, packet: Mapping[str, object]
) -> dict[str, object]:
    context = packet["context"]
    if not isinstance(context, list):
        raise ValueError(f"invalid review context for {row['audit_id']}")
    outline: list[dict[str, object]] = []
    gold_evidence: list[dict[str, object]] = []
    for item in context:
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid review context item for {row['audit_id']}")
        handle = item.get("handle")
        kind = item.get("kind")
        quotes = item.get("gold_quotes")
        if (
            not isinstance(handle, str)
            or not isinstance(kind, str)
            or not isinstance(quotes, list)
        ):
            raise ValueError(f"invalid review context fields for {row['audit_id']}")
        outline.append({"handle": handle, "kind": kind, "is_gold": bool(quotes)})
        if quotes:
            gold_evidence.append(
                {"handle": handle, "kind": kind, "gold_quotes": quotes}
            )
    disputed_fields = [
        field
        for field in REVIEW_FIELDS
        if row[f"reviewer_a_{field}"] != row[f"reviewer_b_{field}"]
    ]
    return {
        "audit_id": row["audit_id"],
        "task_id": row["task_id"],
        "query": packet["query"],
        "disputed_fields": disputed_fields,
        "reviewer_a": {field: row[f"reviewer_a_{field}"] for field in disputed_fields},
        "reviewer_b": {field: row[f"reviewer_b_{field}"] for field in disputed_fields},
        "gold_evidence": gold_evidence,
        "context_outline": outline,
    }


def _type_adjudication_item(
    row: ReviewRow, packet: Mapping[str, object]
) -> dict[str, object]:
    return {
        "audit_id": row["audit_id"],
        "task_id": row["task_id"],
        "query": packet["query"],
        "reviewer_a_type": row[f"reviewer_a_{TYPE_FIELD}"],
        "reviewer_b_type": row[f"reviewer_b_{TYPE_FIELD}"],
        "reviewer_a_notes": row[f"reviewer_a_{NOTES_FIELD}"],
        "reviewer_b_notes": row[f"reviewer_b_{NOTES_FIELD}"],
    }


def _load_review_packets(
    path: Path, identity: Mapping[str, ReviewRow]
) -> dict[str, Mapping[str, object]]:
    packets: dict[str, Mapping[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"review packet row must be an object: {path}:{line_number}"
                )
            audit_id = value.get("audit_id")
            task_id = value.get("task_id")
            if not isinstance(audit_id, str) or not isinstance(task_id, str):
                raise ValueError(f"review packet lacks identity: {path}:{line_number}")
            expected = identity.get(audit_id)
            if expected is None or expected["task_id"] != task_id:
                raise ValueError(
                    f"review packet identity mismatch: {path}:{line_number}"
                )
            if audit_id in packets:
                raise ValueError(f"duplicate audit ID in review packet: {audit_id}")
            if not isinstance(value.get("query"), str) or not isinstance(
                value.get("context"), list
            ):
                raise ValueError(
                    f"review packet lacks review context: {path}:{line_number}"
                )
            packets[audit_id] = value
    if set(packets) != set(identity):
        missing = sorted(set(identity) - set(packets))
        extra = sorted(set(packets) - set(identity))
        raise ValueError(
            f"review packet coverage differs: missing={missing[:5]}, extra={extra[:5]}"
        )
    return packets


def _load_identity(path: Path) -> dict[str, ReviewRow]:
    rows: dict[str, ReviewRow] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"identity row must be an object: {path}:{line_number}"
                )
            row = _string_mapping(value, path=f"{path}:{line_number}")
            audit_id = _required(row, "audit_id", path=path)
            if audit_id in rows:
                raise ValueError(f"duplicate audit ID in identity file: {audit_id}")
            _required(row, "task_id", path=path)
            authored_type = _required(row, "authored_memory_mode", path=path)
            if authored_type not in VALID_TYPES - {"unclear"}:
                raise ValueError(
                    f"invalid authored type for {audit_id}: {authored_type!r}"
                )
            rows[audit_id] = row
    if not rows:
        raise ValueError(f"identity file is empty: {path}")
    return rows


def _load_reviewer(
    path: Path, identity: Mapping[str, ReviewRow]
) -> dict[str, ReviewRow]:
    rows = _read_csv(path)
    expected_fields = {"audit_id", "task_id", *REVIEW_FIELDS}
    if set(rows.fieldnames) != expected_fields:
        raise ValueError(f"reviewer CSV has wrong columns: {path}")
    reviewed: dict[str, ReviewRow] = {}
    for row in rows.records:
        audit_id = _required(row, "audit_id", path=path)
        if audit_id in reviewed:
            raise ValueError(f"duplicate audit ID in reviewer CSV: {audit_id}")
        expected = identity.get(audit_id)
        if expected is None:
            raise ValueError(f"unknown audit ID in reviewer CSV: {audit_id}")
        if row.get("task_id") != expected["task_id"]:
            raise ValueError(f"task ID mismatch for {audit_id} in {path}")
        normalized = dict(row)
        for field in BINARY_FIELDS:
            value = _normalize(row.get(field))
            if value not in VALID_BINARY:
                raise ValueError(f"invalid {field} for {audit_id} in {path}: {value!r}")
            normalized[field] = value
        query_type = _normalize(row.get(TYPE_FIELD))
        if query_type not in VALID_TYPES:
            raise ValueError(
                f"invalid {TYPE_FIELD} for {audit_id} in {path}: {query_type!r}"
            )
        normalized[TYPE_FIELD] = query_type
        normalized[NOTES_FIELD] = row.get(NOTES_FIELD, "").strip()
        requires_note = (
            any(normalized[field] == "no" for field in BINARY_FIELDS)
            or query_type == "unclear"
        )
        if requires_note and not normalized[NOTES_FIELD]:
            raise ValueError(
                f"missing rationale for no or unclear judgment: {audit_id}"
            )
        reviewed[audit_id] = normalized
    if set(reviewed) != set(identity):
        missing = sorted(set(identity) - set(reviewed))
        extra = sorted(set(reviewed) - set(identity))
        raise ValueError(
            f"reviewer CSV coverage differs: missing={missing[:5]}, extra={extra[:5]}"
        )
    if list(reviewed) != list(identity):
        raise ValueError("reviewer CSV must preserve the frozen sample order")
    return reviewed


def _load_adjudication(
    path: Path, identity: Mapping[str, ReviewRow]
) -> list[ReviewRow]:
    loaded = _read_csv(path)
    if tuple(loaded.fieldnames) != _adjudication_fieldnames():
        raise ValueError(f"adjudication CSV has wrong columns: {path}")
    rows: list[ReviewRow] = []
    seen: set[str] = set()
    for row in loaded.records:
        audit_id = _required(row, "audit_id", path=path)
        if audit_id in seen:
            raise ValueError(f"duplicate audit ID in adjudication CSV: {audit_id}")
        seen.add(audit_id)
        expected = identity.get(audit_id)
        if expected is None or row.get("task_id") != expected["task_id"]:
            raise ValueError(f"invalid identity for adjudication row: {audit_id}")
        normalized = dict(row)
        for field in BINARY_FIELDS:
            value = _normalize(row.get(f"final_{field}"))
            if value not in VALID_BINARY:
                raise ValueError(f"missing or invalid final {field} for {audit_id}")
            normalized[f"final_{field}"] = value
        final_type = _normalize(row.get(f"final_{TYPE_FIELD}"))
        if final_type not in VALID_TYPES:
            raise ValueError(f"missing or invalid final type for {audit_id}")
        normalized[f"final_{TYPE_FIELD}"] = final_type
        normalized[f"final_{NOTES_FIELD}"] = row.get(f"final_{NOTES_FIELD}", "").strip()
        normalized["adjudicator_notes"] = row.get("adjudicator_notes", "").strip()
        rows.append(normalized)
    expected_order = list(identity)
    if [row["audit_id"] for row in rows] != expected_order:
        raise ValueError("adjudication CSV must preserve the frozen sample order")
    return rows


class _CsvRows:
    def __init__(self, fieldnames: Sequence[str], records: Sequence[ReviewRow]) -> None:
        self.fieldnames = tuple(fieldnames)
        self.records = tuple(records)


def _read_csv(path: Path) -> _CsvRows:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is missing a header: {path}")
        return _CsvRows(reader.fieldnames, [dict(row) for row in reader])


def _adjudication_fieldnames() -> tuple[str, ...]:
    return (
        "audit_id",
        "task_id",
        *(f"reviewer_a_{field}" for field in REVIEW_FIELDS),
        *(f"reviewer_b_{field}" for field in REVIEW_FIELDS),
        "adjudication_required",
        *(f"final_{field}" for field in REVIEW_FIELDS),
        "adjudicator_notes",
    )


def _write_csv(
    path: Path, rows: Iterable[ReviewRow], *, fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _string_mapping(value: Mapping[str, object], *, path: str) -> ReviewRow:
    result: ReviewRow = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"expected string JSON fields at {path}")
        result[key] = item
    return result


def _required(row: Mapping[str, str], field: str, *, path: Path) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"missing {field} in {path}")
    return value


def _normalize(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
