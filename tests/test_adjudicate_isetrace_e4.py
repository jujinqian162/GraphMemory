from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import pytest

from scripts.e4.adjudicate_isetrace_e4 import main

REVIEW_FIELDS = (
    "valid_query",
    "answerable",
    "gold_accuracy",
    "gold_completeness",
    "reviewed_query_type",
    "notes",
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    _ = path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_review(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = ["audit_id", "task_id", *REVIEW_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_adjudication_initializes_and_finalizes_verified_subset(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "sample_identity.jsonl",
        [
            {
                "audit_id": "E4-0001",
                "task_id": "q1",
                "graph_id": "g1",
                "authored_memory_mode": "direct_recall",
            },
            {
                "audit_id": "E4-0002",
                "task_id": "q2",
                "graph_id": "g2",
                "authored_memory_mode": "linked_recall",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "review_packet.jsonl",
        [
            {
                "audit_id": "E4-0001",
                "task_id": "q1",
                "query": "What happened?",
                "context": [],
            },
            {
                "audit_id": "E4-0002",
                "task_id": "q2",
                "query": "Why changed?",
                "context": [],
            },
        ],
    )
    reviewer_a = [
        {
            "audit_id": "E4-0001",
            "task_id": "q1",
            "valid_query": "yes",
            "answerable": "yes",
            "gold_accuracy": "yes",
            "gold_completeness": "yes",
            "reviewed_query_type": "direct_recall",
            "notes": "",
        },
        {
            "audit_id": "E4-0002",
            "task_id": "q2",
            "valid_query": "yes",
            "answerable": "no",
            "gold_accuracy": "yes",
            "gold_completeness": "no",
            "reviewed_query_type": "multi_fact_recall",
            "notes": "missing causal link",
        },
    ]
    reviewer_b = [
        dict(reviewer_a[0]),
        {
            "audit_id": "E4-0002",
            "task_id": "q2",
            "valid_query": "yes",
            "answerable": "yes",
            "gold_accuracy": "yes",
            "gold_completeness": "yes",
            "reviewed_query_type": "linked_recall",
            "notes": "",
        },
    ]
    _write_review(tmp_path / "reviewer_a.csv", reviewer_a)
    _write_review(tmp_path / "reviewer_b.csv", reviewer_b)

    assert main(["--audit-dir", str(tmp_path), "--initialize"]) == 0
    initial = cast(
        dict[str, object],
        json.loads((tmp_path / "initial_adjudication_summary.json").read_text()),
    )
    assert initial["full_record_agreement_count"] == 1
    assert initial["disagreement_count"] == 1
    disagreement = (tmp_path / "disagreement_packet.jsonl").read_text().splitlines()
    assert len(disagreement) == 1
    assert "authored_memory_mode" not in json.loads(disagreement[0])
    assert len((tmp_path / "disagreement_index.jsonl").read_text().splitlines()) == 1
    assert (
        len((tmp_path / "type_adjudication_queue.jsonl").read_text().splitlines()) == 1
    )
    assert (
        len((tmp_path / "quality_adjudication_queue.jsonl").read_text().splitlines())
        == 1
    )

    decisions_path = tmp_path / "decisions.json"
    _ = decisions_path.write_text(
        json.dumps(
            {
                "type_reviewer": {"E4-0002": "reviewer_b"},
                "binary_final": {
                    "E4-0002": {
                        "answerable": "yes",
                        "gold_completeness": "yes",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--audit-dir",
                str(tmp_path),
                "--apply-decisions",
                str(decisions_path),
            ]
        )
        == 0
    )

    assert (
        main(["--audit-dir", str(tmp_path), "--finalize", "--minimum-verified", "2"])
        == 0
    )
    assert json.loads((tmp_path / "verified_task_ids.json").read_text()) == ["q1", "q2"]
    summary = cast(
        dict[str, object],
        json.loads((tmp_path / "adjudication_summary.json").read_text()),
    )
    assert summary["meets_minimum_verified_target"] is True
    assert summary["query_type_accuracy"] == 1.0


def test_reviewer_rejects_adverse_judgment_without_rationale(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "sample_identity.jsonl",
        [
            {
                "audit_id": "E4-0001",
                "task_id": "q1",
                "graph_id": "g1",
                "authored_memory_mode": "direct_recall",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "review_packet.jsonl",
        [
            {
                "audit_id": "E4-0001",
                "task_id": "q1",
                "query": "What happened?",
                "context": [],
            }
        ],
    )
    reviewer = {
        "audit_id": "E4-0001",
        "task_id": "q1",
        "valid_query": "yes",
        "answerable": "yes",
        "gold_accuracy": "yes",
        "gold_completeness": "no",
        "reviewed_query_type": "direct_recall",
        "notes": "",
    }
    _write_review(tmp_path / "reviewer_a.csv", [reviewer])
    _write_review(tmp_path / "reviewer_b.csv", [reviewer])

    with pytest.raises(ValueError, match="missing rationale"):
        _ = main(["--audit-dir", str(tmp_path), "--initialize"])
