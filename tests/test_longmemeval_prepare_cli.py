from __future__ import annotations

import pytest

from graph_memory.io import read_json, write_json
import scripts.prepare_longmemeval as prepare_longmemeval


def _raw_valid_example(raw_id: str = "example_001") -> dict[str, object]:
    return {
        "question_id": raw_id,
        "question_type": "single-session-user",
        "question": "Where did I say I planned to meet Alex?",
        "answer": "At the library.",
        "question_date": "2024-01-10T12:00:00",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2024-01-01T09:00:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Let's meet Alex at the library tomorrow.", "has_answer": True},
                {"role": "assistant", "content": "That sounds good."},
            ],
        ],
        "answer_session_ids": ["s1"],
    }


def _raw_without_turn_support() -> dict[str, object]:
    raw = _raw_valid_example("abstention_001")
    raw["haystack_sessions"] = [[{"role": "user", "content": "No precise support label."}]]
    raw["answer_session_ids"] = []
    return raw


def test_prepare_longmemeval_writes_separated_artifacts_and_summary(tmp_path) -> None:
    raw_path = tmp_path / "raw.json"
    input_path = tmp_path / "out" / "test.input.json"
    label_path = tmp_path / "out" / "test.labels.json"
    combined_path = tmp_path / "out" / "test.combined.json"
    write_json(raw_path, [_raw_valid_example(), _raw_without_turn_support()])

    exit_code = prepare_longmemeval.main(
        [
            "--input",
            str(raw_path),
            "--output_input",
            str(input_path),
            "--output_labels",
            str(label_path),
            "--output_combined",
            str(combined_path),
            "--max_examples",
            "1",
            "--seed",
            "13",
            "--offset",
            "0",
        ]
    )

    summary = read_json(input_path.with_name("test.input.run_summary.json"))
    task_inputs = read_json(input_path)
    labels = read_json(label_path)
    combined = read_json(combined_path)

    assert exit_code == 0
    assert len(task_inputs) == 1
    assert len(labels) == 1
    assert "gold_answer" not in task_inputs[0]
    assert task_inputs[0]["candidate_items"][0]["item_id"] == "m0"
    assert labels[0]["gold_support_item_ids"] == ["m0"]
    assert combined[0]["gold_answer"] == "At the library."
    assert summary["status"] == "success"
    assert summary["counts"]["raw_examples"] == 2
    assert summary["counts"]["valid_examples"] == 1
    assert summary["counts"]["invalid_examples_dropped"] == 1
    assert summary["counts"]["turn_support_tasks"] == 1


def test_prepare_longmemeval_strict_invalid_examples_fails_on_first_invalid_record(tmp_path) -> None:
    raw_path = tmp_path / "raw.json"
    input_path = tmp_path / "out" / "test.input.json"
    label_path = tmp_path / "out" / "test.labels.json"
    write_json(raw_path, [_raw_without_turn_support()])

    with pytest.raises(ValueError, match="Invalid LongMemEval raw example index=0"):
        prepare_longmemeval.main(
            [
                "--input",
                str(raw_path),
                "--output_input",
                str(input_path),
                "--output_labels",
                str(label_path),
                "--strict_invalid_examples",
            ]
        )

    summary = read_json(input_path.with_name("test.input.run_summary.json"))
    assert summary["status"] == "failed"
    assert "Invalid LongMemEval raw example" in summary["error"]
