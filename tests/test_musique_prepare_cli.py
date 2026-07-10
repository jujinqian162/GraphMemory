from __future__ import annotations

import json

import pytest

from graph_memory.io import read_json
import scripts.prepare_musique as prepare_musique


def _raw_valid_example(raw_id: str = "2hop__1_2") -> dict[str, object]:
    return {
        "id": raw_id,
        "question": "Where was the director of Film A born?",
        "answer": "London",
        "answer_aliases": ["London, England"],
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "Film A", "paragraph_text": "Film A was directed by Ada.", "is_supporting": True},
            {
                "idx": 1,
                "title": "Ada Lovelace",
                "paragraph_text": "Ada Lovelace was born in London.",
                "is_supporting": True,
            },
        ],
        "question_decomposition": [
            {"id": "1", "question": "Who directed Film A?", "answer": "Ada Lovelace", "paragraph_support_idx": 0},
            {"id": "2", "question": "Where was #1 born?", "answer": "London", "paragraph_support_idx": 1},
        ],
    }


def _write_jsonl(path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_prepare_musique_writes_separated_artifacts_and_summary(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    input_path = tmp_path / "out" / "test.input.json"
    label_path = tmp_path / "out" / "test.labels.json"
    combined_path = tmp_path / "out" / "test.combined.json"
    _write_jsonl(raw_path, [_raw_valid_example(), {**_raw_valid_example("bad"), "id": ""}])

    exit_code = prepare_musique.main(
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
    assert "is_supporting" not in task_inputs[0]["candidate_paragraphs"][0]
    assert labels[0]["gold_dependency_edges"] == [["p0", "p1"]]
    assert combined[0]["gold_answer"] == "London"
    assert summary["status"] == "success"
    assert summary["counts"]["raw_examples"] == 2
    assert summary["counts"]["valid_examples"] == 1
    assert summary["counts"]["invalid_examples_dropped"] == 1
    assert summary["counts"]["path_supported_tasks"] == 1


def test_prepare_musique_strict_invalid_examples_fails_on_first_invalid_record(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    input_path = tmp_path / "out" / "test.input.json"
    label_path = tmp_path / "out" / "test.labels.json"
    _write_jsonl(raw_path, [{**_raw_valid_example(), "id": ""}])

    with pytest.raises(ValueError, match="Invalid MuSiQue raw example index=0"):
        prepare_musique.main(
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
    assert "Invalid MuSiQue raw example" in summary["error"]
