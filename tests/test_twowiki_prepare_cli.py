from __future__ import annotations

import pytest

from graph_memory.experiment.persistence import write_yaml_atomic
from graph_memory.experiment.stage_models import PrepareOutputs, RawPrepareStageConfig
from graph_memory.experiment.state import read_stage_summary
from graph_memory.io import read_json, write_json
import scripts.prepare_2wiki as prepare_2wiki


def _raw_valid_example() -> dict[str, object]:
    return {
        "_id": "abc123",
        "type": "compositional",
        "question": "Who is Ada's mother?",
        "context": [
            ["Film A", ["Film A was directed by Ada.", "A distractor sentence."]],
            ["Ada Lovelace", ["Ada was the daughter of Beth."]],
        ],
        "supporting_facts": [["Film A", 0], ["Ada Lovelace", 0]],
        "evidences": [["Film A", "director", "Ada"], ["Ada", "mother", "Beth"]],
        "answer": "Beth",
    }


def test_prepare_2wiki_writes_separated_artifacts_and_summary(tmp_path) -> None:
    raw_path = tmp_path / "raw.json"
    input_path = tmp_path / "out" / "test.input.json"
    label_path = tmp_path / "out" / "test.labels.json"
    combined_path = tmp_path / "out" / "test.combined.json"
    summary_path = tmp_path / "out" / "test.input.run_summary.yaml"
    config_path = tmp_path / "prepare.yaml"
    write_json(raw_path, [_raw_valid_example(), {**_raw_valid_example(), "_id": ""}])
    _write_config(
        config_path,
        raw_path=raw_path,
        input_path=input_path,
        label_path=label_path,
        combined_path=combined_path,
        summary_path=summary_path,
    )

    exit_code = prepare_2wiki.main(["--config", str(config_path)])

    summary = read_stage_summary(summary_path)
    task_inputs = read_json(input_path)
    labels = read_json(label_path)
    combined = read_json(combined_path)

    assert exit_code == 0
    assert len(task_inputs) == 1
    assert len(labels) == 1
    assert "gold_answer" not in task_inputs[0]
    assert labels[0]["gold_dependency_edges"] == [["m0", "m2"]]
    assert combined[0]["gold_answer"] == "Beth"
    assert summary.status == "success"
    assert summary.counts["raw_examples"] == 2
    assert summary.counts["valid_examples"] == 1
    assert summary.counts["invalid_examples_dropped"] == 1
    assert summary.counts["path_supported_tasks"] == 1


def test_prepare_2wiki_strict_invalid_examples_fails_on_first_invalid_record(
    tmp_path,
) -> None:
    raw_path = tmp_path / "raw.json"
    input_path = tmp_path / "out" / "test.input.json"
    label_path = tmp_path / "out" / "test.labels.json"
    combined_path = tmp_path / "out" / "test.combined.json"
    summary_path = tmp_path / "out" / "test.input.run_summary.yaml"
    config_path = tmp_path / "prepare.yaml"
    write_json(raw_path, [{**_raw_valid_example(), "_id": ""}])
    _write_config(
        config_path,
        raw_path=raw_path,
        input_path=input_path,
        label_path=label_path,
        combined_path=combined_path,
        summary_path=summary_path,
        strict=True,
    )

    with pytest.raises(ValueError, match="Invalid 2Wiki raw example index=0"):
        prepare_2wiki.main(["--config", str(config_path)])

    summary = read_stage_summary(summary_path)
    assert summary.status == "failed"
    assert summary.error is not None
    assert "Invalid 2Wiki raw example" in summary.error.message


def _write_config(
    path,
    *,
    raw_path,
    input_path,
    label_path,
    combined_path,
    summary_path,
    strict: bool = False,
) -> None:
    write_yaml_atomic(
        path,
        RawPrepareStageConfig(
            stage="prepare",
            kind="raw",
            dataset="twowiki",
            split="test",
            source=raw_path,
            outputs=PrepareOutputs(
                input=input_path,
                labels=label_path,
                combined=combined_path,
                summary=summary_path,
            ),
            count=1,
            seed=13,
            offset=0,
            strict_invalid_examples=strict,
        ),
    )
