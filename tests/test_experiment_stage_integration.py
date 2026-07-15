from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path

from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.service import initialize_experiment
from graph_memory.experiment.status import inspect_invocation_status

ROOT = Path(__file__).resolve().parents[1]


def _smoke_config(name: str):
    source = (ROOT / "tests/fixtures/hotpotqa_smoke.json").as_posix()
    source_overrides = [
        f"dataset.splits.{split}.source={source}" for split in ("train", "dev", "test")
    ]
    window_overrides = [
        override
        for split in ("train", "dev", "test")
        for override in (
            f"dataset.splits.{split}.offset=0",
            f"dataset.splits.{split}.capacity=1",
        )
    ]
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "dataset=hotpotqa",
                "profile=smoke",
                "methods=[bm25]",
                "device=cpu",
                *source_overrides,
                *window_overrides,
            ],
        )
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )


def test_file_dataset_source_kind_is_preserved_in_prepare_binding(
    tmp_path: Path,
) -> None:
    config = _smoke_config(f"source-kind-{tmp_path.name}")
    layout = RunLayout(ROOT, config.name)
    initialized = initialize_experiment(config, layout=layout)
    try:
        assert config.dataset.source_kind == "file"
        prepare = next(
            invocation
            for invocation in initialized.plan.invocations
            if invocation.identifier == "prepare:train"
        )
        assert prepare.inputs[0].kind == "file"
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_hotpotqa_bm25_executes_all_typed_stage_yaml_subprocesses(
    tmp_path: Path,
) -> None:
    config = _smoke_config(f"typed-bm25-{tmp_path.name}")
    layout = RunLayout(ROOT, config.name)
    initialized = initialize_experiment(
        config,
        layout=layout,
    )
    try:
        assert all(
            invocation.stage != "evidence_graphs"
            for invocation in initialized.plan.invocations
        )
        for invocation in initialized.plan.invocations:
            completed = subprocess.run(
                invocation.argv,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, (
                f"{invocation.identifier}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
            row = inspect_invocation_status(invocation)
            assert row.state == "complete", row

        assert initialized.layout.table("main").is_file()
        assert initialized.layout.table("path").is_file()
        assert initialized.layout.table("efficiency").is_file()
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_twowiki_and_musique_graph_workflows_execute_typed_stage_subprocesses(
    tmp_path: Path,
) -> None:
    for dataset, record in (
        ("twowiki", _twowiki_record()),
        ("musique", _musique_record()),
    ):
        suffix = "jsonl" if dataset == "musique" else "json"
        source = tmp_path / f"{dataset}.{suffix}"
        if dataset == "musique":
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
        else:
            source.write_text(json.dumps([record]), encoding="utf-8")
        name = f"typed-{dataset}-{tmp_path.name}"
        config = _cross_dataset_config(name, dataset=dataset, source=source)
        layout = RunLayout(ROOT, name)
        initialized = initialize_experiment(config, layout=layout)
        try:
            assert all(
                invocation.stage != "evidence_graphs"
                for invocation in initialized.plan.invocations
            )
            for invocation in initialized.plan.invocations:
                completed = subprocess.run(
                    invocation.argv,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                assert completed.returncode == 0, (
                    f"{dataset}:{invocation.identifier}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
                assert inspect_invocation_status(invocation).state == "complete"
            assert layout.table("main").is_file()
            assert layout.table("path").is_file()
            assert layout.table("efficiency").is_file()
        finally:
            shutil.rmtree(layout.run_dir, ignore_errors=True)


def _cross_dataset_config(name: str, *, dataset: str, source: Path):
    source_value = source.resolve().as_posix()
    dataset_group = "2wiki" if dataset == "twowiki" else dataset
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                f"dataset={dataset_group}",
                "profile=smoke",
                "methods=[bm25,graphrag]",
                "device=cpu",
                *[
                    f"dataset.splits.{split}.source={source_value}"
                    for split in ("train", "dev", "test")
                ],
                *[
                    override
                    for split in ("train", "dev", "test")
                    for override in (
                        f"dataset.splits.{split}.offset=0",
                        f"dataset.splits.{split}.capacity=1",
                    )
                ],
            ],
        )
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )


def _twowiki_record() -> dict[str, object]:
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


def _musique_record() -> dict[str, object]:
    return {
        "id": "2hop__1_2",
        "question": "Where was the director of Film A born?",
        "answer": "London",
        "answer_aliases": ["London, England"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Film A",
                "paragraph_text": "Film A was directed by Ada.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Ada Lovelace",
                "paragraph_text": "Ada Lovelace was born in London.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "Distractor",
                "paragraph_text": "This paragraph is not useful.",
                "is_supporting": False,
            },
        ],
        "question_decomposition": [
            {
                "id": 1,
                "question": "Who directed Film A?",
                "answer": "Ada Lovelace",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "Where was #1 born?",
                "answer": "London",
                "paragraph_support_idx": 1,
            },
        ],
    }
