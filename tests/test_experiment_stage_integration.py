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
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.experiment.service import initialize_experiment
from graph_memory.experiment.stage_models import PairStageConfig
from graph_memory.experiment.status import inspect_invocation_status
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.datasets.twowiki_provenance import convert_twowiki_source_records

ROOT = Path(__file__).resolve().parents[1]


def _smoke_config(name: str, *, methods: str = "[bm25]"):
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
                f"methods={methods}",
                "device=cpu",
                *source_overrides,
                *window_overrides,
            ],
        )
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )


def test_evidence_dense_ft_keeps_graph_pair_sampling(tmp_path: Path) -> None:
    config = _smoke_config(
        f"evidence-dense-ft-plan-{tmp_path.name}", methods="[dense_ft]"
    )
    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )
    pair = next(
        invocation
        for invocation in plan.invocations
        if invocation.identifier == "pairs:dense_ft"
    )

    assert isinstance(pair.config, PairStageConfig)
    assert pair.config.evidence_graphs is not None
    assert pair.config.sampling.hard_graph_neighbor_per_positive == 1
    assert any(input_ref.role == "evidence_graphs" for input_ref in pair.inputs)
    assert "evidence_graphs:train" in pair.dependencies


def test_prepare_binding_uses_file_source(
    tmp_path: Path,
) -> None:
    config = _smoke_config(f"file-source-{tmp_path.name}")
    layout = RunLayout(ROOT, config.name)
    initialized = initialize_experiment(config, layout=layout)
    try:
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


def test_twowiki_provenance_dense_ft_executes_graphless_typed_workflow(
    tmp_path: Path,
) -> None:
    source = tmp_path / "twowiki_provenance.json"
    converted = convert_twowiki_source_records(
        [_twowiki_provenance_source_record()], candidate_cap=6, seed=13
    )
    assert len(converted.records) == 1
    source.write_text(json.dumps(converted.records), encoding="utf-8")
    name = f"typed-twowiki-provenance-dense-ft-{tmp_path.name}"
    config = _provenance_dense_ft_config(name, source=source)
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
                f"{invocation.identifier}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
            assert inspect_invocation_status(invocation).state == "complete"

        assert layout.train_pairs(RetrievalMethodId.DENSE_FT).is_file()
        assert layout.checkpoint(
            RetrievalMethodId.DENSE_FT, kind="directory"
        ).is_dir()
        assert layout.prediction(RetrievalMethodId.DENSE_FT).is_file()
        assert layout.metric(RetrievalMethodId.DENSE_FT).is_file()
        assert layout.table("main").is_file()
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


def _provenance_dense_ft_config(name: str, *, source: Path):
    source_value = source.resolve().as_posix()
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "dataset=twowiki_provenance",
                "profile=smoke",
                "methods=[dense_ft]",
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


def _twowiki_provenance_source_record() -> dict[str, object]:
    return {
        "_id": "dense-ft-smoke",
        "type": "compositional",
        "question": "In which country is the birthplace of Alpha located?",
        "context": [
            ["Alpha", ["Alpha was born in Bridge City.", "Alpha is an artist."]],
            [
                "Bridge City",
                ["Bridge City is located in Country Z.", "It has a river."],
            ],
            ["Wrong One", ["Wrong One is located elsewhere."]],
            ["Wrong Two", ["Wrong Two mentions Alpha but not the answer."]],
        ],
        "supporting_facts": [["Alpha", 0], ["Bridge City", 0]],
        "evidences": [
            ["Alpha", "birth place", "Bridge City"],
            ["Bridge City", "country", "Country Z"],
        ],
        "evidences_id": [],
        "answer_id": "Country Z",
        "answer": "Country Z",
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
