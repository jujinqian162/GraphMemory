from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hydra import compose, initialize_config_dir
from prefect import flow
import pytest

import graph_memory.experiment.tasks as experiment_tasks
import graph_memory.experiment.workflow as experiment_workflow
from graph_memory.experiment.artifacts import FileSourceRef, identify_external_source
from graph_memory.experiment.config import (
    DenseFinetuneStageConfig,
    PairBuildConfig,
    PrepareSplitConfig,
    ProvenanceRgcnStageConfig,
    RgcnTrainStageConfig,
    TrainableRankingConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.tasks import prefect_storage_settings, prepare_split_task


ROOT = Path(__file__).resolve().parents[1]


class PairInputsCaptured(Exception):
    pass


def test_scientific_tasks_have_no_retry_or_lock_layer() -> None:
    reusable_tasks: tuple[Any, ...] = (
        experiment_tasks.prepare_split_task,
        experiment_tasks.build_evidence_graphs_task,
        experiment_tasks.build_training_pairs_task,
        experiment_tasks.train_dense_ft_task,
        experiment_tasks.train_evidence_rgcn_task,
        experiment_tasks.train_provenance_rgcn_task,
        experiment_tasks.generate_rankings_task,
        experiment_tasks.evaluate_rankings_task,
    )

    assert not hasattr(experiment_tasks, "CrossProcessFileSystemLockManager")
    assert not hasattr(experiment_tasks, "_retry_transient")
    for prefect_task in reusable_tasks:
        assert prefect_task.retries == 0
        assert prefect_task.retry_condition_fn is None


def test_task_signatures_use_precise_stage_inputs() -> None:
    pair_inputs = inspect.signature(
        experiment_tasks.build_training_pairs_task.fn
    ).parameters
    train_inputs = inspect.signature(
        experiment_tasks.train_evidence_rgcn_task.fn
    ).parameters
    ranking_inputs = inspect.signature(
        experiment_tasks.generate_rankings_task.fn
    ).parameters

    assert "method" not in pair_inputs
    assert {"prepared", "evidence_graphs", "dataset", "config"} <= set(pair_inputs)
    assert "runtime_identity" not in train_inputs
    assert "variant" not in train_inputs
    assert "runtime_identity" not in ranking_inputs
    assert "method" in ranking_inputs


def test_stage_configs_exclude_unrelated_method_settings() -> None:
    assert set(PairBuildConfig.model_fields) == {"sampling", "encoder", "device"}
    assert set(DenseFinetuneStageConfig.model_fields) == {
        "method",
        "encoder",
        "train",
    }
    assert set(RgcnTrainStageConfig.model_fields) == {
        "method",
        "variant",
        "encoder",
        "train",
    }
    assert set(ProvenanceRgcnStageConfig.model_fields) == {
        "method",
        "variant",
        "encoder",
        "train",
    }
    assert set(TrainableRankingConfig.model_fields) == {"method", "variant"}


def test_flow_calls_tasks_directly_without_forwarding_or_state_mirrors() -> None:
    owned_helpers = {
        name
        for name, value in vars(experiment_workflow).items()
        if inspect.isfunction(value) and value.__module__ == experiment_workflow.__name__
    }
    assert owned_helpers == {
        "_prepare_config",
        "_resolve_split_sources",
        "_direct_split_source",
        "_transform_split_sources",
        "_transform_encoder_source",
        "_unique_assets",
    }

    source = inspect.getsource(experiment_workflow.run_experiment.fn)
    for task_name in (
        "prepare_split_task",
        "build_training_pairs_task",
        "train_dense_ft_task",
        "train_evidence_rgcn_task",
        "generate_rankings_task",
        "evaluate_rankings_task",
    ):
        assert f"{task_name}(" in source
    assert "return_state" not in source
    assert ".with_options(" not in source
    assert "task_runs" not in source


@pytest.mark.parametrize(
    ("dataset", "expects_graph", "expected_graph_neighbors"),
    (
        ("twowiki_provenance", False, 0),
        ("hotpotqa", True, 1),
    ),
)
def test_dense_ft_flow_uses_family_compatible_pair_inputs(
    monkeypatch,
    tmp_path: Path,
    dataset: str,
    expects_graph: bool,
    expected_graph_neighbors: int,
) -> None:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=dense-ft-pair-contract",
                f"dataset={dataset}",
                "profile=smoke",
                "device=cpu",
                "method=dense_ft",
            ],
        )
    config = resolve_experiment_config(
        parse_composed_config(composed),
        repository_root=ROOT,
    )
    observed: dict[str, object] = {}
    graph_artifact = object()

    monkeypatch.setattr(
        experiment_workflow,
        "prefect_storage_settings",
        lambda *, refresh_cache: nullcontext(),
    )
    monkeypatch.setattr(
        experiment_workflow,
        "_resolve_split_sources",
        lambda config: {split: object() for split in ("train", "dev", "test")},
    )
    monkeypatch.setattr(
        experiment_workflow,
        "prepare_split_task",
        lambda *, source, config: SimpleNamespace(
            artifact=object(),
            split=config.split,
        ),
    )
    monkeypatch.setattr(
        experiment_workflow,
        "resolve_encoder_source",
        lambda encoder: object(),
    )

    def capture_graph(**kwargs):
        observed["built_graph"] = True
        return SimpleNamespace(artifact=graph_artifact)

    def capture_pairs(**kwargs):
        observed["evidence_graphs"] = kwargs["evidence_graphs"]
        observed["config"] = kwargs["config"]
        raise PairInputsCaptured

    monkeypatch.setattr(
        experiment_workflow,
        "build_evidence_graphs_task",
        capture_graph,
    )
    monkeypatch.setattr(
        experiment_workflow,
        "build_training_pairs_task",
        capture_pairs,
    )

    with pytest.raises(PairInputsCaptured):
        experiment_workflow.run_experiment.fn(
            config,
            run_output=tmp_path / "run",
        )

    pair_config = observed["config"]
    assert isinstance(pair_config, PairBuildConfig)
    assert observed.get("built_graph", False) is expects_graph
    assert observed["evidence_graphs"] is (graph_artifact if expects_graph else None)
    assert (
        pair_config.sampling.hard_graph_neighbor_per_positive
        == expected_graph_neighbors
    )


def test_prepare_task_reuses_cache_and_flow_scoped_refreshes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    monkeypatch.setattr(experiment_tasks, "PROCESSED_ROOT", processed)
    monkeypatch.setattr(
        experiment_tasks,
        "SCIENTIFIC_RESULT_STORAGE",
        processed / "prefect" / "results",
    )
    source = identify_external_source(
        ROOT / "tests" / "fixtures" / "hotpotqa_smoke.json",
        repository_root=ROOT,
    )
    assert isinstance(source, FileSourceRef)
    config = PrepareSplitConfig(
        dataset="hotpotqa",
        split="test",
        count=1,
        offset=0,
        seed=13,
        strict_invalid_examples=False,
    )
    implementation_version = f"test-{tmp_path.parent.name}-{tmp_path.name}"

    @flow(name="prepare-cache-contract", persist_result=False)
    def invoke(refresh: bool) -> str:
        with prefect_storage_settings(refresh_cache=refresh):
            state = prepare_split_task(
                source=source,
                config=config,
                implementation_version=implementation_version,
                return_state=True,
            )
        return state.name or str(state.type)

    assert invoke(False) == "Completed"
    assert invoke(False) == "Cached"
    assert invoke(True) == "Completed"
