from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from hydra import compose, initialize_config_dir
from prefect import flow
import pytest

import graph_memory.experiment.tasks as experiment_tasks
import graph_memory.experiment.workflow as experiment_workflow
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPayload,
    DatasetArtifactRef,
    FileSourceRef,
    identify_external_source,
)
from graph_memory.experiment.cache import ScientificInputs
from graph_memory.experiment.config import (
    DenseEncoderConfig,
    PairBuildConfig,
    NegativeSamplingConfig,
    PrepareSplitConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.tasks import prefect_storage_settings, prepare_split_task


ROOT = Path(__file__).resolve().parents[1]


class PairInputsCaptured(Exception):
    pass


@pytest.mark.parametrize(
    ("dataset", "expects_graph", "expected_graph_neighbors"),
    (
        ("hotpotqa", True, 1),
        ("isetrace", False, 0),
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
        lambda *, source, config, trajectory_source=None: object(),
    )
    monkeypatch.setattr(
        experiment_workflow,
        "resolve_encoder_source",
        lambda encoder: object(),
    )

    def capture_graph(**kwargs):
        observed["built_graph"] = True
        return graph_artifact

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
    assert pair_config.method == "dense_ft"
    assert observed.get("built_graph", False) is expects_graph
    assert observed["evidence_graphs"] is (graph_artifact if expects_graph else None)
    assert (
        pair_config.sampling.hard_graph_neighbor_per_positive
        == expected_graph_neighbors
    )


def test_provenance_rgcn_flow_plans_trainable_lifecycle_without_evidence_graphs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=provenance-rgcn-plan",
                "dataset=isetrace",
                "profile=smoke",
                "device=cpu",
                "method=provenance_rgcn",
            ],
        )
    config = resolve_experiment_config(
        parse_composed_config(composed), repository_root=ROOT
    )
    observed: dict[str, object] = {}
    prepared_splits: list[str] = []

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
        "_trajectory_source",
        lambda config: object(),
    )

    def prepare(**kwargs):
        prepared_splits.append(kwargs["config"].split)
        assert kwargs["trajectory_source"] is not None
        return object()

    def pairs(**kwargs):
        observed["pair_graphs"] = kwargs["evidence_graphs"]
        return object()

    def encode(**kwargs):
        observed["encode_graphs"] = (
            kwargs["train_graphs"],
            kwargs["dev_graphs"],
        )
        return object()

    def train(**kwargs):
        observed["trained"] = True
        raise PairInputsCaptured

    monkeypatch.setattr(experiment_workflow, "prepare_split_task", prepare)
    monkeypatch.setattr(
        experiment_workflow, "resolve_encoder_source", lambda encoder: object()
    )
    monkeypatch.setattr(experiment_workflow, "build_training_pairs_task", pairs)
    monkeypatch.setattr(experiment_workflow, "encode_frozen_rgcn_embeddings_task", encode)
    monkeypatch.setattr(experiment_workflow, "train_provenance_rgcn_task", train)
    monkeypatch.setattr(
        experiment_workflow,
        "build_evidence_graphs_task",
        lambda **kwargs: pytest.fail("provenance R-GCN must not build EvidenceGraph"),
    )

    with pytest.raises(PairInputsCaptured):
        experiment_workflow.run_experiment.fn(
            config,
            run_output=tmp_path / "run",
        )

    assert prepared_splits == ["train", "dev", "test"]
    assert observed["pair_graphs"] is None
    assert observed["encode_graphs"] == (None, None)
    assert observed["trained"] is True


def test_scientific_cache_key_excludes_nested_runtime_device() -> None:
    sampling = NegativeSamplingConfig(
        random_seed=13,
        easy_random_per_positive=1,
        hard_bm25_per_positive=1,
        hard_dense_per_positive=1,
        hard_graph_neighbor_per_positive=0,
        hard_pool_size=10,
    )
    encoder = DenseEncoderConfig(
        model_name="model@revision",
        query_prefix="query: ",
        passage_prefix="passage: ",
        batch_size=64,
    )
    cpu = PairBuildConfig(
        method="dense_ft",
        sampling=sampling,
        encoder=encoder,
        device="cpu",
    )
    cuda = cpu.model_copy(update={"device": "cuda:7"})
    policy = ScientificInputs()

    cpu_key = policy.compute_key(None, {"config": cpu}, {})
    cuda_key = policy.compute_key(None, {"config": cuda}, {})
    changed_sampling_key = policy.compute_key(
        None,
        {
            "config": cpu.model_copy(
                update={
                    "sampling": sampling.model_copy(
                        update={"hard_bm25_per_positive": 0}
                    )
                }
            )
        },
        {},
    )

    assert cpu_key == cuda_key
    assert changed_sampling_key != cpu_key


def test_scientific_cache_key_uses_artifact_content_not_materialization_uri() -> None:
    payload = ArtifactPayload(
        role="tasks",
        relative_path="tasks.json",
        kind="file",
        digest="1" * 64,
        size_bytes=10,
        file_count=1,
    )
    first = DatasetArtifactRef(
        uri="/processed/first",
        kind=ArtifactKind.DATASET,
        digest="2" * 64,
        manifest_uri="/processed/first/manifest.json",
        payloads=(payload,),
        origin={"run": "first"},
        size_bytes=10,
        file_count=1,
    )
    second = first.model_copy(
        update={
            "uri": "/processed/second",
            "manifest_uri": "/processed/second/manifest.json",
            "origin": {"run": "second"},
        }
    )
    policy = ScientificInputs()

    assert policy.compute_key(None, {"prepared": first}, {}) == policy.compute_key(
        None, {"prepared": second}, {}
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
