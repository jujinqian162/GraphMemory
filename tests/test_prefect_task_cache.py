from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

import graph_memory.experiment.workflow as experiment_workflow
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPayload,
    DatasetArtifactRef,
)
from graph_memory.experiment.cache import ScientificInputs
from graph_memory.experiment.config import (
    DenseEncoderConfig,
    PairBuildConfig,
    NegativeSamplingConfig,
    parse_composed_config,
    resolve_experiment_config,
)


ROOT = Path(__file__).resolve().parents[1]


class PairInputsCaptured(Exception):
    pass


@pytest.mark.parametrize(
    ("dataset", "variant", "expects_graph", "expected_graph_neighbors"),
    (
        ("hotpotqa", "flat", True, 1),
        ("isetrace", "flat", False, 0),
        ("isetrace", "provenance_unit", False, 0),
    ),
)
def test_dense_ft_flow_uses_family_compatible_pair_inputs(
    monkeypatch,
    tmp_path: Path,
    dataset: str,
    variant: str,
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
                f"method.variant={variant}",
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
        lambda *, source, config, trajectory_source=None, authoring_metadata_source=None: (
            object()
        ),
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
    assert pair_config.candidate_view == variant
    assert observed.get("built_graph", False) is expects_graph
    assert observed["evidence_graphs"] is (graph_artifact if expects_graph else None)
    assert (
        pair_config.sampling.hard_graph_neighbor_per_positive
        == expected_graph_neighbors
    )


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
    provenance_view_key = policy.compute_key(
        None,
        {"config": cpu.model_copy(update={"candidate_view": "provenance_unit"})},
        {},
    )

    assert cpu_key == cuda_key
    assert changed_sampling_key != cpu_key
    assert provenance_view_key != cpu_key


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
