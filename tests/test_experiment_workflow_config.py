from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    DenseFinetuneMethodConfig,
    ProvenanceRgcnMethodConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.workflow import _prepare_config


ROOT = Path(__file__).resolve().parents[1]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=["name=config-test", "profile=smoke", "device=cpu", *overrides],
        )


def test_isetrace_smoke_profile_caps_each_split_to_one_task() -> None:
    resolved = resolve_experiment_config(
        parse_composed_config(
            _compose("dataset=isetrace", "profile=smoke", "method=provenance_path")
        ),
        repository_root=ROOT,
    )

    assert {split.count for split in resolved.dataset.splits.values()} == {1}
    assert _prepare_config(resolved, "test").count == 1


def test_isetrace_training_size_override_reaches_prepare_identity() -> None:
    resolved = resolve_experiment_config(
        parse_composed_config(
            _compose(
                "dataset=isetrace",
                "profile=full",
                "method=provenance_unit_dense_ft_rgcn",
                "dataset.trajectories.splits.train=241",
            )
        ),
        repository_root=ROOT,
    )

    assert resolved.dataset.trajectories is not None
    assert resolved.dataset.trajectories.splits.model_dump() == {
        "train": 241,
        "dev": 352,
        "test": 1207,
    }
    prepare_train = _prepare_config(resolved, "train")
    assert prepare_train.trajectory_splits is not None
    assert prepare_train.trajectory_splits.model_dump() == {
        "train": 241,
        "dev": 352,
        "test": 1207,
    }


def test_isetrace_dense_ft_uses_effective_text_only_sampling() -> None:
    composed = parse_composed_config(
        _compose("dataset=isetrace", "method=dense_ft", "profile=full")
    )
    resolved = resolve_experiment_config(composed, repository_root=ROOT)

    assert isinstance(composed.method, DenseFinetuneMethodConfig)
    assert composed.method.pairs.hard_graph_neighbor_per_positive == 1
    assert isinstance(resolved.method, DenseFinetuneMethodConfig)
    assert resolved.method.pairs.hard_graph_neighbor_per_positive == 0

    evidence = resolve_experiment_config(
        parse_composed_config(
            _compose("dataset=hotpotqa", "method=dense_ft", "profile=full")
        ),
        repository_root=ROOT,
    )
    assert isinstance(evidence.method, DenseFinetuneMethodConfig)
    assert evidence.method.pairs.hard_graph_neighbor_per_positive == 1


def test_provenance_variant_is_the_only_ablation_authority() -> None:
    composed = parse_composed_config(
        _compose(
            "dataset=isetrace",
            "method=provenance_unit_dense_ft_rgcn",
            "method.variant=full_rgcn",
            "method.train.model.ablation=random_edges",
        )
    )
    resolved = resolve_experiment_config(composed, repository_root=ROOT)

    assert isinstance(resolved.method, ProvenanceRgcnMethodConfig)
    assert resolved.method.variant == "full_rgcn"
    assert resolved.method.train.model.ablation == "full_rgcn"


def test_evidence_dataset_rejects_provenance_rgcn_method() -> None:
    composed = parse_composed_config(
        _compose("dataset=hotpotqa", "method=provenance_rgcn")
    )

    with pytest.raises(ValueError, match="does not support"):
        resolve_experiment_config(composed, repository_root=ROOT)
