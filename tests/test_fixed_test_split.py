from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.workflow import _prepare_config

ROOT = Path(__file__).resolve().parents[1]


def _resolved(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=fixed-test-split",
                "profile=smoke",
                "device=cpu",
                "method=bm25",
                *overrides,
            ],
        )
    return resolve_experiment_config(
        parse_composed_config(composed), repository_root=ROOT
    )


def test_test_split_uses_split_seed_not_training_seed() -> None:
    config = _resolved("seed=41", "split_seed=13")
    test_prepare = _prepare_config(config, "test")
    assert test_prepare.seed == 13


def test_test_split_is_stable_across_training_seeds() -> None:
    config_a = _resolved("seed=13", "split_seed=13")
    config_b = _resolved("seed=41", "split_seed=13")
    assert (
        _prepare_config(config_a, "test").seed
        == _prepare_config(config_b, "test").seed
        == 13
    )


def test_train_and_dev_keep_using_training_seed() -> None:
    config = _resolved("seed=41", "split_seed=13")
    assert _prepare_config(config, "train").seed == 41
    assert _prepare_config(config, "dev").seed == 41


def test_isetrace_trajectory_selection_uses_split_seed() -> None:
    config_a = _resolved("dataset=isetrace", "seed=13", "split_seed=41")
    config_b = _resolved("dataset=isetrace", "seed=29", "split_seed=41")

    for split in ("train", "dev", "test"):
        prepare_a = _prepare_config(config_a, split)
        prepare_b = _prepare_config(config_b, split)
        assert prepare_a.seed == prepare_b.seed == 41
        assert prepare_a.trajectory_splits == prepare_b.trajectory_splits
        assert prepare_a.trajectory_splits is not None
        if split == "test":
            assert prepare_a.trajectory_splits.test == 1207
