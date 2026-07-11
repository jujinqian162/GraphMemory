from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from graph_memory.experiment.config import (
    ExperimentConfig,
    resolve_experiment_config,
    validate_composed_config,
)


REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _compose(
    *,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name=config_name,
            overrides=["name=test-run", *(overrides or [])],
        )
    return validate_composed_config(config)


def test_default_composition_is_closed_complete_and_resolvable() -> None:
    config = _compose()
    resolved = resolve_experiment_config(config, repository_root=REPO_ROOT)

    assert config.dataset.name == "hotpotqa"
    assert config.profile.name == "quick"
    assert [method.value for method in config.methods] == [
        "bm25",
        "dense",
        "bm25_graph_rerank",
        "dense_graph_rerank",
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    ]
    assert set(type(config.method_configs).model_fields) == {
        "bm25",
        "dense",
        "memory_stream",
        "bm25_graph_rerank",
        "dense_graph_rerank",
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    }
    assert resolved.dataset.prepare_script.is_absolute()
    assert resolved.dataset.splits["test"].source.is_absolute()
    assert resolved.tracking.database.is_absolute()
    assert resolved.tracking.tracking_uri.startswith("sqlite:///")
    assert "???" not in json.dumps(resolved.normalized())


def test_missing_name_unknown_override_and_unknown_model_key_fail_before_resolution() -> (
    None
):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        missing = compose(config_name="config")
        with pytest.raises(MissingMandatoryValue):
            OmegaConf.to_container(missing, resolve=True, throw_on_missing=True)
        with pytest.raises(ConfigCompositionException):
            compose(config_name="config", overrides=["name=test", "unknown=1"])

    primitive = _compose().model_dump(mode="python", by_alias=True)
    primitive["unknown"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(primitive)


@pytest.mark.parametrize("value", [True, "14", 14.0])
def test_scientific_integer_rejects_bool_string_and_float_coercion(
    value: object,
) -> None:
    primitive = _compose().model_dump(mode="python", by_alias=True)
    primitive["seed"] = value
    with pytest.raises(ValidationError, match="integer"):
        ExperimentConfig.model_validate(primitive)


def test_dataset_group_and_cloud_full_policy_resolve_complete_windows() -> None:
    hotpot = resolve_experiment_config(
        _compose(overrides=["profile=cloud-full"]),
        repository_root=REPO_ROOT,
    )
    twowiki = resolve_experiment_config(
        _compose(overrides=["dataset=2wiki", "profile=cloud-full"]),
        repository_root=REPO_ROOT,
    )
    musique = resolve_experiment_config(
        _compose(overrides=["dataset=musique", "profile=cloud-full"]),
        repository_root=REPO_ROOT,
    )

    assert {name: split.count for name, split in hotpot.dataset.splits.items()} == {
        "train": 90025,
        "dev": 500,
        "test": 6869,
    }
    assert {name: split.count for name, split in twowiki.dataset.splits.items()} == {
        "train": 167454,
        "dev": 500,
        "test": 12076,
    }
    assert {name: split.count for name, split in musique.dataset.splits.items()} == {
        "train": 19938,
        "dev": 500,
        "test": 1917,
    }
    assert twowiki.dataset.name == "twowiki"
    assert twowiki.dataset.prepare_script.name == "prepare_2wiki.py"


def test_fixed_count_beyond_dataset_capacity_fails_during_resolution() -> None:
    config = _compose(overrides=["profile=cloud-full"])
    primitive = config.model_dump(mode="python", by_alias=True)
    primitive["profile"]["splits"]["test"] = {"kind": "fixed", "count": 7000}
    invalid = ExperimentConfig.model_validate(primitive)
    with pytest.raises(ValueError, match=r"offset\+count=7500 beyond capacity=7369"):
        resolve_experiment_config(invalid, repository_root=REPO_ROOT)


def test_method_subset_seed_device_and_method_override_have_one_path() -> None:
    baseline = _compose()
    config = _compose(
        overrides=[
            "methods=[bm25,dense]",
            "seed=14",
            "device=cpu",
            "method_configs.dense_ft.train.trainer.learning_rate=3e-5",
        ]
    )

    assert [method.value for method in config.methods] == ["bm25", "dense"]
    assert config.method_configs.bm25.model_dump() == {"method": "bm25"}
    assert "device" not in config.method_configs.bm25.model_dump()
    assert config.method_configs.dense_ft.pairs.random_seed == 14
    assert config.method_configs.dense_ft.train.trainer.random_seed == 14
    assert config.method_configs.dense_ft.train.trainer.device == "cpu"
    assert config.method_configs.dense_ft.train.trainer.learning_rate == pytest.approx(
        3e-5
    )
    assert (
        config.method_configs.dense_rgcn_graph_retriever.train.trainer.learning_rate
        == baseline.method_configs.dense_rgcn_graph_retriever.train.trainer.learning_rate
    )


def test_ablation_values_and_explicit_2wiki_overrides_are_closed() -> None:
    config = _compose(
        overrides=[
            "dataset=2wiki",
            "profile=tiny",
            "methods=[bm25,dense,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft,dense_ft_rgcn_graph_retriever]",
        ]
    )
    assert config.dataset.name == "twowiki"
    assert config.profile.name == "tiny"
    assert [method.value for method in config.methods] == [
        "bm25",
        "dense",
        "dense_graph_rerank",
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    ]

    assert _compose(overrides=["ablation.variants=all"]).ablation.variants == "all"
    assert _compose(overrides=["ablation.variants=[wo_graph]"]).ablation.variants == [
        "wo_graph"
    ]
    with pytest.raises(ValidationError, match="baseline alias"):
        primitive = _compose().model_dump(mode="python", by_alias=True)
        primitive["ablation"]["variants"] = ["full_rgcn"]
        ExperimentConfig.model_validate(primitive)
