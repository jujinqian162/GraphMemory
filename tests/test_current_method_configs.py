from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.registry.method_configs import (
    DenseFinetuneMethodConfig,
    RgcnMethodConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_rgcn_method_config_loads_as_current_typed_config() -> None:
    config = CONFIG_LOADER.load(
        Registry.configs.TRAINABLE_METHOD,
        [
            "--config",
            str(REPO_ROOT / "configs" / "methods" / "dense_rgcn_graph_retriever.json"),
            "--profile",
            "smoke",
        ],
    )

    assert isinstance(config, RgcnMethodConfig)
    assert config.encoder.model_name == "models/intfloat-e5-base-v2"
    assert config.pairs.easy_random_per_positive == 1
    assert config.train.model.hidden_dim == 32
    assert config.train.trainer.batch_size == 1
    assert config.train.trainer.epochs == 1


def test_repository_dense_ft_method_config_loads_as_current_typed_config() -> None:
    config = CONFIG_LOADER.load(
        Registry.configs.TRAINABLE_METHOD,
        [
            "--config",
            str(REPO_ROOT / "configs" / "methods" / "dense_ft.json"),
            "--profile",
            "smoke",
        ],
    )

    assert isinstance(config, DenseFinetuneMethodConfig)
    assert config.encoder.model_name == "models/intfloat-e5-base-v2"
    assert config.pairs.hard_bm25_per_positive == 2
    assert config.train.trainer.train_batch_size == 1
    assert config.train.trainer.eval_batch_size == 4
    assert config.train.trainer.device == "cpu"


@pytest.mark.parametrize(
    "retired_field,retired_value",
    [
        ("schema_version", 2),
        ("defaults", {}),
        ("pair_sampling", {}),
        ("optimization", {}),
        ("model", {}),
    ],
)
def test_method_config_rejects_retired_top_level_fields(
    tmp_path: Path,
    retired_field: str,
    retired_value: object,
) -> None:
    payload = {
        "method": "dense_ft",
        "default_profile": "quick",
        "encoder": {
            "model_name": "models/e5",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "batch_size": 8,
        },
        "pairs": {
            "random_seed": 13,
            "easy_random_per_positive": 1,
            "hard_bm25_per_positive": 1,
            "hard_dense_per_positive": 0,
            "hard_graph_neighbor_per_positive": 1,
            "hard_pool_size": 10,
        },
        "train": {
            "data": {"hard_negatives_per_positive": 1},
            "trainer": {
                "learning_rate": 0.00002,
                "train_batch_size": 1,
                "eval_batch_size": 4,
                "epochs": 1,
                "warmup_steps": 0,
                "max_grad_norm": 1.0,
                "random_seed": 13,
                "device": "cpu",
                "use_amp": False,
            },
            "selection": {
                "best_metric": "eval_dev_cos_sim_map@100",
                "higher_is_better": True,
            },
        },
        "profiles": {"quick": {}},
        retired_field: retired_value,
    }
    config_path = tmp_path / "retired.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fields"):
        CONFIG_LOADER.load(
            Registry.configs.TRAINABLE_METHOD,
            ["--config", str(config_path), "--profile", "quick"],
        )


def test_method_config_rejects_legacy_encoder_model_alias(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "method": "dense_ft",
        "default_profile": "quick",
        "defaults": {
            "encoder": {
                "model": "models/e5",
                "query_prefix": "query: ",
                "passage_prefix": "passage: ",
                "batch_size": 8,
            }
        },
        "profiles": {"quick": {}},
    }
    config_path = tmp_path / "legacy.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        CONFIG_LOADER.load(
            Registry.configs.TRAINABLE_METHOD,
            ["--config", str(config_path), "--profile", "quick"],
        )


def test_method_config_rejects_missing_fields_even_when_dataclass_has_defaults(tmp_path: Path) -> None:
    payload = json.loads((REPO_ROOT / "configs" / "methods" / "dense_ft.json").read_text(encoding="utf-8"))
    del payload["train"]["trainer"]["learning_rate"]
    config_path = tmp_path / "missing-field.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields.*learning_rate"):
        CONFIG_LOADER.load(
            Registry.configs.TRAINABLE_METHOD,
            ["--config", str(config_path), "--profile", "quick"],
        )


def test_method_config_rejects_unknown_fields_in_unselected_profiles(tmp_path: Path) -> None:
    payload = json.loads((REPO_ROOT / "configs" / "methods" / "dense_ft.json").read_text(encoding="utf-8"))
    payload["profiles"]["full"]["train"]["trainer"]["legacy_learning_rate"] = 0.1
    config_path = tmp_path / "unknown-unselected-profile-field.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fields.*legacy_learning_rate"):
        CONFIG_LOADER.load(
            Registry.configs.TRAINABLE_METHOD,
            ["--config", str(config_path), "--profile", "quick"],
        )
