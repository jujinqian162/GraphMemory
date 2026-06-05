from __future__ import annotations

import json
from pathlib import Path

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.training_config import load_trainable_training_config
from scripts.workflow.manifest import resolve_training_config_path


TRAINABLE_METHOD = "dense_rgcn_graph_retriever"


def test_schema_v2_method_config_is_shallow_and_profiled() -> None:
    path = Path("configs/methods/dense_rgcn_graph_retriever.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    resolved = load_trainable_training_config(path, profile="smoke")

    assert payload["schema_version"] == 2
    assert "defaults" not in payload
    assert payload["method"] == TRAINABLE_METHOD
    assert resolved["schema_version"] == 2
    assert resolved["profile"] == "smoke"
    assert resolved["encoder"]["model"] == "models/intfloat-e5-base-v2"
    assert resolved["model"]["hidden_dim"] == 32
    assert resolved["optimization"]["epochs"] == 1
    assert resolved["optimization"]["batch_size"] == 1
    assert resolved["pair_sampling"]["easy_random_per_positive"] == 1


def test_existing_training_config_path_remains_readable() -> None:
    path = resolve_training_config_path(TRAINABLE_METHOD, "base")
    resolved = load_trainable_training_config(path, profile="smoke")

    assert path == Path("configs/training/dense_rgcn_graph_retriever/base.json")
    assert resolved["schema_version"] == 1
    assert resolved["profile"] == "smoke"
    assert resolved["optimization"]["epochs"] == 1


def test_training_config_path_resolves_explicit_schema_v2_method_file() -> None:
    path = resolve_training_config_path(TRAINABLE_METHOD, "configs/methods/dense_rgcn_graph_retriever.json")

    assert path == Path("configs/methods/dense_rgcn_graph_retriever.json")
    assert load_trainable_training_config(path, profile="quick")["schema_version"] == 2


def test_train_stage_config_loader_reads_schema_v2_method_file(tmp_path: Path) -> None:
    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_labels",
            str(tmp_path / "train.labels.json"),
            "--train_graphs",
            str(tmp_path / "train.graphs.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--dev_graphs",
            str(tmp_path / "dev.graphs.json"),
            "--output_dir",
            str(tmp_path / "rgcn_run"),
            "--config",
            "configs/methods/dense_rgcn_graph_retriever.json",
            "--epochs",
            "1",
        ],
    )

    assert config.job.trainer.batch_size == 8
    assert config.job.trainer.epochs == 1
    assert config.job.pairs.hard_dense_per_positive == 0
