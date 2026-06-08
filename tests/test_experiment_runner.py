from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.workflow import (
    build_stage_plan,
    format_commands,
    initialize_experiment,
    inspect_experiment_status,
    load_experiment_config,
)
from graph_memory.training_config import load_trainable_training_config
import scripts.experiment as experiment_script


TRAINABLE_METHOD = "dense_rgcn_graph_retriever"


def _assert_repository_profile_resolution(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    profile: str,
) -> None:
    configured_profile = config["profiles"][profile]
    effective_config = manifest["effective_config"]
    for split in ("train", "dev", "test"):
        max_examples = effective_config["splits"][split]["max_examples"]
        assert max_examples == configured_profile[f"{split}_examples"]
        assert isinstance(max_examples, int)
        assert max_examples > 0

    training_config_path = config["training_configs"][TRAINABLE_METHOD]
    expected_training = load_trainable_training_config(training_config_path, profile=profile)
    training = effective_config["training"][TRAINABLE_METHOD]
    assert training == expected_training
    assert training["profile"] == profile
    assert training["optimization"]["batch_size"] > 0
    assert training["optimization"]["epochs"] > 0
    assert isinstance(training["optimization"]["device"], str)
    assert training["optimization"]["device"]


def _write_experiment_config(path: Path, raw_path: Path) -> None:
    payload = {
        "recipe": "hotpotqa_evidence_retrieval",
        "dataset": "hotpotqa",
        "task": "evidence_retrieval",
        "raw": {
            "train": str(raw_path),
            "dev": str(raw_path),
        },
        "profiles": {
            "quick": {
                "train_examples": 1,
                "dev_examples": 1,
                "test_examples": 1,
            },
            "full": {
                "train_examples": 5000,
                "dev_examples": 500,
                "test_examples": 1000,
            },
        },
        "defaults": {
            "seed": 13,
            "top_k": 10,
            "dense_encoder": "intfloat/e5-base-v2",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        },
        "graph": {
            "max_query_overlap": 20,
            "max_entity_neighbors": 10,
            "max_bridge_edges": 50,
            "use_spacy": False,
        },
        "search_spaces": {"graph_rerank": "configs/search_spaces/graph_rerank.json"},
        "split_offsets": {"train": 0, "dev": 0, "test": 0},
        "methods": ["bm25", "dense", "bm25_graph_rerank", "dense_graph_rerank"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trainable_experiment_config(path: Path, raw_path: Path, training_config_path: Path) -> None:
    _write_experiment_config(path, raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["methods"] = ["bm25", TRAINABLE_METHOD]
    payload["training_configs"] = {TRAINABLE_METHOD: str(training_config_path)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_rgcn_training_config(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "method": TRAINABLE_METHOD,
        "default_profile": "quick",
        "defaults": {
            "encoder": {
                "model": "intfloat/e5-base-v2",
                "query_prefix": "query: ",
                "passage_prefix": "passage: ",
            },
            "model": {
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.1,
                "ablation": "full_rgcn",
            },
            "optimization": {
                "optimizer": "AdamW",
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 0.0001,
                "max_grad_norm": 1.0,
                "random_seed": 13,
                "pos_weight": True,
                "device": "cpu",
            },
            "pair_sampling": {
                "random_seed": 13,
                "easy_random_per_positive": 2,
                "hard_bm25_per_positive": 2,
                "hard_dense_per_positive": 0,
                "hard_graph_neighbor_per_positive": 1,
                "hard_pool_size": 30,
            },
            "selection": {
                "best_metric": "dev_composite",
                "higher_is_better": True,
            },
            "reporting": {
                "render_training_curves": True,
            },
        },
        "profiles": {
            "smoke": {
                "model": {"hidden_dim": 32, "num_layers": 1},
                "optimization": {"epochs": 1, "batch_size": 1, "device": "cpu"},
                "pair_sampling": {
                    "easy_random_per_positive": 1,
                    "hard_bm25_per_positive": 1,
                    "hard_dense_per_positive": 0,
                    "hard_graph_neighbor_per_positive": 1,
                },
            },
            "quick": {
                "optimization": {"epochs": 5, "batch_size": 16, "device": "cpu"},
            },
            "full": {
                "model": {"hidden_dim": 256, "num_layers": 2},
                "optimization": {"epochs": 10, "batch_size": 32, "device": "cuda"},
                "pair_sampling": {"hard_dense_per_positive": 2},
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_initialize_experiment_merges_profile_and_cli_overrides(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)

    config = load_experiment_config(config_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        cli_overrides={"top_k": 5},
    )

    run_dir = tmp_path / "runs" / "quick_valid_100"
    effective_config = json.loads((run_dir / "config" / "effective_config.json").read_text(encoding="utf-8"))
    manifest_json = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert effective_config["top_k"] == 5
    assert effective_config["splits"]["train"]["max_examples"] == 1
    assert effective_config["splits"]["dev"]["max_examples"] == 1
    assert effective_config["splits"]["test"]["max_examples"] == 1
    assert manifest["profile"] == "quick"
    assert manifest_json["effective_config"]["top_k"] == 5


def test_manifest_writes_resolved_typed_stage_config_projections(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)

    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )
    run_dir = tmp_path / "runs" / "quick_valid_100"
    manifest_json = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["stage_configs"]["retrieve"]["bm25"] == {
        "io": {
            "tasks": (run_dir / "inputs" / "test.input.json").as_posix(),
            "graphs": None,
            "output": (run_dir / "predictions" / "test.bm25.ranked.json").as_posix(),
            "summary": (run_dir / "predictions" / "test.bm25.ranked.run_summary.json").as_posix(),
            "graph_config": None,
            "encoder_model": "intfloat/e5-base-v2",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        },
        "job": {"method": "bm25", "top_k": 10},
    }
    assert manifest["stage_configs"]["evaluate"]["bm25"]["io"] == {
        "predictions": (run_dir / "predictions" / "test.bm25.ranked.json").as_posix(),
        "labels": (run_dir / "inputs" / "test.labels.json").as_posix(),
        "graphs": (run_dir / "graphs" / "test.graphs.json").as_posix(),
        "output": (run_dir / "metrics" / "test.bm25.metrics.csv").as_posix(),
        "failure_cases_output": (run_dir / "debug" / "failure_cases_bm25.jsonl").as_posix(),
    }
    assert manifest_json["stage_configs"] == manifest["stage_configs"]


def test_stage_plan_prefers_stage_config_projection_but_keeps_old_manifest_fallback(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )
    projected_prediction = tmp_path / "projected" / "projection-wins.ranked.json"
    manifest["stage_configs"]["retrieve"]["bm25"]["io"]["output"] = projected_prediction.as_posix()
    manifest["stage_configs"]["evaluate"]["bm25"]["io"]["predictions"] = projected_prediction.as_posix()
    manifest["artifacts"]["predictions"]["bm25"] = (tmp_path / "legacy" / "legacy-loses.ranked.json").as_posix()

    projected_commands = build_stage_plan(manifest, stages=["retrieve", "evaluate"], methods=["bm25"])
    rendered_projected = [" ".join(command.argv) for command in projected_commands]

    assert any(projected_prediction.as_posix() in command for command in rendered_projected)
    assert not any("legacy-loses.ranked.json" in command for command in rendered_projected)

    legacy_manifest = dict(manifest)
    legacy_manifest.pop("stage_configs")
    legacy_commands = build_stage_plan(legacy_manifest, stages=["retrieve", "evaluate"], methods=["bm25"])
    rendered_legacy = [" ".join(command.argv) for command in legacy_commands]

    assert any("legacy-loses.ranked.json" in command for command in rendered_legacy)


def test_training_config_resolves_profile_over_defaults(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = tmp_path / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)

    manifest = initialize_experiment(
        "quick_rgcn",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    resolved = manifest["effective_config"]["training"][TRAINABLE_METHOD]
    learned = manifest["artifacts"]["learned"][TRAINABLE_METHOD]
    effective_training_config_path = tmp_path / "runs" / "quick_rgcn" / "learned" / TRAINABLE_METHOD / "effective_training_config.json"

    assert resolved["profile"] == "quick"
    assert resolved["optimization"]["batch_size"] == 16
    assert resolved["optimization"]["epochs"] == 5
    assert resolved["model"]["hidden_dim"] == 128
    assert Path(learned["train_pairs"]) == tmp_path / "runs" / "quick_rgcn" / "learned" / TRAINABLE_METHOD / "train.pairs.json"
    assert Path(learned["best_checkpoint"]) == tmp_path / "runs" / "quick_rgcn" / "learned" / TRAINABLE_METHOD / "checkpoints" / "best.pt"
    assert Path(learned["effective_training_config"]) == effective_training_config_path
    assert json.loads(effective_training_config_path.read_text(encoding="utf-8")) == resolved


def test_method_first_default_workflow_selects_required_stages(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = tmp_path / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)
    config = load_experiment_config(config_path)

    bm25_manifest = initialize_experiment(
        "quick_bm25",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )
    graph_manifest = initialize_experiment(
        "quick_graph",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["dense_graph_rerank"],
    )
    rgcn_manifest = initialize_experiment(
        "quick_rgcn",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    assert bm25_manifest["selected_stages"] == ["prepare", "graphs", "retrieve", "evaluate", "aggregate"]
    assert graph_manifest["selected_stages"] == ["prepare", "graphs", "tune", "retrieve", "evaluate", "aggregate"]
    assert rgcn_manifest["selected_stages"] == ["prepare", "graphs", "pairs", "train", "retrieve", "evaluate", "aggregate"]


def test_stage_range_is_selected_over_method_workflow(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_bm25",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )

    commands = build_stage_plan(
        manifest,
        methods=["bm25"],
        from_stage="prepare",
        to_stage="retrieve",
    )

    assert [command.stage for command in commands] == [
        "prepare",
        "prepare",
        "prepare",
        "graphs",
        "graphs",
        "graphs",
        "retrieve",
    ]

    with pytest.raises(ValueError, match="available workflow stages"):
        build_stage_plan(manifest, methods=["bm25"], from_stage="tune")


def test_trainable_stage_range_includes_supervision_and_training(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = tmp_path / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)
    manifest = initialize_experiment(
        "quick_rgcn",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    commands = build_stage_plan(
        manifest,
        methods=[TRAINABLE_METHOD],
        from_stage="prepare",
        to_stage="retrieve",
    )

    assert [command.stage for command in commands] == [
        "prepare",
        "prepare",
        "prepare",
        "graphs",
        "graphs",
        "graphs",
        "pairs",
        "train",
        "retrieve",
    ]
    assert all("dense_graph_rerank.dev_selected.json" not in " ".join(command.argv) for command in commands)


def test_experiment_config_name_and_training_config_name_resolve(tmp_path):
    config = load_experiment_config("hotpotqa_evidence_retrieval")
    assert config["recipe"] == "hotpotqa_evidence_retrieval"

    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_trainable_experiment_config(
        config_path,
        raw_path,
        Path("base"),
    )

    manifest = initialize_experiment(
        "quick_rgcn_named_config",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    assert manifest["effective_config"]["training"][TRAINABLE_METHOD]["method"] == TRAINABLE_METHOD


def test_initialize_experiment_generates_deterministic_run_paths(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)

    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25", "dense_graph_rerank"],
    )

    run_dir = tmp_path / "runs" / "quick_valid_100"
    assert Path(manifest["paths"]["run_dir"]) == run_dir
    assert Path(manifest["artifacts"]["inputs"]["test"]["input"]) == run_dir / "inputs" / "test.input.json"
    assert Path(manifest["artifacts"]["graphs"]["test"]) == run_dir / "graphs" / "test.graphs.json"
    assert (
        Path(manifest["artifacts"]["tuned"]["dense_graph_rerank"])
        == run_dir / "tuned" / "dense_graph_rerank.dev_selected.json"
    )
    assert (
        Path(manifest["artifacts"]["predictions"]["dense_graph_rerank"])
        == run_dir / "predictions" / "test.dense_graph_rerank.ranked.json"
    )
    assert Path(manifest["artifacts"]["tables"]["main"]) == run_dir / "tables" / "main_results.csv"


def test_experiment_plan_uses_config_not_training_cli_sprawl(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = tmp_path / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)
    manifest = initialize_experiment(
        "quick_rgcn",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    commands = build_stage_plan(
        manifest,
        stages=["pairs", "train", "retrieve"],
        methods=[TRAINABLE_METHOD],
    )
    rendered_by_stage = {command.stage.value: " ".join(command.argv) for command in commands}

    assert "scripts/build_train_pairs.py" in rendered_by_stage["pairs"]
    assert "--config" in rendered_by_stage["pairs"]
    assert "effective_training_config.json" in rendered_by_stage["pairs"]
    assert "--easy_random_per_positive" not in rendered_by_stage["pairs"]
    assert "scripts/train_graph_retriever.py" in rendered_by_stage["train"]
    assert "--config" in rendered_by_stage["train"]
    assert "effective_training_config.json" in rendered_by_stage["train"]
    for training_flag in ("--batch_size", "--epochs", "--hidden_dim", "--learning_rate", "--device"):
        assert training_flag not in rendered_by_stage["train"]
    assert "--checkpoint" in rendered_by_stage["retrieve"]
    assert "checkpoints/best.pt" in rendered_by_stage["retrieve"]
    assert "--device cpu" in rendered_by_stage["retrieve"]


def test_plan_generates_low_level_commands_without_outputs(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )

    commands = build_stage_plan(manifest, stages=["prepare", "graphs", "retrieve"], methods=["bm25"])
    rendered = [" ".join(command.argv) for command in commands]

    assert any("scripts/prepare_hotpotqa.py" in command for command in rendered)
    assert any("--output_input" in command and "inputs/test.input.json" in command for command in rendered)
    assert any("scripts/build_graphs.py" in command for command in rendered)
    assert any("scripts/run_retrieval.py" in command and "--method bm25" in command for command in rendered)
    assert not Path(manifest["artifacts"]["inputs"]["test"]["input"]).exists()
    assert not Path(manifest["artifacts"]["predictions"]["bm25"]).exists()


def test_format_commands_renders_readable_blocks_with_color(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )
    commands = build_stage_plan(manifest, stages=["retrieve", "evaluate"], methods=["bm25"])

    rendered = format_commands(commands, color=True)

    assert "\n\n" in rendered
    assert "[1] retrieve method=bm25" in rendered
    assert "script: scripts/run_retrieval.py" in rendered
    assert "  \x1b[36m--method\x1b[0m bm25" in rendered
    assert "\n  \x1b[36m--tasks\x1b[0m " in rendered
    assert "[2] evaluate method=bm25" in rendered


def test_plan_filters_methods_and_rejects_unknown_methods(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
    )
    tuned_path = Path(manifest["artifacts"]["tuned"]["dense_graph_rerank"])
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_path.write_text("{}", encoding="utf-8")

    commands = build_stage_plan(manifest, stages=["retrieve", "evaluate"], methods=["dense_graph_rerank"])
    rendered = [" ".join(command.argv) for command in commands]

    assert any("--method dense_graph_rerank" in command for command in rendered)
    assert any("test.dense_graph_rerank.ranked.json" in command for command in rendered)
    assert not any("--method bm25 " in f"{command} " for command in rendered)

    with pytest.raises(ValueError, match="Unsupported method"):
        build_stage_plan(manifest, stages=["retrieve"], methods=["not_a_method"])


def test_tune_plan_uses_graph_rerank_search_space_config(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25_graph_rerank"],
    )

    commands = build_stage_plan(manifest, stages=["tune"], methods=["bm25_graph_rerank"])
    rendered = [" ".join(command.argv) for command in commands]

    assert any("--grid_config configs/search_spaces/graph_rerank.json" in command for command in rendered)


def test_retrieve_graph_rerank_requires_tune_stage_or_existing_tuned_config(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["dense_graph_rerank"],
    )

    with pytest.raises(ValueError, match="requires tuned graph config"):
        build_stage_plan(manifest, stages=["retrieve"], methods=["dense_graph_rerank"])

    commands_with_tune = build_stage_plan(
        manifest,
        stages=["tune", "retrieve"],
        methods=["dense_graph_rerank"],
    )
    assert [command.stage for command in commands_with_tune] == ["tune", "retrieve"]

    tuned_path = Path(manifest["artifacts"]["tuned"]["dense_graph_rerank"])
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_path.write_text("{}", encoding="utf-8")

    commands_after_tune = build_stage_plan(manifest, stages=["retrieve"], methods=["dense_graph_rerank"])
    assert [command.stage for command in commands_after_tune] == ["retrieve"]


def test_trainable_retrieve_requires_checkpoint_or_train_stage(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = tmp_path / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)
    manifest = initialize_experiment(
        "quick_rgcn",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    with pytest.raises(ValueError, match="requires a trained checkpoint"):
        build_stage_plan(manifest, stages=["retrieve"], methods=[TRAINABLE_METHOD])

    commands_with_train = build_stage_plan(
        manifest,
        stages=["pairs", "train", "retrieve"],
        methods=[TRAINABLE_METHOD],
    )
    assert [command.stage for command in commands_with_train] == ["pairs", "train", "retrieve"]

    checkpoint_path = Path(manifest["artifacts"]["learned"][TRAINABLE_METHOD]["best_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"placeholder")

    commands_after_train = build_stage_plan(manifest, stages=["retrieve"], methods=[TRAINABLE_METHOD])
    assert [command.stage for command in commands_after_train] == ["retrieve"]


def test_repository_experiment_configs_have_clear_roles():
    assert Path("configs/experiments/hotpotqa_evidence_retrieval.json").exists()
    assert Path("configs/training/dense_rgcn_graph_retriever/base.json").exists()
    assert Path("configs/search_spaces/graph_rerank.json").exists()
    assert Path("configs/published/README.md").exists()


def test_repository_cloud_profiles_resolve_experiment_and_training_configs(tmp_path):
    config = load_experiment_config("configs/experiments/hotpoqa_dev_full.json")

    quick_manifest = initialize_experiment(
        "cloud_quick",
        config=config,
        run_root=tmp_path / "runs",
        profile="cloud-quick",
        methods=[TRAINABLE_METHOD],
    )
    full_manifest = initialize_experiment(
        "cloud_full",
        config=config,
        run_root=tmp_path / "runs",
        profile="cloud-full",
        methods=[TRAINABLE_METHOD],
    )

    _assert_repository_profile_resolution(config, quick_manifest, profile="cloud-quick")
    _assert_repository_profile_resolution(config, full_manifest, profile="cloud-full")


def test_status_reports_missing_complete_and_stale_outputs(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    manifest = initialize_experiment(
        "quick_valid_100",
        config=load_experiment_config(config_path),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )

    missing_status = inspect_experiment_status(manifest)
    assert _state_for(missing_status, "retrieve", "bm25") == "missing"

    prediction_path = Path(manifest["artifacts"]["predictions"]["bm25"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("[]\n", encoding="utf-8")
    summary_path = prediction_path.with_name(f"{prediction_path.stem}.run_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "effective_config": {"method": "bm25", "top_k": manifest["effective_config"]["top_k"]},
                "inputs": {"tasks": manifest["artifacts"]["inputs"]["test"]["input"]},
                "outputs": {"predictions": str(prediction_path)},
            }
        ),
        encoding="utf-8",
    )

    complete_status = inspect_experiment_status(manifest)
    assert _state_for(complete_status, "retrieve", "bm25") == "complete"

    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "effective_config": {"method": "dense", "top_k": manifest["effective_config"]["top_k"]},
                "inputs": {"tasks": manifest["artifacts"]["inputs"]["test"]["input"]},
                "outputs": {"predictions": str(prediction_path)},
            }
        ),
        encoding="utf-8",
    )

    stale_status = inspect_experiment_status(manifest)
    assert _state_for(stale_status, "retrieve", "bm25") == "stale"


def test_experiment_cli_init_and_plan(tmp_path, capsys):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)

    assert experiment_script.main(
        [
            "init",
            "quick_valid_100",
            "--config",
            str(config_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--profile",
            "quick",
            "--methods",
            "bm25",
        ]
    ) == 0
    assert (tmp_path / "runs" / "quick_valid_100" / "manifest.json").exists()

    assert experiment_script.main(
        [
            "plan",
            "quick_valid_100",
            "--run-root",
            str(tmp_path / "runs"),
            "--stages",
            "retrieve,evaluate",
            "--methods",
            "bm25",
        ]
    ) == 0
    output = capsys.readouterr().out

    assert "scripts/run_retrieval.py" in output
    assert "scripts/evaluate_retrieval.py" in output


def test_experiment_cli_accepts_method_range_and_lists_resources(tmp_path, capsys):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)

    assert experiment_script.main(
        [
            "init",
            "quick_bm25",
            "--config",
            str(config_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--profile",
            "quick",
            "--method",
            "bm25",
        ]
    ) == 0
    assert experiment_script.main(
        [
            "plan",
            "quick_bm25",
            "--run-root",
            str(tmp_path / "runs"),
            "--method",
            "bm25",
            "--from",
            "prepare",
            "--to",
            "retrieve",
            "--color",
            "never",
        ]
    ) == 0
    plan_output = capsys.readouterr().out

    assert "[1] prepare split=train" in plan_output
    assert "script: scripts/prepare_hotpotqa.py" in plan_output
    assert "script: scripts/run_retrieval.py" in plan_output
    assert "scripts/evaluate_retrieval.py" not in plan_output

    assert experiment_script.main(["methods", "list"]) == 0
    methods_output = capsys.readouterr().out
    assert "bm25" in methods_output
    assert "dense_rgcn_graph_retriever" in methods_output
    assert "prepare, graphs, pairs, train, retrieve, evaluate, aggregate" in methods_output

    assert experiment_script.main(["configs", "list"]) == 0
    configs_output = capsys.readouterr().out
    assert "hotpotqa_evidence_retrieval" in configs_output
    assert "dense_rgcn_graph_retriever/base" in configs_output

    assert experiment_script.main(["profile", "list", "--config", "hotpotqa_evidence_retrieval"]) == 0
    profiles_output = capsys.readouterr().out
    repository_config = load_experiment_config("hotpotqa_evidence_retrieval")
    quick_profile = repository_config["profiles"]["quick"]
    split_sources = repository_config["split_sources"]
    split_offsets = repository_config["split_offsets"]
    defaults = repository_config["defaults"]
    assert "quick" in profiles_output
    assert f"train={quick_profile['train_examples']}" in profiles_output
    assert f"train[source={split_sources['train']}" in profiles_output
    assert f"dev[source={split_sources['dev']}" in profiles_output
    assert f"test[source={split_sources['test']}" in profiles_output
    assert f"offset={split_offsets['test']}" in profiles_output
    assert f"seed={defaults['seed']}" in profiles_output

    assert experiment_script.main(["profiles", "list", "--config", "hotpotqa_evidence_retrieval"]) == 0
    plural_profiles_output = capsys.readouterr().out
    assert plural_profiles_output == profiles_output


def test_initialize_rejects_config_change_for_existing_manifest(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)
    config = load_experiment_config(config_path)
    initialize_experiment(
        "quick_valid_100",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25"],
    )

    with pytest.raises(ValueError, match="different config"):
        initialize_experiment(
            "quick_valid_100",
            config=config,
            run_root=tmp_path / "runs",
            profile="quick",
            methods=["dense"],
        )


def test_experiment_cli_runs_bm25_smoke_pipeline(tmp_path):
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_experiment_config(config_path, raw_path)

    assert experiment_script.main(
        [
            "run",
            "smoke_bm25",
            "--config",
            str(config_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--profile",
            "quick",
            "--methods",
            "bm25",
            "--stages",
            "prepare,graphs,retrieve,evaluate,aggregate",
        ]
    ) == 0

    run_dir = tmp_path / "runs" / "smoke_bm25"
    assert (run_dir / "inputs" / "test.input.json").exists()
    assert (run_dir / "graphs" / "test.graphs.json").exists()
    assert (run_dir / "predictions" / "test.bm25.ranked.json").exists()
    assert (run_dir / "metrics" / "test.bm25.metrics.csv").exists()
    assert (run_dir / "tables" / "main_results.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["retrieve:bm25"]["state"] == "complete"


def _state_for(status_rows: list[dict[str, str]], stage: str, method: str | None = None) -> str:
    for row in status_rows:
        if row["stage"] == stage and row.get("method") == method:
            return row["state"]
    raise AssertionError(f"missing status row for stage={stage} method={method}")
