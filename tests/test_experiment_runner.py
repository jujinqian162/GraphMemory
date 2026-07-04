from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory.io import read_json, write_json
from scripts.workflow import (
    build_stage_plan,
    format_commands,
    run_stage_plan,
    initialize_experiment,
    inspect_experiment_status,
    list_config_entries,
    load_experiment_config,
)
import scripts.experiment as experiment_script
from scripts.workflow.types import StageCommand, StageId

TRAINABLE_METHOD = "dense_rgcn_graph_retriever"
DENSE_FT_SEEDED_RGCN_METHOD = "dense_ft_rgcn_graph_retriever"
DENSE_FT_METHOD = "dense_ft"


def test_initialize_experiment_resolves_method_configs_and_stage_configs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "current",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[TRAINABLE_METHOD, DENSE_FT_METHOD],
        force=True,
    )

    assert "schema_version" not in manifest
    assert set(manifest["effective_config"]["resolved_method_configs"]) == {TRAINABLE_METHOD, DENSE_FT_METHOD}
    assert "training" not in manifest["effective_config"]
    assert "training_configs" not in manifest["effective_config"]

    for method in (TRAINABLE_METHOD, DENSE_FT_METHOD):
        learned = manifest["artifacts"]["learned"][method]
        assert Path(learned["effective_method_config"]).is_file()
        assert "effective_training_config" not in learned
        for stage in ("pairs", "train", "retrieve", "evaluate"):
            stage_path = Path(manifest["stage_configs"][stage][method])
            assert stage_path.is_file()
            assert isinstance(read_json(stage_path), dict)


def _config_with_seeded_rgcn_method_config(tmp_path: Path) -> dict[str, object]:
    config = load_experiment_config()
    method_config = read_json(Path("configs/methods/dense_rgcn_graph_retriever.json"))
    if not isinstance(method_config, dict):
        raise AssertionError("R-GCN method config fixture must be an object.")
    method_config["method"] = DENSE_FT_SEEDED_RGCN_METHOD
    method_config_path = tmp_path / "dense_ft_rgcn_graph_retriever.json"
    write_json(method_config_path, method_config)
    config["method_configs"] = {
        **config.get("method_configs", {}),
        DENSE_FT_SEEDED_RGCN_METHOD: method_config_path.as_posix(),
    }
    return config


def test_seeded_rgcn_plan_orders_dense_ft_dependency_before_rgcn(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "seeded-rgcn-plan",
        config=_config_with_seeded_rgcn_method_config(tmp_path),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT_SEEDED_RGCN_METHOD],
        force=True,
    )

    commands = build_stage_plan(
        manifest,
        from_stage="pairs",
        to_stage="evaluate",
        methods=[DENSE_FT_SEEDED_RGCN_METHOD],
    )

    assert [(command.stage, command.method) for command in commands] == [
        (StageId.PAIRS, DENSE_FT_METHOD),
        (StageId.PAIRS, DENSE_FT_SEEDED_RGCN_METHOD),
        (StageId.TRAIN, DENSE_FT_METHOD),
        (StageId.TRAIN, DENSE_FT_SEEDED_RGCN_METHOD),
        (StageId.RETRIEVE, DENSE_FT_SEEDED_RGCN_METHOD),
        (StageId.EVALUATE, DENSE_FT_SEEDED_RGCN_METHOD),
    ]
    rendered = format_commands(commands, color=False)
    assert "scripts/build_train_pairs.py" in rendered
    assert "scripts/train_method.py" in rendered
    assert "scripts/run_retrieval.py" in rendered
    assert "scripts/evaluate_retrieval.py" in rendered


def test_run_stage_plan_prints_plan_style_command_before_execution(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    command = StageCommand(
        stage=StageId.RETRIEVE,
        method="dense",
        argv=[
            "uv",
            "run",
            "python",
            "scripts/run_retrieval.py",
            "--config",
            "runs/example/retrieve.dense.json",
        ],
    )
    observed_argv: list[list[str]] = []
    observed_stdout: list[str] = []

    def fake_run(argv: list[str], *, check: bool) -> None:
        observed_argv.append(argv)
        observed_stdout.append(capsys.readouterr().out)
        assert check is True

    monkeypatch.setattr("scripts.workflow.planner.subprocess.run", fake_run)

    run_stage_plan([command], color=True)

    assert observed_argv == [command.argv]
    assert len(observed_stdout) == 1
    assert "[1] retrieve method=dense" in observed_stdout[0]
    assert "script: scripts/run_retrieval.py" in observed_stdout[0]
    assert "command:\n  uv\n  run\n  python\n  scripts/run_retrieval.py" in observed_stdout[0]
    assert "  \033[36m--config\033[0m runs/example/retrieve.dense.json" in observed_stdout[0]


def test_seeded_rgcn_from_train_requires_dense_ft_train_pairs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "seeded-rgcn-resume",
        config=_config_with_seeded_rgcn_method_config(tmp_path),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT_SEEDED_RGCN_METHOD],
        force=True,
    )
    seeded_pairs = Path(manifest["artifacts"]["learned"][DENSE_FT_SEEDED_RGCN_METHOD]["train_pairs"])
    seeded_pairs.parent.mkdir(parents=True, exist_ok=True)
    seeded_pairs.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="dense_ft.*train.pairs.json"):
        build_stage_plan(
            manifest,
            from_stage="train",
            to_stage="train",
            methods=[DENSE_FT_SEEDED_RGCN_METHOD],
        )


def test_stage_plan_uses_stage_config_commands_for_trainable_methods(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "plan",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT_METHOD],
        force=True,
    )

    commands = build_stage_plan(
        manifest,
        from_stage="pairs",
        to_stage="evaluate",
        methods=[DENSE_FT_METHOD],
    )

    assert [command.stage for command in commands] == [
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
    ]
    for command in commands:
        assert command.argv[2] == "--config"
        assert len(command.argv) == 4
    rendered = format_commands(commands, color=False)
    assert "--model_dir" not in rendered
    assert "--checkpoint" not in rendered


def test_trainable_retrieve_requires_checkpoint_or_train_stage(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "missing-checkpoint",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[TRAINABLE_METHOD],
        force=True,
    )

    with pytest.raises(ValueError, match="Trainable retrieval requires a trained checkpoint"):
        build_stage_plan(manifest, from_stage="retrieve", to_stage="retrieve", methods=[TRAINABLE_METHOD])

    Path(manifest["artifacts"]["learned"][TRAINABLE_METHOD]["best_checkpoint"]).parent.mkdir(parents=True)
    Path(manifest["artifacts"]["learned"][TRAINABLE_METHOD]["best_checkpoint"]).write_bytes(b"fake")

    commands = build_stage_plan(manifest, from_stage="retrieve", to_stage="retrieve", methods=[TRAINABLE_METHOD])
    assert [command.stage for command in commands] == [StageId.RETRIEVE]


def test_config_listing_uses_methods_not_training() -> None:
    kinds = {entry.kind for entry in list_config_entries("all")}

    assert "method" in kinds
    assert "training" not in kinds


def test_hotpotqa_canonical_config_owns_dev_profile_without_dev_full_duplicate(tmp_path: Path) -> None:
    config = load_experiment_config("hotpotqa_evidence_retrieval")
    experiment_names = {entry.name for entry in list_config_entries("experiments")}

    assert config["profiles"]["dev"] == {
        "dev_examples": 500,
        "test_examples": 6869,
        "train_examples": 1,
    }
    assert "hotpotqa_evidence_retrieval" in experiment_names
    assert "hotpoqa_dev_full" not in experiment_names
    assert not Path("configs/experiments/hotpoqa_dev_full.json").exists()

    manifest = initialize_experiment(
        "hotpotqa-dev-profile",
        config=config,
        run_root=tmp_path,
        profile="dev",
        methods=[
            TRAINABLE_METHOD,
            "learned_graph_rgcn_retriever",
            DENSE_FT_METHOD,
            DENSE_FT_SEEDED_RGCN_METHOD,
        ],
        force=True,
    )
    assert manifest["effective_config"]["splits"]["test"]["max_examples"] == 6869


def test_experiment_cli_init_plan_and_list_current_resources(tmp_path: Path, capsys) -> None:
    assert experiment_script.main(
        [
            "init",
            "cli-current",
            "--run-root",
            str(tmp_path),
            "--profile",
            "smoke",
            "--methods",
            DENSE_FT_METHOD,
            "--force",
        ]
    ) == 0
    init_out = capsys.readouterr().out
    assert "manifest.json" in init_out

    assert experiment_script.main(
        [
            "plan",
            "cli-current",
            "--run-root",
            str(tmp_path),
            "--from",
            "pairs",
            "--to",
            "train",
            "--methods",
            DENSE_FT_METHOD,
            "--no-cache",
        ]
    ) == 0
    plan_out = capsys.readouterr().out
    assert "scripts/build_train_pairs.py" in plan_out
    assert "scripts/train_method.py" in plan_out
    assert "--config" in plan_out

    assert experiment_script.main(["configs", "list", "--kind", "methods"]) == 0
    configs_out = capsys.readouterr().out
    assert "dense_ft" in configs_out


def test_status_uses_current_method_config_and_train_artifact_shape(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "status",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT_METHOD],
        force=True,
    )
    rows = inspect_experiment_status(manifest)
    train = next(row for row in rows if row["stage"] == "train" and row.get("method") == DENSE_FT_METHOD)

    assert train["state"] == "missing"
    assert train["path"].endswith("checkpoints/best_model")
    assert Path(train["path"]).suffix == ""


def test_retrieval_output_without_run_summary_is_stale(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "retrieval-status",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=["bm25"],
        force=True,
    )
    prediction_path = Path(manifest["artifacts"]["predictions"]["bm25"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("[]", encoding="utf-8")

    rows = inspect_experiment_status(manifest)
    retrieval = next(row for row in rows if row["stage"] == "retrieve" and row.get("method") == "bm25")

    assert retrieval["state"] == "stale"


def test_experiment_config_rejects_retired_training_configs_key(tmp_path: Path) -> None:
    config = load_experiment_config()
    config["training_configs"] = {"dense_ft": "configs/training/dense_ft/base.json"}
    path = tmp_path / "legacy-experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="training_configs"):
        load_experiment_config(path)

    with pytest.raises(ValueError, match="training_configs"):
        initialize_experiment(
            "legacy-experiment",
            config=config,
            run_root=tmp_path,
            profile="smoke",
            methods=[DENSE_FT_METHOD],
            force=True,
        )
