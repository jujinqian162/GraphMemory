from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.workflow.contracts import validate_current_manifest
from scripts.workflow.manifest import initialize_experiment, load_experiment_config
from scripts.workflow.workflows import (
    build_evaluate_commands,
    build_pair_commands,
    build_retrieve_commands,
    build_train_commands,
)


RGCN = "dense_rgcn_graph_retriever"
DENSE_FT = "dense_ft"


def test_initialize_writes_current_manifest_and_complete_stage_configs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "current",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[RGCN, DENSE_FT],
        force=True,
    )

    assert "schema_version" not in manifest
    assert set(manifest["stage_configs"]) == {"importance", "pairs", "train", "retrieve", "evaluate"}
    for stage, methods in manifest["stage_configs"].items():
        expected_methods = set() if stage == "importance" else {RGCN, DENSE_FT}
        assert set(methods) == expected_methods
        for method, path in methods.items():
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            assert isinstance(payload, dict), (stage, method)

    commands = [
        build_pair_commands(manifest, [DENSE_FT])[0],
        build_train_commands(manifest, [RGCN])[0],
        build_retrieve_commands(manifest, [DENSE_FT])[0],
        build_evaluate_commands(manifest, [DENSE_FT])[0],
    ]
    assert all(command.argv[2] == "--config" for command in commands)
    assert all(len(command.argv) == 4 for command in commands)


def test_missing_stage_config_fails_before_command_creation(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "missing-stage",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT],
        force=True,
    )
    del manifest["stage_configs"]["train"][DENSE_FT]

    with pytest.raises(ValueError, match="stage_configs.train is missing selected methods"):
        validate_current_manifest(manifest)

    with pytest.raises(ValueError, match="requires stage config"):
        build_train_commands(manifest, [DENSE_FT])


def test_current_manifest_rejects_version_fields() -> None:
    manifest = {
        "schema_version": 1,
        "experiment_name": "legacy",
        "recipe": "legacy",
        "profile": "smoke",
        "created_at": "now",
        "updated_at": "now",
        "paths": {},
        "selected_methods": [],
        "selected_stages": [],
        "effective_config": {},
        "artifacts": {},
        "run_units": [],
        "stage_status": {},
        "stage_configs": {
            "pairs": {},
            "train": {},
            "retrieve": {},
            "evaluate": {},
        },
    }

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_current_manifest(manifest)


def test_force_rebuild_replaces_an_old_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "force-rebuild"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")

    manifest = initialize_experiment(
        "force-rebuild",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT],
        force=True,
    )

    assert "schema_version" not in manifest
    validate_current_manifest(manifest)


def test_force_rebuild_removes_existing_run_artifacts(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "force-clean",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT],
        force=True,
    )
    stale_model_dir = Path(manifest["artifacts"]["learned"][DENSE_FT]["best_checkpoint"])
    stale_model_dir.mkdir(parents=True)
    stale_marker = stale_model_dir / "stale-model.bin"
    stale_marker.write_bytes(b"stale")

    rebuilt = initialize_experiment(
        "force-clean",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="full",
        methods=[DENSE_FT],
        force=True,
    )

    assert not stale_marker.exists()
    assert rebuilt["profile"] == "full"
