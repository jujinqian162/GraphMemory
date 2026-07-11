from __future__ import annotations

from pathlib import Path

from tests.refactor_7_9_call_sites import find_calls


ROOT = Path(__file__).resolve().parents[1]


def test_audited_optional_modes_are_test_only_or_unused() -> None:
    inventory = {
        name: find_calls(ROOT, name)
        for name in (
            "initialize_from_hydra",
            "execute_experiment",
            "stage_lifecycle",
            "load_stage_execution",
            "prune_completed_prefix",
        )
    }

    assert not any(
        "repository_root" in site.keywords
        for site in inventory["initialize_from_hydra"]
    )
    assert all(
        site.is_test
        for site in inventory["execute_experiment"]
        if "tracking" in site.keywords
    )
    assert all(
        site.is_test
        for site in inventory["stage_lifecycle"]
        if "hook" in site.keywords
    )
    assert all(
        "script" in site.keywords
        for site in inventory["load_stage_execution"]
        if not site.is_test
    )
    assert all(
        "cache_enabled" in site.keywords
        for site in inventory["prune_completed_prefix"]
        if not site.is_test
    )


def test_required_public_files_and_flat_config_tree() -> None:
    public_files = {
        ROOT / "experiment" / name
        for name in ("plan.py", "run.py", "status.py", "inspect.py", "reset.py")
    }
    assert all(path.is_file() for path in public_files)
    assert (ROOT / "configs" / "config.yaml").is_file()
    assert not (ROOT / "configs" / "experiment").exists()


def test_forbidden_config_and_tracking_surfaces_are_absent() -> None:
    forbidden_names = {
        "_base.yaml",
        "2wiki_tiny.yaml",
        "2wiki_rgcn_ablation_7methods.yaml",
        "hotpotqa_dev_full.yaml",
        "hotpotqa_rgcn_ablation_selected.yaml",
        "status_command.yaml",
        "inspect_command.yaml",
        "reset_command.yaml",
    }
    config_files = tuple((ROOT / "configs").rglob("*.yaml"))
    assert not {path.name for path in config_files} & forbidden_names
    assert not (ROOT / "configs" / "tracking").exists()
    assert all("method_configs@method_configs." not in path.read_text(encoding="utf-8") for path in config_files)

    active_sources = [ROOT / "configs", ROOT / "graph_memory", ROOT / "experiment"]
    offenders = [
        path
        for source in active_sources
        if source.exists()
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml"}
        and "mlruns/" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_fixed_tracking_paths_are_canonical() -> None:
    root_config = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    assert "runs/.mlflow/tracking.db" in root_config
    assert "runs/.mlflow/artifacts" in root_config

