from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory.experiment import (
    build_stage_plan,
    initialize_experiment,
    inspect_experiment_status,
    load_experiment_config,
)
import scripts.experiment as experiment_script


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


def test_repository_experiment_configs_have_clear_roles():
    assert Path("configs/experiments/hotpotqa_evidence_retrieval.json").exists()
    assert Path("configs/search_spaces/graph_rerank.json").exists()
    assert Path("configs/published/README.md").exists()


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
