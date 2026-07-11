from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.service import initialize_experiment
from scripts.deliver.collect_run_artifacts import collect_run_artifacts, main

ROOT = Path(__file__).resolve().parents[1]


def test_collect_run_artifacts_preserves_selected_paths(tmp_path: Path) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(
        run_dir, output_root=tmp_path / "results", max_file_size_bytes=1024 * 1024
    )

    output_dir = tmp_path / "results" / "rgcn_full_train"
    copied_paths = {entry["relative_path"] for entry in manifest["copied"]}

    assert {
        "run_state.yaml",
        "config/resolved.yaml",
        "config/overrides.yaml",
        "tables/main_results.csv",
        "metrics/test.dense_rgcn_graph_retriever.metrics.csv",
        "metrics/test.dense_rgcn_graph_retriever.metrics.run_summary.yaml",
        "graphs/test.graphs.stats.json",
        "graphs/test.graphs.run_summary.yaml",
        "learned/dense_rgcn_graph_retriever/train_metrics.jsonl",
        "learned/dense_rgcn_graph_retriever/train.run_summary.yaml",
        "learned/dense_rgcn_graph_retriever/train.pairs.summary.json",
        "learned/dense_rgcn_graph_retriever/train.pairs.run_summary.yaml",
        "tuned/dense_graph_rerank.dev_selected.json",
        "debug/failure_cases_dense_rgcn_graph_retriever.jsonl",
    }.issubset(copied_paths)
    assert (output_dir / "tables" / "main_results.csv").read_text(
        encoding="utf-8"
    ) == "method,Recall@5\nrgcn,0.8\n"


def test_collect_run_artifacts_excludes_large_intermediates_and_records_reasons(
    tmp_path: Path,
) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(
        run_dir, output_root=tmp_path / "results", max_file_size_bytes=32
    )

    skipped = {entry["relative_path"]: entry["reason"] for entry in manifest["skipped"]}

    assert skipped["graphs/test.graphs.json"] == "excluded_graph"
    assert skipped["inputs/train.input.json"] == "excluded_input"
    assert skipped["predictions/test.dense.ranked.json"] == "excluded_prediction"
    assert (
        skipped["learned/dense_rgcn_graph_retriever/checkpoints/best.pt"]
        == "excluded_checkpoint"
    )
    assert (
        skipped["learned/dense_rgcn_graph_retriever/train.pairs.json"]
        == "excluded_train_pairs"
    )
    assert (
        skipped["debug/failure_cases_dense_rgcn_graph_retriever.jsonl"] == "too_large"
    )


def test_collect_run_artifacts_preserves_ablation_lightweight_outputs(
    tmp_path: Path,
) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(
        run_dir, output_root=tmp_path / "results", max_file_size_bytes=1024 * 1024
    )

    copied_paths = {entry["relative_path"] for entry in manifest["copied"]}
    skipped = {entry["relative_path"]: entry["reason"] for entry in manifest["skipped"]}

    assert {
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train_metrics.jsonl",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.run_summary.yaml",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.pairs.summary.json",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.pairs.run_summary.yaml",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/metrics/test.metrics.csv",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/metrics/test.metrics.run_summary.yaml",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/debug/failure_cases.jsonl",
    }.issubset(copied_paths)
    assert (
        skipped["ablations/dense_rgcn_graph_retriever/wo_bridge/train.pairs.json"]
        == "excluded_train_pairs"
    )
    assert (
        skipped["ablations/dense_rgcn_graph_retriever/wo_bridge/checkpoints/best.pt"]
        == "excluded_checkpoint"
    )
    assert (
        skipped[
            "ablations/dense_rgcn_graph_retriever/wo_bridge/predictions/test.ranked.json"
        ]
        == "excluded_prediction"
    )


def test_collect_run_artifacts_writes_delivery_manifest(tmp_path: Path) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(
        run_dir, output_root=tmp_path / "results", max_file_size_bytes=1024 * 1024
    )
    manifest_path = tmp_path / "results" / "rgcn_full_train" / "delivery_manifest.json"

    assert manifest_path.exists()
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored == manifest
    assert stored["source_run_dir"] == str(run_dir.resolve())
    assert stored["output_dir"] == str(
        (tmp_path / "results" / "rgcn_full_train").resolve()
    )
    assert stored["total_copied_bytes"] > 0
    assert stored["max_file_size_bytes"] == 1024 * 1024


def test_collect_run_artifacts_missing_run_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        collect_run_artifacts(
            tmp_path / "runs" / "missing", output_root=tmp_path / "results"
        )

    assert not (tmp_path / "results" / "missing" / "delivery_manifest.json").exists()


def test_collect_run_artifacts_cli_uses_name_convention_and_default_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _make_run_tree(tmp_path)
    assert run_dir == tmp_path / "runs" / "rgcn_full_train"
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "--name",
            "rgcn_full_train",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "copied=" in captured.out
    assert not (
        tmp_path / "results" / "rgcn_full_train" / "delivery_manifest.json"
    ).exists()


def test_collect_run_artifacts_help_documents_name_based_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--name" in captured.out
    assert "--run-dir" not in captured.out
    assert "Contract" in captured.out
    assert "runs/<name>" in captured.out
    assert "results/<name>" in captured.out


def _make_run_tree(tmp_path: Path) -> Path:
    name = "rgcn_full_train"
    raw = tmp_path / "data/hotpotqa/raw"
    raw.mkdir(parents=True)
    (raw / "train.json").write_text("[]\n", encoding="utf-8")
    (raw / "dev.json").write_text("[]\n", encoding="utf-8")
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"), version_base="1.3"
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "profile=smoke",
                "methods=[bm25]",
                "stages.to=prepare",
            ],
        )
    config = resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=tmp_path,
    )
    layout = RunLayout(tmp_path, name)
    initialized = initialize_experiment(
        config,
        layout=layout,
        overrides=(
            f"name={name}",
            "profile=smoke",
            "methods=[bm25]",
            "stages.to=prepare",
        ),
    )
    assert initialized.state.name == name
    run_dir = layout.run_dir
    files = {
        "tables/main_results.csv": "method,Recall@5\nrgcn,0.8\n",
        "metrics/test.dense_rgcn_graph_retriever.metrics.csv": "metric,value\nRecall@5,0.8\n",
        "metrics/test.dense_rgcn_graph_retriever.metrics.run_summary.yaml": "status: success\n",
        "graphs/test.graphs.stats.json": "{}\n",
        "graphs/test.graphs.run_summary.yaml": "status: success\n",
        "graphs/test.graphs.json": "x" * 128,
        "inputs/train.input.json": "x" * 128,
        "predictions/test.dense.ranked.json": "x" * 128,
        "learned/dense_rgcn_graph_retriever/train_metrics.jsonl": '{"epoch":1}\n',
        "learned/dense_rgcn_graph_retriever/train.run_summary.yaml": "status: success\n",
        "learned/dense_rgcn_graph_retriever/train.pairs.json": "x" * 128,
        "learned/dense_rgcn_graph_retriever/train.pairs.summary.json": "{}\n",
        "learned/dense_rgcn_graph_retriever/train.pairs.run_summary.yaml": "status: success\n",
        "learned/dense_rgcn_graph_retriever/checkpoints/best.pt": "x" * 128,
        "tuned/dense_graph_rerank.dev_selected.json": "{}\n",
        "tuned/dense_graph_rerank.dev_selected.candidates.json": "x" * 128,
        "debug/failure_cases_dense_rgcn_graph_retriever.jsonl": '{"task_id":"1"}\n' * 4,
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train_metrics.jsonl": '{"epoch":1}\n',
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.run_summary.yaml": "status: success\n",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.pairs.json": "x" * 128,
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.pairs.summary.json": "{}\n",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/train.pairs.run_summary.yaml": "status: success\n",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/checkpoints/best.pt": "x" * 128,
        "ablations/dense_rgcn_graph_retriever/wo_bridge/predictions/test.ranked.json": "x"
        * 128,
        "ablations/dense_rgcn_graph_retriever/wo_bridge/metrics/test.metrics.csv": "metric,value\nRecall@5,0.7\n",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/metrics/test.metrics.run_summary.yaml": "status: success\n",
        "ablations/dense_rgcn_graph_retriever/wo_bridge/debug/failure_cases.jsonl": '{"task_id":"1"}\n',
    }
    for relative_path, content in files.items():
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return run_dir
