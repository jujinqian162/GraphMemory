from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.deliver.collect_run_artifacts import collect_run_artifacts, main


def test_collect_run_artifacts_preserves_selected_paths(tmp_path: Path) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(run_dir, output_root=tmp_path / "results", max_file_size_bytes=1024 * 1024)

    output_dir = tmp_path / "results" / "rgcn_full_train"
    copied_paths = {entry["relative_path"] for entry in manifest["copied"]}

    assert output_dir / "manifest.json" in [output_dir / path for path in copied_paths]
    assert {
        "manifest.json",
        "config/effective_config.json",
        "tables/main_results.csv",
        "metrics/test.dense_rgcn_graph_retriever.metrics.csv",
        "metrics/test.dense_rgcn_graph_retriever.metrics.run_summary.json",
        "graphs/test.graphs.stats.json",
        "graphs/test.graphs.run_summary.json",
        "learned/dense_rgcn_graph_retriever/effective_training_config.json",
        "learned/dense_rgcn_graph_retriever/train_metrics.jsonl",
        "learned/dense_rgcn_graph_retriever/train_run_summary.json",
        "learned/dense_rgcn_graph_retriever/train.pairs.summary.json",
        "learned/dense_rgcn_graph_retriever/train.pairs.run_summary.json",
        "tuned/dense_graph_rerank.dev_selected.json",
        "debug/failure_cases_dense_rgcn_graph_retriever.jsonl",
    }.issubset(copied_paths)
    assert (output_dir / "tables" / "main_results.csv").read_text(encoding="utf-8") == "method,Recall@5\nrgcn,0.8\n"


def test_collect_run_artifacts_excludes_large_intermediates_and_records_reasons(tmp_path: Path) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(run_dir, output_root=tmp_path / "results", max_file_size_bytes=32)

    skipped = {entry["relative_path"]: entry["reason"] for entry in manifest["skipped"]}

    assert skipped["graphs/test.graphs.json"] == "excluded_graph"
    assert skipped["inputs/train.input.json"] == "excluded_input"
    assert skipped["predictions/test.dense.ranked.json"] == "excluded_prediction"
    assert skipped["learned/dense_rgcn_graph_retriever/checkpoints/best.pt"] == "excluded_checkpoint"
    assert skipped["learned/dense_rgcn_graph_retriever/train.pairs.json"] == "excluded_train_pairs"
    assert skipped["debug/failure_cases_dense_rgcn_graph_retriever.jsonl"] == "too_large"


def test_collect_run_artifacts_writes_delivery_manifest(tmp_path: Path) -> None:
    run_dir = _make_run_tree(tmp_path)
    manifest = collect_run_artifacts(run_dir, output_root=tmp_path / "results", max_file_size_bytes=1024 * 1024)
    manifest_path = tmp_path / "results" / "rgcn_full_train" / "delivery_manifest.json"

    assert manifest_path.exists()
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored == manifest
    assert stored["source_run_dir"] == str(run_dir.resolve())
    assert stored["output_dir"] == str((tmp_path / "results" / "rgcn_full_train").resolve())
    assert stored["total_copied_bytes"] > 0
    assert stored["max_file_size_bytes"] == 1024 * 1024


def test_collect_run_artifacts_missing_run_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        collect_run_artifacts(tmp_path / "runs" / "missing", output_root=tmp_path / "results")

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
    assert not (tmp_path / "results" / "rgcn_full_train" / "delivery_manifest.json").exists()


def test_collect_run_artifacts_help_documents_name_based_contract(capsys: pytest.CaptureFixture[str]) -> None:
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
    run_dir = tmp_path / "runs" / "rgcn_full_train"
    files = {
        "manifest.json": "{}\n",
        "config/effective_config.json": "{}\n",
        "tables/main_results.csv": "method,Recall@5\nrgcn,0.8\n",
        "metrics/test.dense_rgcn_graph_retriever.metrics.csv": "metric,value\nRecall@5,0.8\n",
        "metrics/test.dense_rgcn_graph_retriever.metrics.run_summary.json": "{}\n",
        "graphs/test.graphs.stats.json": "{}\n",
        "graphs/test.graphs.run_summary.json": "{}\n",
        "graphs/test.graphs.json": "x" * 128,
        "inputs/train.input.json": "x" * 128,
        "predictions/test.dense.ranked.json": "x" * 128,
        "learned/dense_rgcn_graph_retriever/effective_training_config.json": "{}\n",
        "learned/dense_rgcn_graph_retriever/train_metrics.jsonl": '{"epoch":1}\n',
        "learned/dense_rgcn_graph_retriever/train_run_summary.json": "{}\n",
        "learned/dense_rgcn_graph_retriever/train.pairs.json": "x" * 128,
        "learned/dense_rgcn_graph_retriever/train.pairs.summary.json": "{}\n",
        "learned/dense_rgcn_graph_retriever/train.pairs.run_summary.json": "{}\n",
        "learned/dense_rgcn_graph_retriever/checkpoints/best.pt": "x" * 128,
        "tuned/dense_graph_rerank.dev_selected.json": "{}\n",
        "tuned/dense_graph_rerank.dev_selected.candidates.json": "x" * 128,
        "debug/failure_cases_dense_rgcn_graph_retriever.jsonl": '{"task_id":"1"}\n' * 4,
    }
    for relative_path, content in files.items():
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return run_dir
