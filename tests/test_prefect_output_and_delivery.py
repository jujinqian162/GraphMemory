from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from pydantic import JsonValue

from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    EvaluationArtifactRef,
    PredictionsArtifactRef,
    ProcessedAssetStore,
)
from graph_memory.experiment.config import (
    ResolvedExperimentConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.output import project_run_output
from graph_memory.experiment.results import FinalExperimentResult
from graph_memory.io import write_json, write_jsonl
from graph_memory.stages.results import EvaluationResult, RankingResult
from scripts.deliver.collect_run_artifacts import collect_run_artifacts


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> ResolvedExperimentConfig:
    source = (ROOT / "tests" / "fixtures" / "hotpotqa_smoke.json").as_posix()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "dataset=hotpotqa",
                "profile=smoke",
                "method=bm25",
                "device=cpu",
                *[
                    f"dataset.splits.{split}.source={source}"
                    for split in ("train", "dev", "test")
                ],
            ],
        )
    return resolve_experiment_config(
        parse_composed_config(composed), repository_root=ROOT
    )


def _result(store: ProcessedAssetStore) -> FinalExperimentResult:
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.PREDICTIONS,
        namespace="bm25",
        task_identity="test-rank",
        origin={"stage": "rank", "method": "bm25"},
    ) as publisher:
        write_json(publisher.workspace / "rankings.json", [{"task_id": "one"}])
        ranking_ref = publisher.publish({"rankings": "rankings.json"})
    assert isinstance(ranking_ref, PredictionsArtifactRef)

    metric_row: dict[str, JsonValue] = {
        "Method": "bm25",
        "Recall@5": 1.0,
        "Recall@10": 1.0,
        "MRR": 1.0,
        "nDCG@10": 1.0,
        "Evidence F1": 1.0,
        "Full Support EM": 1.0,
        "Path Recall@5": 1.0,
        "Path Recall@10": 1.0,
        "Path MRR": 1.0,
        "Avg Retrieved Nodes": 1.0,
        "Avg Retrieved Edges": 0.0,
        "Retrieval Latency / Query": 3.0,
        "Index Build Time": "NA",
        "Graph Construction Time": "NA",
        "Memory Size": "NA",
    }
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.EVALUATION,
        namespace="bm25",
        task_identity="test-evaluate",
        origin={"stage": "evaluate", "method": "bm25"},
    ) as publisher:
        write_json(publisher.workspace / "metrics.json", [metric_row])
        write_jsonl(publisher.workspace / "failure_cases.jsonl", [])
        write_jsonl(publisher.workspace / "per_task.jsonl", [])
        evaluation_ref = publisher.publish(
            {
                "metrics": "metrics.json",
                "failure_cases": "failure_cases.jsonl",
                "per_task": "per_task.jsonl",
            }
        )
    assert isinstance(evaluation_ref, EvaluationArtifactRef)

    ranking = RankingResult(
        method="bm25",
        artifact=ranking_ref,
        provenance={"method": "bm25"},
        production_seconds=3.0,
    )
    evaluation = EvaluationResult(
        method="bm25",
        artifact=evaluation_ref,
        metric_rows=(metric_row,),
        per_task_rows=(),
        failure_case_count=0,
    )
    return FinalExperimentResult(
        method="bm25",
        variant=None,
        ranking=ranking,
        evaluation=evaluation,
        assets=(ranking_ref, evaluation_ref),
    )


def test_output_projection_is_complete_and_never_copies_processed_assets(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    store = ProcessedAssetStore(repository / "data" / "processed")
    output = repository / "runs" / "named-run"
    result = _result(store)

    project_run_output(
        output,
        repository_root=repository,
        config=_config("named-run"),
        overrides=("method=bm25",),
        result=result,
    )

    summary = yaml.safe_load((output / "workflow" / "summary.yaml").read_text())
    manifest = yaml.safe_load((output / "assets" / "manifest.yaml").read_text())
    assert summary["method"] == "bm25"
    assert summary["seed"] == 13
    assert summary["cache_refresh"] is False
    assert {asset["digest"] for asset in manifest["assets"]} == {
        asset.digest for asset in result.assets
    }
    assert not (output / "workflow" / "task_results.yaml").exists()
    with (output / "metrics" / "final.metrics.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Method"] == "bm25"
    assert row["Retrieval Latency / Query"] == "NA"
    assert not any(path.suffix in {".pt", ".ckpt"} for path in output.rglob("*"))


def test_delivery_mirrors_output_tree_and_records_asset_references(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    output = repository / "runs" / "named-run"
    result = _result(ProcessedAssetStore(repository / "data" / "processed"))
    project_run_output(
        output,
        repository_root=repository,
        config=_config("named-run"),
        overrides=(),
        result=result,
    )

    delivery = collect_run_artifacts(output, output_root=repository / "results")
    delivered = repository / "results" / "named-run"
    assert delivery["run_mode"] == "single"
    assert (delivered / "assets" / "manifest.yaml").is_file()
    assert not (delivered / "data" / "processed").exists()
    manifest = json.loads(
        (delivered / "delivery_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["copied_count"] == delivery["copied_count"]


def test_delivery_detects_every_output_only_multirun_child(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "study"
    for selector in ("0_method=bm25", "1_method=dense"):
        summary = source / selector / "workflow" / "summary.yaml"
        summary.parent.mkdir(parents=True)
        summary.write_text("method: test\n", encoding="utf-8")

    manifest = collect_run_artifacts(source, output_root=tmp_path / "results")

    assert manifest["run_mode"] == "multirun"
    assert manifest["jobs"] == ["0_method=bm25", "1_method=dense"]
    assert all(
        (
            tmp_path / "results" / "study" / selector / "workflow" / "summary.yaml"
        ).is_file()
        for selector in manifest["jobs"]
    )


def test_run_output_guard_rejects_root_and_external_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    result = _result(ProcessedAssetStore(repository / "data" / "processed"))
    for invalid in (repository / "runs", repository / "outside"):
        try:
            project_run_output(
                invalid,
                repository_root=repository,
                config=_config("invalid-run"),
                overrides=(),
                result=result,
            )
        except ValueError as error:
            assert "named child below runs" in str(error)
        else:
            raise AssertionError(f"expected output guard rejection for {invalid}")
