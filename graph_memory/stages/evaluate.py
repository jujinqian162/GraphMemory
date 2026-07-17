from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.metrics import FailureCase, MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.selection import evidence_evaluation_request_for_dataset
from graph_memory.evaluation.suites import evidence_metric_suite
from graph_memory.evaluation.tables import WIDE_METRIC_COLUMNS
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    EvaluationArtifactRef,
    EvidenceGraphArtifactRef,
    PredictionsArtifactRef,
    ProcessedAssetStore,
    artifact_payload_path,
)
from graph_memory.experiment.config import DatasetName
from graph_memory.io import read_json, write_csv, write_jsonl
from graph_memory.stages.results import EvaluationResult
from graph_memory.validation import validate_metric_rows


@dataclass(frozen=True)
class EvaluateStageResult:
    metric_rows: list[MetricRow]
    failure_cases: list[FailureCase]


def run_evaluate_stage(
    *,
    dataset: DatasetName,
    top_k: int,
    failure_case_limit: int,
    predictions: list[RankedResult],
    labels: list[object],
    graphs: list[EvidenceGraph],
) -> EvaluateStageResult:
    request = evidence_evaluation_request_for_dataset(
        dataset,
        predictions=predictions,
        labels=labels,
        graphs=graphs,
    )
    suite = evidence_metric_suite()
    metric_rows = suite.evaluate(request)
    failure_cases = suite.build_failure_cases(
        request,
        top_k=top_k,
        limit=failure_case_limit,
    )
    validate_metric_rows(metric_rows)
    return EvaluateStageResult(metric_rows=metric_rows, failure_cases=failure_cases)


def materialize_evaluation(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    top_k: int,
    failure_case_limit: int,
    predictions: PredictionsArtifactRef,
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    implementation_version: str,
) -> EvaluationResult:
    method = str(predictions.origin["method"])
    variant_value = predictions.origin.get("variant")
    variant = variant_value if isinstance(variant_value, str) else None
    prediction_values = cast(
        list[RankedResult], read_json(artifact_payload_path(predictions, "predictions"))
    )
    labels = cast(list[object], read_json(artifact_payload_path(prepared, "labels")))
    graphs = (
        cast(
            list[EvidenceGraph],
            read_json(artifact_payload_path(evidence_graphs, "graphs")),
        )
        if evidence_graphs is not None
        else []
    )
    result = run_evaluate_stage(
        dataset=dataset,
        top_k=top_k,
        failure_case_limit=failure_case_limit,
        predictions=prediction_values,
        labels=labels,
        graphs=graphs,
    )
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.EVALUATION,
        namespace=method,
        task_identity=f"evaluate-{method}",
        origin={
            "stage": "evaluate",
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "predictions_digest": predictions.digest,
            "labels_digest": prepared.digest,
            "graph_digest": None if evidence_graphs is None else evidence_graphs.digest,
            "implementation_version": implementation_version,
        },
    ) as publisher:
        write_csv(
            publisher.workspace / "metrics.csv",
            result.metric_rows,
            WIDE_METRIC_COLUMNS,
        )
        write_jsonl(publisher.workspace / "failure_cases.jsonl", result.failure_cases)
        artifact = publisher.publish(
            {
                "metrics": "metrics.csv",
                "failure_cases": "failure_cases.jsonl",
            },
            shape={
                "metric_rows": len(result.metric_rows),
                "failure_cases": len(result.failure_cases),
            },
        )
    assert isinstance(artifact, EvaluationArtifactRef)
    return EvaluationResult(
        method=method,
        artifact=artifact,
        metric_rows=tuple(
            cast(dict[str, JsonValue], dict(row)) for row in result.metric_rows
        ),
        failure_case_count=len(result.failure_cases),
    )


__all__ = ["EvaluateStageResult", "materialize_evaluation", "run_evaluate_stage"]
