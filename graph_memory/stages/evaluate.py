from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pydantic import TypeAdapter

from graph_memory.datasets.isetrace.benchmark_records import ISETraceQueryMetadata
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.evaluation.contracts import FailureCase, MetricRow, PerTaskMetricRow
from graph_memory.retrieval.results import RankedResult
from graph_memory.datasets.selection import evidence_evaluation_request_for_dataset
from graph_memory.evaluation.suites import (
    build_evidence_failure_cases,
    evaluate_evidence_with_per_task,
)
from graph_memory.evaluation.span_suite import SpanEvidenceMetricSuite
from graph_memory.evaluation.requests import SpanEvidenceEvaluationRequest
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


RANKED_RESULTS_ADAPTER = TypeAdapter(list[RankedResult])
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
ISETRACE_QUERY_METADATA_ADAPTER = TypeAdapter(list[ISETraceQueryMetadata])


def run_evaluate_stage(
    *,
    dataset: DatasetName,
    top_k: int,
    failure_case_limit: int,
    predictions: list[RankedResult],
    labels: list[object],
    graphs: list[EvidenceGraph],
    query_metadata: Sequence[ISETraceQueryMetadata] = (),
) -> tuple[list[MetricRow], list[FailureCase], list[PerTaskMetricRow]]:
    request = evidence_evaluation_request_for_dataset(
        dataset,
        predictions=predictions,
        labels=labels,
        graphs=graphs,
    )
    if isinstance(request, SpanEvidenceEvaluationRequest):
        span_suite = SpanEvidenceMetricSuite()
        metric_rows, per_task_rows = span_suite.evaluate_with_per_task(request)
        failure_cases = span_suite.build_failure_cases(
            request,
            top_k=top_k,
            limit=failure_case_limit,
        )
    else:
        metric_rows, per_task_rows = evaluate_evidence_with_per_task(request)
        failure_cases = build_evidence_failure_cases(
            request,
            top_k=top_k,
            limit=failure_case_limit,
        )
    if dataset == "isetrace" and query_metadata:
        per_task_rows = _attach_isetrace_query_metadata(
            per_task_rows,
            query_metadata=query_metadata,
        )
    return metric_rows, failure_cases, per_task_rows


def _attach_isetrace_query_metadata(
    rows: Sequence[PerTaskMetricRow],
    *,
    query_metadata: Sequence[ISETraceQueryMetadata],
) -> list[PerTaskMetricRow]:
    metadata_by_id = {item.task_id: item for item in query_metadata}
    if len(metadata_by_id) != len(query_metadata):
        raise ValueError("ISETrace evaluation query metadata task IDs must be unique")
    row_ids = {row.task_id for row in rows}
    if set(metadata_by_id) != row_ids:
        raise ValueError("ISETrace evaluation rows and query metadata must align")
    return [
        row.model_copy(
            update={
                "graph_id": metadata_by_id[row.task_id].graph_id,
                "query_origin": metadata_by_id[row.task_id].query_origin,
                "memory_mode": metadata_by_id[row.task_id].memory_mode,
            }
        )
        for row in rows
    ]


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
) -> EvaluationArtifactRef:
    method = str(predictions.origin["method"])
    variant_value = predictions.origin.get("variant")
    variant = variant_value if isinstance(variant_value, str) else None
    prediction_values = RANKED_RESULTS_ADAPTER.validate_python(
        read_json(artifact_payload_path(predictions, "predictions"))
    )
    labels = cast(list[object], read_json(artifact_payload_path(prepared, "labels")))
    graphs = (
        EVIDENCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(evidence_graphs, "graphs"))
        )
        if evidence_graphs is not None
        else []
    )
    query_metadata = (
        ISETRACE_QUERY_METADATA_ADAPTER.validate_python(
            read_json(artifact_payload_path(prepared, "query_metadata"))
        )
        if dataset == "isetrace"
        else []
    )
    metric_rows, failure_cases, per_task_rows = run_evaluate_stage(
        dataset=dataset,
        top_k=top_k,
        failure_case_limit=failure_case_limit,
        predictions=prediction_values,
        labels=labels,
        graphs=graphs,
        query_metadata=query_metadata,
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
            [row.model_dump(mode="json", by_alias=True) for row in metric_rows],
            WIDE_METRIC_COLUMNS,
        )
        write_jsonl(
            publisher.workspace / "failure_cases.jsonl",
            [row.model_dump(mode="json") for row in failure_cases],
        )
        write_jsonl(
            publisher.workspace / "per_task.jsonl",
            [row.model_dump(mode="json", by_alias=True) for row in per_task_rows],
        )
        artifact = publisher.publish(
            {
                "metrics": "metrics.csv",
                "failure_cases": "failure_cases.jsonl",
                "per_task": "per_task.jsonl",
            },
            shape={
                "metric_rows": len(metric_rows),
                "failure_cases": len(failure_cases),
                "per_task_rows": len(per_task_rows),
            },
        )
    assert isinstance(artifact, EvaluationArtifactRef)
    return artifact


__all__ = ["materialize_evaluation", "run_evaluate_stage"]
