from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
)
from graph_memory.query_synthesis.provenance.authoring import MemoryQueryMode
from graph_memory.retrieval.methods.ids import RetrievalMethodId

UnitMetric = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
OptionalUnitMetric: TypeAlias = UnitMetric | Literal["N/A"]
MetricValue: TypeAlias = str | float
MetricTableRow: TypeAlias = dict[str, MetricValue]
MetricSuiteRow: TypeAlias = dict[str, MetricValue]


class MetricRow(DomainModel):
    method: RetrievalMethodId = Field(alias="Method")
    evaluation_schema: NonEmptyStr = Field(alias="Evaluation Schema")
    recall_at_2: UnitMetric = Field(alias="Recall@2")
    recall_at_5: UnitMetric = Field(alias="Recall@5")
    recall_at_10: UnitMetric = Field(alias="Recall@10")
    evidence_f1_at_5: UnitMetric = Field(alias="Evidence F1@5")
    evidence_f1_at_10: UnitMetric = Field(alias="Evidence F1@10")
    evidence_density_at_5: OptionalUnitMetric = Field(
        default="N/A", alias="Evidence Density@5"
    )
    evidence_density_at_10: OptionalUnitMetric = Field(
        default="N/A", alias="Evidence Density@10"
    )
    coverage_at_256_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@256 Tokens"
    )
    coverage_at_512_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@512 Tokens"
    )
    coverage_at_1024_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@1024 Tokens"
    )
    coverage_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@2048 Tokens"
    )
    coverage_at_4096_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@4096 Tokens"
    )
    coverage_at_8192_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@8192 Tokens"
    )
    full_support_at_256_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@256 Tokens"
    )
    full_support_at_512_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@512 Tokens"
    )
    full_support_at_1024_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@1024 Tokens"
    )
    full_support_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@2048 Tokens"
    )
    full_support_at_4096_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@4096 Tokens"
    )
    full_support_at_8192_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@8192 Tokens"
    )
    coverage_budget_auc: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage Budget-AUC"
    )
    full_support_budget_auc: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support Budget-AUC"
    )
    span_f1_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Span F1@2048 Tokens"
    )
    evidence_density_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Evidence Density@2048 Tokens"
    )
    full_support_at_5: UnitMetric = Field(alias="Full Support@5")
    full_support_at_10: UnitMetric = Field(alias="Full Support@10")
    mrr: UnitMetric = Field(alias="MRR")
    connected_evidence_recall_at_5: OptionalUnitMetric = Field(
        alias="Connected Evidence Recall@5"
    )
    connected_evidence_recall_at_10: OptionalUnitMetric = Field(
        alias="Connected Evidence Recall@10"
    )
    query_evidence_connectivity_at_10: OptionalUnitMetric = Field(
        alias="Query-Evidence Connectivity@10"
    )
    path_recall_at_10: OptionalUnitMetric = Field(alias="Path Recall@10")
    edge_recall_at_10: OptionalUnitMetric = Field(alias="Edge Recall@10")
    edge_precision_at_10: OptionalUnitMetric = Field(alias="Edge Precision@10")
    edge_f1_at_10: OptionalUnitMetric = Field(alias="Edge F1@10")
    retrieval_latency_per_query: NonNegativeFiniteFloat = Field(
        alias="Retrieval Latency / Query"
    )
    index_build_time: NonNegativeFiniteFloat = Field(alias="Index Build Time")
    graph_construction_time: NonNegativeFiniteFloat = Field(
        alias="Graph Construction Time"
    )
    memory_size: NonNegativeFiniteFloat | Literal["N/A"] = Field(alias="Memory Size")
    avg_retrieved_nodes: NonNegativeFiniteFloat = Field(alias="Avg Retrieved Nodes")
    avg_retrieved_edges: NonNegativeFiniteFloat = Field(alias="Avg Retrieved Edges")


class TaskMetricRow(DomainModel):
    recall_at_2: UnitMetric = Field(alias="Recall@2")
    recall_at_5: UnitMetric = Field(alias="Recall@5")
    recall_at_10: UnitMetric = Field(alias="Recall@10")
    evidence_f1_at_5: UnitMetric = Field(alias="Evidence F1@5")
    evidence_f1_at_10: UnitMetric = Field(alias="Evidence F1@10")
    evidence_density_at_5: OptionalUnitMetric = Field(
        default="N/A", alias="Evidence Density@5"
    )
    evidence_density_at_10: OptionalUnitMetric = Field(
        default="N/A", alias="Evidence Density@10"
    )
    coverage_at_256_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@256 Tokens"
    )
    coverage_at_512_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@512 Tokens"
    )
    coverage_at_1024_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@1024 Tokens"
    )
    coverage_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@2048 Tokens"
    )
    coverage_at_4096_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@4096 Tokens"
    )
    coverage_at_8192_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage@8192 Tokens"
    )
    full_support_at_256_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@256 Tokens"
    )
    full_support_at_512_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@512 Tokens"
    )
    full_support_at_1024_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@1024 Tokens"
    )
    full_support_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@2048 Tokens"
    )
    full_support_at_4096_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@4096 Tokens"
    )
    full_support_at_8192_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support@8192 Tokens"
    )
    coverage_budget_auc: OptionalUnitMetric = Field(
        default="N/A", alias="Coverage Budget-AUC"
    )
    full_support_budget_auc: OptionalUnitMetric = Field(
        default="N/A", alias="Full Support Budget-AUC"
    )
    span_f1_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Span F1@2048 Tokens"
    )
    evidence_density_at_2048_tokens: OptionalUnitMetric = Field(
        default="N/A", alias="Evidence Density@2048 Tokens"
    )
    full_support_at_5: UnitMetric = Field(alias="Full Support@5")
    full_support_at_10: UnitMetric = Field(alias="Full Support@10")
    mrr: UnitMetric = Field(alias="MRR")
    connected_evidence_recall_at_5: UnitMetric = Field(
        alias="Connected Evidence Recall@5"
    )
    connected_evidence_recall_at_10: UnitMetric = Field(
        alias="Connected Evidence Recall@10"
    )
    query_evidence_connectivity_at_10: UnitMetric = Field(
        alias="Query-Evidence Connectivity@10"
    )
    retrieval_latency_per_query: NonNegativeFiniteFloat = Field(
        alias="Retrieval Latency / Query"
    )
    memory_size: NonNegativeFiniteFloat = Field(alias="Memory Size")
    avg_retrieved_nodes: NonNegativeFiniteFloat = Field(alias="Avg Retrieved Nodes")
    avg_retrieved_edges: NonNegativeFiniteFloat = Field(alias="Avg Retrieved Edges")


class PerTaskMetricRow(TaskMetricRow):
    task_id: NonEmptyStr
    graph_id: NonEmptyStr | None = None
    memory_mode: MemoryQueryMode | None = None
    query_intent: NonEmptyStr | None = None
    motif_type: NonEmptyStr | None = None
    review_status: NonEmptyStr | None = None


class FailureCase(DomainModel):
    debug_type: NonEmptyStr
    task_id: NonEmptyStr
    method: RetrievalMethodId
    failure_type: NonEmptyStr
    gold_evidence_item_ids: tuple[NonEmptyStr, ...]
    retrieved_top_k: tuple[NonEmptyStr, ...]
    missing_gold_nodes: tuple[NonEmptyStr, ...]
    connected_gold_in_top_k: bool


def evidence_metric_columns() -> list[str]:
    return [
        field.serialization_alias or name
        for name, field in MetricRow.model_fields.items()
    ]


EVIDENCE_METRIC_COLUMNS = evidence_metric_columns()


__all__ = [
    "EVIDENCE_METRIC_COLUMNS",
    "FailureCase",
    "MetricRow",
    "MetricSuiteRow",
    "MetricTableRow",
    "MetricValue",
    "PerTaskMetricRow",
    "TaskMetricRow",
    "evidence_metric_columns",
]
