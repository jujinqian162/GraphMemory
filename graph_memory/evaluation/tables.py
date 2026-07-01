from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.contracts.metrics import MetricTableRow, MetricTableSchema, MetricValue

MAIN_RESULT_COLUMNS = [
    "Method",
    "Recall@2",
    "Recall@5",
    "Recall@10",
    "Evidence F1@5",
    "Evidence F1@10",
    "Full Support@5",
    "Full Support@10",
    "MRR",
]

PATH_RESULT_COLUMNS = [
    "Method",
    "Connected Evidence Recall@5",
    "Connected Evidence Recall@10",
    "Query-Evidence Connectivity@10",
    "Path Recall@10",
    "Edge Recall@10",
]

EFFICIENCY_RESULT_COLUMNS = [
    "Method",
    "Index Build Time",
    "Graph Construction Time",
    "Retrieval Latency / Query",
    "Memory Size",
    "Avg Retrieved Nodes",
    "Avg Retrieved Edges",
]

WIDE_METRIC_COLUMNS = [
    *MAIN_RESULT_COLUMNS,
    "Connected Evidence Recall@5",
    "Connected Evidence Recall@10",
    "Query-Evidence Connectivity@10",
    "Path Recall@10",
    "Edge Recall@10",
    "Retrieval Latency / Query",
    "Index Build Time",
    "Graph Construction Time",
    "Memory Size",
    "Avg Retrieved Nodes",
    "Avg Retrieved Edges",
]

LONGMEMEVAL_MAIN_RESULT_COLUMNS = [
    "Method",
    "Turn Recall@5",
    "Turn Recall@10",
    "Full Turn Support@10",
    "Session Recall@5",
    "Session Recall@10",
    "Full Session Support@10",
    "MRR",
]

LONGMEMEVAL_PATH_RESULT_COLUMNS = [
    "Method",
    "Path Recall@10",
    "Edge Recall@10",
]

LONGMEMEVAL_EFFICIENCY_RESULT_COLUMNS = [
    "Method",
    "Retrieval Latency / Query",
    "Memory Size",
    "Avg Retrieved Nodes",
    "Avg Retrieved Edges",
]

LONGMEMEVAL_WIDE_METRIC_COLUMNS = [
    *LONGMEMEVAL_MAIN_RESULT_COLUMNS,
    "Path Recall@10",
    "Edge Recall@10",
    "Retrieval Latency / Query",
    "Memory Size",
    "Avg Retrieved Nodes",
    "Avg Retrieved Edges",
]

EVIDENCE_METRIC_TABLE_SCHEMA = MetricTableSchema(
    name="evidence",
    main_columns=tuple(MAIN_RESULT_COLUMNS),
    path_columns=tuple(PATH_RESULT_COLUMNS),
    efficiency_columns=tuple(EFFICIENCY_RESULT_COLUMNS),
    wide_columns=tuple(WIDE_METRIC_COLUMNS),
)

LONGMEMEVAL_METRIC_TABLE_SCHEMA = MetricTableSchema(
    name="longmemeval",
    main_columns=tuple(LONGMEMEVAL_MAIN_RESULT_COLUMNS),
    path_columns=tuple(LONGMEMEVAL_PATH_RESULT_COLUMNS),
    efficiency_columns=tuple(LONGMEMEVAL_EFFICIENCY_RESULT_COLUMNS),
    wide_columns=tuple(LONGMEMEVAL_WIDE_METRIC_COLUMNS),
)

_METRIC_TABLE_SCHEMAS = {
    EVIDENCE_METRIC_TABLE_SCHEMA.name: EVIDENCE_METRIC_TABLE_SCHEMA,
    LONGMEMEVAL_METRIC_TABLE_SCHEMA.name: LONGMEMEVAL_METRIC_TABLE_SCHEMA,
}


def metric_table_schema_for_suite(name: str) -> MetricTableSchema:
    try:
        return _METRIC_TABLE_SCHEMAS[name]
    except KeyError as error:
        allowed = ", ".join(sorted(_METRIC_TABLE_SCHEMAS))
        raise ValueError(f"Unsupported metric suite: {name}. Allowed values: {allowed}.") from error


def split_metric_tables(
    rows: Sequence[Mapping[str, MetricValue]],
    *,
    schema: MetricTableSchema = EVIDENCE_METRIC_TABLE_SCHEMA,
) -> tuple[list[MetricTableRow], list[MetricTableRow], list[MetricTableRow]]:
    main_rows = [_select_columns(row, schema.main_columns) for row in rows]
    path_rows = [_select_columns(row, schema.path_columns) for row in rows]
    efficiency_rows = [_select_columns(row, schema.efficiency_columns) for row in rows]
    return main_rows, path_rows, efficiency_rows


def _select_columns(row: Mapping[str, MetricValue], columns: Sequence[str]) -> MetricTableRow:
    return {column: row[column] for column in columns}


__all__ = [
    "EFFICIENCY_RESULT_COLUMNS",
    "EVIDENCE_METRIC_TABLE_SCHEMA",
    "LONGMEMEVAL_EFFICIENCY_RESULT_COLUMNS",
    "LONGMEMEVAL_MAIN_RESULT_COLUMNS",
    "LONGMEMEVAL_METRIC_TABLE_SCHEMA",
    "LONGMEMEVAL_PATH_RESULT_COLUMNS",
    "LONGMEMEVAL_WIDE_METRIC_COLUMNS",
    "MAIN_RESULT_COLUMNS",
    "PATH_RESULT_COLUMNS",
    "WIDE_METRIC_COLUMNS",
    "metric_table_schema_for_suite",
    "split_metric_tables",
]
