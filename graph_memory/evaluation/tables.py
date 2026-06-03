from __future__ import annotations

from graph_memory.contracts.metrics import MetricRow, MetricTableRow

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


def split_metric_tables(
    rows: list[MetricRow],
) -> tuple[list[MetricTableRow], list[MetricTableRow], list[MetricTableRow]]:
    main_rows = [_select_columns(row, MAIN_RESULT_COLUMNS) for row in rows]
    path_rows = [_select_columns(row, PATH_RESULT_COLUMNS) for row in rows]
    efficiency_rows = [_select_columns(row, EFFICIENCY_RESULT_COLUMNS) for row in rows]
    return main_rows, path_rows, efficiency_rows


def _select_columns(row: MetricRow, columns: list[str]) -> MetricTableRow:
    return {column: row[column] for column in columns}


__all__ = [
    "EFFICIENCY_RESULT_COLUMNS",
    "MAIN_RESULT_COLUMNS",
    "PATH_RESULT_COLUMNS",
    "WIDE_METRIC_COLUMNS",
    "split_metric_tables",
]
