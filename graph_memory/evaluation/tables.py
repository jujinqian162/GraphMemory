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


def split_metric_tables(
    rows: list[MetricRow],
) -> tuple[list[MetricTableRow], list[MetricTableRow], list[MetricTableRow]]:
    main_columns, path_columns, efficiency_columns, _ = metric_columns_for_rows(rows)
    main_rows = [_select_columns(row, main_columns) for row in rows]
    path_rows = [_select_columns(row, path_columns) for row in rows]
    efficiency_rows = [_select_columns(row, efficiency_columns) for row in rows]
    return main_rows, path_rows, efficiency_rows


def metric_columns_for_rows(
    rows: list[MetricRow],
) -> tuple[list[str], list[str], list[str], list[str]]:
    if _uses_longmemeval_columns(rows):
        return (
            LONGMEMEVAL_MAIN_RESULT_COLUMNS,
            LONGMEMEVAL_PATH_RESULT_COLUMNS,
            LONGMEMEVAL_EFFICIENCY_RESULT_COLUMNS,
            LONGMEMEVAL_WIDE_METRIC_COLUMNS,
        )
    return (
        MAIN_RESULT_COLUMNS,
        PATH_RESULT_COLUMNS,
        EFFICIENCY_RESULT_COLUMNS,
        WIDE_METRIC_COLUMNS,
    )


def _uses_longmemeval_columns(rows: list[MetricRow]) -> bool:
    return bool(rows) and "Turn Recall@5" in rows[0]


def _select_columns(row: MetricRow, columns: list[str]) -> MetricTableRow:
    return {column: row[column] for column in columns}


__all__ = [
    "EFFICIENCY_RESULT_COLUMNS",
    "LONGMEMEVAL_EFFICIENCY_RESULT_COLUMNS",
    "LONGMEMEVAL_MAIN_RESULT_COLUMNS",
    "LONGMEMEVAL_PATH_RESULT_COLUMNS",
    "LONGMEMEVAL_WIDE_METRIC_COLUMNS",
    "MAIN_RESULT_COLUMNS",
    "PATH_RESULT_COLUMNS",
    "WIDE_METRIC_COLUMNS",
    "metric_columns_for_rows",
    "split_metric_tables",
]