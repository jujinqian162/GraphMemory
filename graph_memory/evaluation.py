from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from graph_memory.types import (
    FailureCase,
    GraphEdge,
    MemoryGraph,
    MemoryTaskLabels,
    MetricRow,
    MetricTableRow,
    NodeId,
    RankedResult,
    TaskMetricRow,
)
from graph_memory.validation import ContractValidationError, validate_task_id_alignment

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


def recall_at(ranked_nodes: list[str], gold_nodes: set[str], k: int) -> float:
    _require_gold_nodes(gold_nodes)
    selected = set(ranked_nodes[:k])
    return len(selected & gold_nodes) / len(gold_nodes)


def evidence_f1_at(ranked_nodes: list[str], gold_nodes: set[str], k: int) -> float:
    _require_gold_nodes(gold_nodes)
    hits = len(set(ranked_nodes[:k]) & gold_nodes)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(gold_nodes)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def full_support_at(ranked_nodes: list[str], gold_nodes: set[str], k: int) -> float:
    _require_gold_nodes(gold_nodes)
    return 1.0 if gold_nodes.issubset(set(ranked_nodes[:k])) else 0.0


def mrr(ranked_nodes: list[str], gold_nodes: set[str]) -> float:
    _require_gold_nodes(gold_nodes)
    for index, node_id in enumerate(ranked_nodes, start=1):
        if node_id in gold_nodes:
            return 1.0 / index
    return 0.0


def connected_evidence_at(ranked_nodes: list[NodeId], gold_nodes: set[NodeId], graph: MemoryGraph, k: int) -> float:
    _require_gold_nodes(gold_nodes)
    selected = set(ranked_nodes[:k])
    if not gold_nodes.issubset(selected):
        return 0.0
    if len(gold_nodes) == 1:
        return 1.0
    adjacency = _undirected_adjacency(graph.get("edges", []), selected)
    first_gold = next(iter(gold_nodes))
    reachable = _reachable_from(first_gold, adjacency)
    return 1.0 if gold_nodes.issubset(reachable) else 0.0


def query_evidence_connectivity_at(ranked_nodes: list[NodeId], gold_nodes: set[NodeId], graph: MemoryGraph, k: int) -> float:
    _require_gold_nodes(gold_nodes)
    selected = set(ranked_nodes[:k])
    if not gold_nodes.issubset(selected):
        return 0.0
    allowed_nodes = selected | {"q"}
    adjacency = _directed_adjacency(graph.get("edges", []), allowed_nodes)
    reachable = _reachable_from("q", adjacency)
    return 1.0 if gold_nodes.issubset(reachable) else 0.0


def evaluate_results(
    predictions: list[RankedResult], labels: list[MemoryTaskLabels], graphs: list[MemoryGraph]
) -> list[MetricRow]:
    prediction_task_ids = {prediction["task_id"] for prediction in predictions}
    label_task_ids = {label["task_id"] for label in labels}
    graph_task_ids = {graph["task_id"] for graph in graphs}
    validate_task_id_alignment("prediction/label join", prediction_task_ids, label_task_ids)
    validate_task_id_alignment("prediction/graph join", prediction_task_ids, graph_task_ids)

    labels_by_task_id = {label["task_id"]: label for label in labels}
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    methods = {prediction["method"] for prediction in predictions}
    if len(methods) != 1:
        raise ContractValidationError(f"Invalid evaluation input: expected one method per file, got methods={sorted(methods)}.")
    method = next(iter(methods)) if methods else ""

    per_task_rows: list[TaskMetricRow] = []
    for prediction in predictions:
        task_id = prediction["task_id"]
        ranked_node_ids = [ranked_node["node_id"] for ranked_node in prediction["ranked_nodes"]]
        label = labels_by_task_id[task_id]
        graph = graphs_by_task_id[task_id]
        gold_nodes = set(label["gold_evidence_nodes"])
        _validate_gold_nodes_exist(task_id, gold_nodes, graph)
        per_task_rows.append(
            {
                "Recall@2": recall_at(ranked_node_ids, gold_nodes, 2),
                "Recall@5": recall_at(ranked_node_ids, gold_nodes, 5),
                "Recall@10": recall_at(ranked_node_ids, gold_nodes, 10),
                "Evidence F1@5": evidence_f1_at(ranked_node_ids, gold_nodes, 5),
                "Evidence F1@10": evidence_f1_at(ranked_node_ids, gold_nodes, 10),
                "Full Support@5": full_support_at(ranked_node_ids, gold_nodes, 5),
                "Full Support@10": full_support_at(ranked_node_ids, gold_nodes, 10),
                "MRR": mrr(ranked_node_ids, gold_nodes),
                "Connected Evidence Recall@5": connected_evidence_at(ranked_node_ids, gold_nodes, graph, 5),
                "Connected Evidence Recall@10": connected_evidence_at(ranked_node_ids, gold_nodes, graph, 10),
                "Query-Evidence Connectivity@10": query_evidence_connectivity_at(ranked_node_ids, gold_nodes, graph, 10),
                "Retrieval Latency / Query": float(prediction["latency_ms"]),
                "Memory Size": float(_memory_node_count(graph)),
                "Avg Retrieved Nodes": float(len(prediction["retrieved_subgraph"]["nodes"])),
                "Avg Retrieved Edges": float(len(prediction["retrieved_subgraph"]["edges"])),
            }
        )

    aggregate_row: MetricRow = {
        "Method": method,
        "Recall@2": 0.0,
        "Recall@5": 0.0,
        "Recall@10": 0.0,
        "Evidence F1@5": 0.0,
        "Evidence F1@10": 0.0,
        "Full Support@5": 0.0,
        "Full Support@10": 0.0,
        "MRR": 0.0,
        "Connected Evidence Recall@5": 0.0,
        "Connected Evidence Recall@10": 0.0,
        "Query-Evidence Connectivity@10": 0.0,
        "Path Recall@10": "N/A",
        "Edge Recall@10": "N/A",
        "Retrieval Latency / Query": 0.0,
        "Index Build Time": 0.0,
        "Graph Construction Time": 0.0,
        "Memory Size": 0.0,
        "Avg Retrieved Nodes": 0.0,
        "Avg Retrieved Edges": 0.0,
    }
    for column in [
        "Recall@2",
        "Recall@5",
        "Recall@10",
        "Evidence F1@5",
        "Evidence F1@10",
        "Full Support@5",
        "Full Support@10",
        "MRR",
        "Connected Evidence Recall@5",
        "Connected Evidence Recall@10",
        "Query-Evidence Connectivity@10",
        "Retrieval Latency / Query",
        "Memory Size",
        "Avg Retrieved Nodes",
        "Avg Retrieved Edges",
    ]:
        aggregate_row[column] = _mean(row[column] for row in per_task_rows)
    return [aggregate_row]


def split_metric_tables(
    rows: list[MetricRow],
) -> tuple[list[MetricTableRow], list[MetricTableRow], list[MetricTableRow]]:
    main_rows = [_select_columns(row, MAIN_RESULT_COLUMNS) for row in rows]
    path_rows = [_select_columns(row, PATH_RESULT_COLUMNS) for row in rows]
    efficiency_rows = [_select_columns(row, EFFICIENCY_RESULT_COLUMNS) for row in rows]
    return main_rows, path_rows, efficiency_rows


def build_failure_cases(
    predictions: list[RankedResult],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    *,
    top_k: int = 10,
    limit: int = 0,
) -> list[FailureCase]:
    if limit <= 0:
        return []
    labels_by_task_id = {label["task_id"]: label for label in labels}
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    cases: list[FailureCase] = []
    for prediction in predictions:
        task_id = prediction["task_id"]
        ranked_node_ids = [ranked_node["node_id"] for ranked_node in prediction["ranked_nodes"]]
        gold_nodes = set(labels_by_task_id[task_id]["gold_evidence_nodes"])
        if full_support_at(ranked_node_ids, gold_nodes, top_k) == 1.0:
            continue
        retrieved_top_k = ranked_node_ids[:top_k]
        cases.append(
            {
                "debug_type": "failure_case",
                "task_id": task_id,
                "method": prediction["method"],
                "failure_type": f"missing_full_support_at_{top_k}",
                "gold_evidence_nodes": sorted(gold_nodes),
                "retrieved_top_k": retrieved_top_k,
                "missing_gold_nodes": sorted(gold_nodes - set(retrieved_top_k)),
                "connected_gold_in_top_k": bool(
                    connected_evidence_at(ranked_node_ids, gold_nodes, graphs_by_task_id[task_id], top_k)
                ),
            }
        )
        if len(cases) >= limit:
            break
    return cases


def _undirected_adjacency(edges: list[GraphEdge], allowed_nodes: set[str]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in allowed_nodes and target in allowed_nodes:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def _directed_adjacency(edges: list[GraphEdge], allowed_nodes: set[str]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in allowed_nodes or target not in allowed_nodes:
            continue
        adjacency[source].add(target)
        if not edge.get("directed", False):
            adjacency[target].add(source)
    return adjacency


def _reachable_from(start_node: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = {start_node}
    queue: deque[str] = deque([start_node])
    while queue:
        node_id = queue.popleft()
        for neighbor in adjacency.get(node_id, set()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _require_gold_nodes(gold_nodes: set[str]) -> None:
    if not gold_nodes:
        raise ContractValidationError("Gold evidence nodes must be non-empty.")


def _validate_gold_nodes_exist(task_id: str, gold_nodes: set[str], graph: MemoryGraph) -> None:
    graph_node_ids = {str(node.get("id")) for node in graph.get("nodes", [])}
    missing = sorted(gold_nodes - graph_node_ids)
    if missing:
        raise ContractValidationError(f"Invalid evaluation input: task_id={task_id} gold nodes missing from graph: {missing}.")


def _memory_node_count(graph: MemoryGraph) -> int:
    return sum(1 for node in graph.get("nodes", []) if node.get("id") != "q")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _select_columns(row: MetricRow, columns: list[str]) -> MetricTableRow:
    return {column: row[column] for column in columns}
