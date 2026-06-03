from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import FailureCase
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskLabels
from graph_memory.evaluation.connectivity import connected_evidence_at
from graph_memory.evaluation.metrics import full_support_at


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


__all__ = ["build_failure_cases"]
