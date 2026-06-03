from __future__ import annotations

from graph_memory.contracts.common import NodeId
from graph_memory.validation import ContractValidationError


def recall_at(ranked_nodes: list[NodeId], gold_nodes: set[NodeId], k: int) -> float:
    require_gold_nodes(gold_nodes)
    selected = set(ranked_nodes[:k])
    return len(selected & gold_nodes) / len(gold_nodes)


def evidence_f1_at(ranked_nodes: list[NodeId], gold_nodes: set[NodeId], k: int) -> float:
    require_gold_nodes(gold_nodes)
    hits = len(set(ranked_nodes[:k]) & gold_nodes)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(gold_nodes)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def full_support_at(ranked_nodes: list[NodeId], gold_nodes: set[NodeId], k: int) -> float:
    require_gold_nodes(gold_nodes)
    return 1.0 if gold_nodes.issubset(set(ranked_nodes[:k])) else 0.0


def mrr(ranked_nodes: list[NodeId], gold_nodes: set[NodeId]) -> float:
    require_gold_nodes(gold_nodes)
    for index, node_id in enumerate(ranked_nodes, start=1):
        if node_id in gold_nodes:
            return 1.0 / index
    return 0.0


def require_gold_nodes(gold_nodes: set[NodeId]) -> None:
    if not gold_nodes:
        raise ContractValidationError("Gold evidence nodes must be non-empty.")


__all__ = ["evidence_f1_at", "full_support_at", "mrr", "recall_at", "require_gold_nodes"]
