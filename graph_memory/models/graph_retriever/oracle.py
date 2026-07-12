from __future__ import annotations

from dataclasses import dataclass

from graph_memory.evaluation.requests import EvidenceLabel


@dataclass(frozen=True)
class DynamicOracleState:
    valid_next_ids: frozenset[str]
    masked_future_ids: frozenset[str]
    stop_target: bool
    oracle_reachable: bool


@dataclass(frozen=True)
class DynamicEvidenceOracle:
    label: EvidenceLabel
    candidate_node_ids: tuple[str, ...]
    max_steps: int

    def __post_init__(self) -> None:
        task_id = self.label.task_id
        candidates = set(self.candidate_node_ids)
        gold = set(self.label.gold_evidence_item_ids)
        missing = sorted(gold - candidates)
        if missing:
            raise ValueError(f"task_id={task_id} has missing graph nodes: {missing}")
        if len(gold) > self.max_steps:
            raise ValueError(
                f"task_id={task_id} cannot complete {len(gold)} gold nodes within max_steps={self.max_steps}"
            )
        edges = self.label.gold_dependency_edges
        if len(edges) != len(set(edges)):
            raise ValueError(f"task_id={task_id} has duplicate dependency edges")
        invalid_endpoints = sorted(
            {node_id for edge in edges for node_id in edge} - gold
        )
        if invalid_endpoints:
            raise ValueError(
                f"task_id={task_id} dependency edges reference non-gold nodes: {invalid_endpoints}"
            )
        if _has_cycle(gold, edges):
            raise ValueError(f"task_id={task_id} has cyclic gold dependencies")

    def state(self, selected_ids: tuple[str, ...]) -> DynamicOracleState:
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(
                f"task_id={self.label.task_id} selected evidence contains duplicates"
            )
        candidates = set(self.candidate_node_ids)
        unknown = sorted(set(selected_ids) - candidates)
        if unknown:
            raise ValueError(
                f"task_id={self.label.task_id} selected evidence is outside graph nodes: {unknown}"
            )
        gold = set(self.label.gold_evidence_item_ids)
        selected = set(selected_ids)
        predecessors: dict[str, set[str]] = {node_id: set() for node_id in gold}
        for source, target in self.label.gold_dependency_edges:
            predecessors[target].add(source)
        remaining = gold - selected
        valid = frozenset(
            node_id for node_id in remaining if predecessors[node_id].issubset(selected)
        )
        masked = frozenset(remaining - set(valid))
        capacity_reachable = len(selected_ids) + len(remaining) <= self.max_steps
        order_reachable = _selected_order_respects_dependencies(
            selected_ids, self.label.gold_dependency_edges
        )
        return DynamicOracleState(
            valid_next_ids=valid,
            masked_future_ids=masked,
            stop_target=not remaining,
            oracle_reachable=capacity_reachable and order_reachable,
        )


def _has_cycle(nodes: set[str], edges: tuple[tuple[str, str], ...]) -> bool:
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in outgoing[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(nodes)


def _selected_order_respects_dependencies(
    selected_ids: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> bool:
    positions = {node_id: index for index, node_id in enumerate(selected_ids)}
    return all(
        target not in positions
        or (source in positions and positions[source] < positions[target])
        for source, target in edges
    )


__all__ = ["DynamicEvidenceOracle", "DynamicOracleState"]
