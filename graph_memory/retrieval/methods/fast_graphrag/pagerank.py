from __future__ import annotations

from collections.abc import Mapping


def personalized_pagerank(
    adjacency: Mapping[str, Mapping[str, float]],
    personalization: Mapping[str, float],
    *,
    damping: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, float]:
    nodes = _sorted_nodes(adjacency, personalization)
    if not nodes:
        return {}
    restart = _normalized_distribution(nodes, personalization)
    scores = dict(restart)
    transitions = _normalized_transitions(nodes, adjacency)

    for _ in range(max_iterations):
        next_scores = {
            node: (1.0 - damping) * restart[node]
            for node in nodes
        }
        dangling_score = sum(scores[node] for node in nodes if not transitions[node])
        for target in nodes:
            next_scores[target] += damping * dangling_score * restart[target]
        for source in nodes:
            for target, weight in transitions[source].items():
                next_scores[target] += damping * scores[source] * weight
        delta = sum(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if delta < tolerance:
            break

    return _renormalized(scores)


def _sorted_nodes(
    adjacency: Mapping[str, Mapping[str, float]],
    personalization: Mapping[str, float],
) -> tuple[str, ...]:
    nodes = set(adjacency)
    nodes.update(personalization)
    for targets in adjacency.values():
        nodes.update(targets)
    return tuple(sorted(nodes))


def _normalized_distribution(nodes: tuple[str, ...], weights: Mapping[str, float]) -> dict[str, float]:
    positive_weights = {
        node: max(float(weights.get(node, 0.0)), 0.0)
        for node in nodes
    }
    total = sum(positive_weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(nodes)
        return {node: uniform for node in nodes}
    return {node: positive_weights[node] / total for node in nodes}


def _normalized_transitions(
    nodes: tuple[str, ...],
    adjacency: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    node_set = set(nodes)
    transitions: dict[str, dict[str, float]] = {}
    for source in nodes:
        raw_targets = {
            target: max(float(weight), 0.0)
            for target, weight in adjacency.get(source, {}).items()
            if target in node_set
        }
        total = sum(raw_targets.values())
        transitions[source] = (
            {target: weight / total for target, weight in sorted(raw_targets.items())}
            if total > 0.0
            else {}
        )
    return transitions


def _renormalized(scores: Mapping[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0.0:
        return dict(scores)
    return {node: score / total for node, score in scores.items()}


__all__ = ["personalized_pagerank"]
