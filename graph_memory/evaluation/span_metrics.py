from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graph_memory.retrieval.results import (
    RankedNodeRecord,
    RetrievedSubgraph,
)
from graph_memory.trajectories import SourceSpan

SpanKey = tuple[str, str]
Interval = tuple[int, int]


@dataclass(frozen=True)
class SpanRetrievalMetrics:
    coverage: float
    density: float
    full_support: float
    f1: float


def span_metrics_at(
    ranked_nodes: tuple[RankedNodeRecord, ...],
    gold_spans: tuple[SourceSpan, ...],
    k: int,
) -> SpanRetrievalMetrics:
    if k <= 0:
        return SpanRetrievalMetrics(0.0, 0.0, 0.0, 0.0)
    return _span_metrics(ranked_nodes[:k], gold_spans)


def span_metrics_under_token_budget(
    ranked_nodes: tuple[RankedNodeRecord, ...],
    gold_spans: tuple[SourceSpan, ...],
    token_budget: int,
) -> SpanRetrievalMetrics:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    selected: list[RankedNodeRecord] = []
    used_tokens = 0
    for node in ranked_nodes:
        if node.token_count <= 0:
            raise ValueError(
                f"ranked node={node.node_id} requires a positive token_count "
                "for token-budget evaluation"
            )
        if used_tokens + node.token_count > token_budget:
            break
        selected.append(node)
        used_tokens += node.token_count
    return _span_metrics(tuple(selected), gold_spans)


def _span_metrics(
    selected: tuple[RankedNodeRecord, ...],
    gold_spans: tuple[SourceSpan, ...],
) -> SpanRetrievalMetrics:
    gold = _group_intervals(gold_spans)
    retrieved = _group_intervals(
        tuple(span for node in selected for span in node.source_spans)
    )
    gold_length = _group_length(gold)
    if gold_length <= 0:
        raise ValueError("gold evidence spans must cover at least one character")
    covered = _intersection_length(gold, retrieved)
    retrieved_cost = sum(
        _span_length(span) for node in selected for span in node.source_spans
    )
    coverage = covered / gold_length
    density = covered / retrieved_cost if retrieved_cost else 0.0
    full_support = 1.0 if covered == gold_length else 0.0
    f1 = 2.0 * coverage * density / (coverage + density) if coverage + density else 0.0
    return SpanRetrievalMetrics(coverage, density, full_support, f1)


def connected_span_coverage_at(
    ranked_nodes: tuple[RankedNodeRecord, ...],
    gold_spans: tuple[SourceSpan, ...],
    subgraph: RetrievedSubgraph,
    k: int,
) -> float:
    selected = ranked_nodes[:k]
    if not selected:
        return 0.0
    selected_by_id = {node.node_id: node for node in selected}
    components = _components(tuple(selected_by_id), subgraph)
    gold = _group_intervals(gold_spans)
    gold_length = _group_length(gold)
    return max(
        _intersection_length(
            gold,
            _group_intervals(
                tuple(
                    span
                    for node_id in component
                    for span in selected_by_id[node_id].source_spans
                )
            ),
        )
        / gold_length
        for component in components
    )


def _components(
    node_ids: tuple[str, ...], subgraph: RetrievedSubgraph
) -> tuple[frozenset[str], ...]:
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in subgraph.edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    components: list[frozenset[str]] = []
    unseen = set(node_ids)
    while unseen:
        root = min(unseen)
        pending = [root]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        unseen -= component
        components.append(frozenset(component))
    return tuple(components)


def span_mrr(
    ranked_nodes: tuple[RankedNodeRecord, ...],
    gold_spans: tuple[SourceSpan, ...],
) -> float:
    gold = _group_intervals(gold_spans)
    for rank, node in enumerate(ranked_nodes, start=1):
        if _intersection_length(gold, _group_intervals(node.source_spans)) > 0:
            return 1.0 / rank
    return 0.0


def missing_gold_spans(
    ranked_nodes: tuple[RankedNodeRecord, ...],
    gold_spans: tuple[SourceSpan, ...],
    k: int,
) -> tuple[SourceSpan, ...]:
    retrieved = _group_intervals(
        tuple(span for node in ranked_nodes[:k] for span in node.source_spans)
    )
    return tuple(
        span
        for span in gold_spans
        if _intersection_length(
            _group_intervals((span,)),
            retrieved,
        )
        < _span_length(span)
    )


def _span_key(span: SourceSpan) -> SpanKey:
    if span.json_pointer is None:
        raise ValueError("span metrics require json_pointer")
    return span.event_id, span.json_pointer


def _span_length(span: SourceSpan) -> int:
    if span.char_start is None or span.char_end is None:
        raise ValueError("span metrics require exact character offsets")
    return span.char_end - span.char_start


def _group_intervals(
    spans: tuple[SourceSpan, ...],
) -> dict[SpanKey, tuple[Interval, ...]]:
    grouped: dict[SpanKey, list[Interval]] = defaultdict(list)
    for span in spans:
        if span.char_start is None or span.char_end is None:
            raise ValueError("span metrics require exact character offsets")
        grouped[_span_key(span)].append((span.char_start, span.char_end))
    return {key: _merge_intervals(intervals) for key, intervals in grouped.items()}


def _merge_intervals(intervals: list[Interval]) -> tuple[Interval, ...]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _group_length(grouped: dict[SpanKey, tuple[Interval, ...]]) -> int:
    return sum(
        end - start for intervals in grouped.values() for start, end in intervals
    )


def _intersection_length(
    left: dict[SpanKey, tuple[Interval, ...]],
    right: dict[SpanKey, tuple[Interval, ...]],
) -> int:
    total = 0
    for key, left_intervals in left.items():
        right_intervals = right.get(key, ())
        left_index = 0
        right_index = 0
        while left_index < len(left_intervals) and right_index < len(right_intervals):
            left_start, left_end = left_intervals[left_index]
            right_start, right_end = right_intervals[right_index]
            total += max(0, min(left_end, right_end) - max(left_start, right_start))
            if left_end <= right_end:
                left_index += 1
            else:
                right_index += 1
    return total


__all__ = [
    "SpanRetrievalMetrics",
    "connected_span_coverage_at",
    "missing_gold_spans",
    "span_metrics_at",
    "span_metrics_under_token_budget",
    "span_mrr",
]
