from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import cast

from tqdm.auto import tqdm

from graph_memory.contracts.common import EdgeType
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.contracts import (
    EvidenceGraph,
    GraphEdge,
    GraphItemNode,
    GraphNode,
    QuestionNode,
)
from graph_memory.graphs.requests import (
    EvidenceGraphBuildNode,
    EvidenceGraphBuildRequest,
)
from graph_memory.text.entities import extract_entities, title_aliases
from graph_memory.text.lexical import compute_idf, lexical_score


def build_graphs(
    requests: list[EvidenceGraphBuildRequest],
    config: GraphBuildConfig,
    *,
    progress_desc: str | None = None,
) -> list[EvidenceGraph]:
    iterator = requests
    if progress_desc is not None:
        iterator = tqdm(requests, desc=progress_desc, unit="graph")
    return [_build_graph(request, config) for request in iterator]


def _build_graph(
    request: EvidenceGraphBuildRequest,
    config: GraphBuildConfig,
) -> EvidenceGraph:
    documents = [_node_document_text(node) for node in request.nodes]
    idf = compute_idf([request.query_text, *documents])
    entities_by_node_id = {
        node.node_id: _node_entities(node, config) for node in request.nodes
    }
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()

    def add(
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float,
        *,
        directed: bool,
    ) -> None:
        left, right = (source, target) if directed else tuple(sorted((source, target)))
        key = (left, right, edge_type)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                weight=weight,
                directed=directed,
            )
        )

    for edge in request.input_visible_edges:
        add(
            edge.source,
            edge.target,
            cast(EdgeType, edge.edge_type),
            edge.weight,
            directed=edge.directed,
        )
    _add_sequential_edges(request.nodes, add)
    _add_query_overlap_edges(request, config, idf, entities_by_node_id, add)
    _add_entity_overlap_edges(request.nodes, config, entities_by_node_id, add)
    _add_bridge_edges(request.nodes, config, entities_by_node_id, add)

    nodes: list[GraphNode] = [
        QuestionNode(text=request.query_text),
        *[_graph_item_node(node) for node in request.nodes],
    ]
    return EvidenceGraph(task_id=request.task_id, nodes=tuple(nodes), edges=tuple(edges))


def _add_sequential_edges(nodes, add) -> None:
    items_by_group: dict[str, list[EvidenceGraphBuildNode]] = defaultdict(list)
    for node in nodes:
        group_key = node.group_key or node.source_ref
        if group_key is not None:
            items_by_group[group_key].append(node)
    for grouped_items in items_by_group.values():
        ordered = sorted(grouped_items, key=lambda node: node.sequence_index or 0)
        for left, right in zip(ordered, ordered[1:]):
            if (
                left.sequence_index is not None
                and right.sequence_index is not None
                and right.sequence_index - left.sequence_index == 1
            ):
                add(left.node_id, right.node_id, "sequential", 1.0, directed=False)


def _add_query_overlap_edges(request, config, idf, entities_by_node_id, add) -> None:
    query_entities = extract_entities(request.query_text, use_spacy=config.use_spacy)
    scored_targets: list[tuple[float, str]] = []
    for node in request.nodes:
        score = lexical_score(
            request.query_text,
            _node_document_text(node),
            idf,
            title_aliases=title_aliases(node.source_ref or ""),
            query_entities=query_entities,
            passage_entities=entities_by_node_id[node.node_id],
        )
        if score > 0.0:
            scored_targets.append((score, node.node_id))
    for score, node_id in sorted(scored_targets, key=lambda item: (-item[0], item[1]))[
        : config.max_query_overlap
    ]:
        add("q", node_id, "query_overlap", score, directed=True)


def _add_entity_overlap_edges(nodes, config, entities_by_node_id, add) -> None:
    candidates: list[tuple[float, str, str]] = []
    for left, right in combinations(nodes, 2):
        score = float(
            len(
                entities_by_node_id[left.node_id]
                & entities_by_node_id[right.node_id]
            )
        )
        if score > 0.0:
            candidates.append((score, left.node_id, right.node_id))
    neighbor_counts: dict[str, int] = defaultdict(int)
    for score, source, target in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if (
            neighbor_counts[source] >= config.max_entity_neighbors
            or neighbor_counts[target] >= config.max_entity_neighbors
        ):
            continue
        add(source, target, "entity_overlap", score, directed=False)
        neighbor_counts[source] += 1
        neighbor_counts[target] += 1


def _add_bridge_edges(nodes, config, entities_by_node_id, add) -> None:
    candidates: list[tuple[float, str, str]] = []
    for left, right in combinations(nodes, 2):
        if _source_group(left) == _source_group(right):
            continue
        shared = entities_by_node_id[left.node_id] & entities_by_node_id[right.node_id]
        score = (
            float(len(shared))
            + _title_mention_score(left, right)
            + _title_mention_score(right, left)
        )
        if score > 0.0:
            candidates.append((score, left.node_id, right.node_id))
    for score, source, target in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )[: config.max_bridge_edges]:
        add(source, target, "bridge", score, directed=False)


def _node_entities(
    node: EvidenceGraphBuildNode,
    config: GraphBuildConfig,
) -> set[str]:
    entities = extract_entities(_node_document_text(node), use_spacy=config.use_spacy)
    if node.source_ref:
        entities.update(title_aliases(node.source_ref))
    return entities


def _node_document_text(node: EvidenceGraphBuildNode) -> str:
    return f"{node.source_ref}. {node.text}" if node.source_ref else node.text


def _title_mention_score(
    left: EvidenceGraphBuildNode,
    right: EvidenceGraphBuildNode,
) -> float:
    if left.source_ref is None:
        return 0.0
    right_text = right.text.lower()
    return float(
        sum(
            1
            for alias in title_aliases(left.source_ref)
            if alias and alias in right_text
        )
    )


def _source_group(node: EvidenceGraphBuildNode) -> str | None:
    return node.group_key or node.source_ref


def _graph_item_node(node: EvidenceGraphBuildNode) -> GraphItemNode:
    return GraphItemNode(
        id=node.node_id,
        node_kind=node.node_kind,
        text=node.text,
        source_ref=node.source_ref,
        group_key=node.group_key,
        sequence_index=node.sequence_index,
        metadata=dict(node.metadata) or None,
    )
