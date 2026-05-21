from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from graph_memory.entities import extract_entities, title_aliases
from graph_memory.text import compute_idf, lexical_score
from graph_memory.types import EdgeType, GraphBuildConfig, GraphEdge, GraphNode, MemoryGraph, MemoryItem, MemoryTaskInput, NodeId


def build_graph(task_input: MemoryTaskInput, config: GraphBuildConfig) -> MemoryGraph:
    memory_items: list[MemoryItem] = task_input["memory_items"]
    nodes: list[GraphNode] = [{"id": "q", "node_type": "question", "text": task_input["query"]}, *memory_items]
    documents = [f'{item["source"]}. {item["text"]}' for item in memory_items]
    idf = compute_idf([task_input["query"], *documents])
    entities_by_node_id = _entities_by_node(memory_items, config=config)

    edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    _add_sequential_edges(edges, seen_edges, memory_items)
    _add_query_overlap_edges(edges, seen_edges, task_input, idf, entities_by_node_id, config)
    _add_entity_overlap_edges(edges, seen_edges, memory_items, entities_by_node_id, config)
    _add_bridge_edges(edges, seen_edges, memory_items, entities_by_node_id, config)

    return {
        "task_id": task_input["task_id"],
        "nodes": nodes,
        "edges": edges,
    }


def build_graphs(task_inputs: list[MemoryTaskInput], config: GraphBuildConfig) -> list[MemoryGraph]:
    return [build_graph(task_input, config) for task_input in task_inputs]


def _add_sequential_edges(
    edges: list[GraphEdge], seen_edges: set[tuple[str, str, str]], memory_items: list[MemoryItem]
) -> None:
    items_by_source: dict[str, list[MemoryItem]] = defaultdict(list)
    for item in memory_items:
        items_by_source[item["source"]].append(item)

    for source_items in items_by_source.values():
        ordered_items = sorted(source_items, key=lambda item: item["sentence_id"])
        for left, right in zip(ordered_items, ordered_items[1:]):
            if right["sentence_id"] - left["sentence_id"] == 1:
                _append_edge(edges, seen_edges, left["id"], right["id"], "sequential", 1.0, directed=False)


def _add_query_overlap_edges(
    edges: list[GraphEdge],
    seen_edges: set[tuple[str, str, str]],
    task_input: MemoryTaskInput,
    idf: dict[str, float],
    entities_by_node_id: dict[NodeId, set[str]],
    config: GraphBuildConfig,
) -> None:
    query_entities = extract_entities(task_input["query"], use_spacy=config.use_spacy)
    scored_targets: list[tuple[float, str]] = []
    for item in task_input["memory_items"]:
        passage = f'{item["source"]}. {item["text"]}'
        score = lexical_score(
            task_input["query"],
            passage,
            idf,
            title_aliases=title_aliases(item["source"]),
            query_entities=query_entities,
            passage_entities=entities_by_node_id[item["id"]],
        )
        if score > 0.0:
            scored_targets.append((score, item["id"]))

    for score, node_id in sorted(scored_targets, key=lambda scored: (-scored[0], scored[1]))[: config.max_query_overlap]:
        _append_edge(edges, seen_edges, "q", node_id, "query_overlap", score, directed=True)


def _add_entity_overlap_edges(
    edges: list[GraphEdge],
    seen_edges: set[tuple[str, str, str]],
    memory_items: list[MemoryItem],
    entities_by_node_id: dict[NodeId, set[str]],
    config: GraphBuildConfig,
) -> None:
    candidates: list[tuple[float, str, str]] = []
    for left, right in combinations(memory_items, 2):
        score = float(len(entities_by_node_id[left["id"]] & entities_by_node_id[right["id"]]))
        if score > 0.0:
            candidates.append((score, left["id"], right["id"]))

    neighbor_counts: dict[str, int] = defaultdict(int)
    for score, source, target in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2])):
        if neighbor_counts[source] >= config.max_entity_neighbors or neighbor_counts[target] >= config.max_entity_neighbors:
            continue
        _append_edge(edges, seen_edges, source, target, "entity_overlap", score, directed=False)
        neighbor_counts[source] += 1
        neighbor_counts[target] += 1


def _add_bridge_edges(
    edges: list[GraphEdge],
    seen_edges: set[tuple[str, str, str]],
    memory_items: list[MemoryItem],
    entities_by_node_id: dict[NodeId, set[str]],
    config: GraphBuildConfig,
) -> None:
    candidates: list[tuple[float, str, str]] = []
    for left, right in combinations(memory_items, 2):
        if left["source"] == right["source"]:
            continue
        shared_entities = entities_by_node_id[left["id"]] & entities_by_node_id[right["id"]]
        cross_title_mentions = _title_mention_score(left, right) + _title_mention_score(right, left)
        score = float(len(shared_entities)) + cross_title_mentions
        if score > 0.0:
            candidates.append((score, left["id"], right["id"]))

    for score, source, target in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))[
        : config.max_bridge_edges
    ]:
        _append_edge(edges, seen_edges, source, target, "bridge", score, directed=False)


def _entities_by_node(memory_items: list[MemoryItem], config: GraphBuildConfig) -> dict[NodeId, set[str]]:
    entities_by_node_id: dict[NodeId, set[str]] = {}
    for item in memory_items:
        entities = extract_entities(f'{item["source"]}. {item["text"]}', use_spacy=config.use_spacy)
        entities.update(title_aliases(item["source"]))
        entities_by_node_id[item["id"]] = entities
    return entities_by_node_id


def _title_mention_score(left: MemoryItem, right: MemoryItem) -> float:
    right_text = right["text"].lower()
    return float(sum(1 for alias in title_aliases(left["source"]) if alias and alias in right_text))


def _append_edge(
    edges: list[GraphEdge],
    seen_edges: set[tuple[str, str, str]],
    source: str,
    target: str,
    edge_type: EdgeType,
    weight: float,
    *,
    directed: bool,
) -> None:
    edge_key = _edge_key(source, target, edge_type, directed=directed)
    if edge_key in seen_edges:
        return
    seen_edges.add(edge_key)
    edges.append(
        {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "weight": weight,
            "directed": directed,
        }
    )


def _edge_key(source: str, target: str, edge_type: str, *, directed: bool) -> tuple[str, str, str]:
    if directed:
        return source, target, edge_type
    left, right = sorted([source, target])
    return left, right, edge_type
