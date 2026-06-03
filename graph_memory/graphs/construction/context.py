from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.tasks import MemoryItem, MemoryTaskInput
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.text.entities import extract_entities, title_aliases
from graph_memory.text.lexical import compute_idf


@dataclass(frozen=True)
class PreparedGraphInput:
    task_input: MemoryTaskInput
    documents: list[str]
    idf: dict[str, float]
    entities_by_node_id: dict[NodeId, set[str]]


def prepare_graph_input(task_input: MemoryTaskInput, config: GraphBuildConfig) -> PreparedGraphInput:
    memory_items: list[MemoryItem] = task_input["memory_items"]
    documents = [f'{item["source"]}. {item["text"]}' for item in memory_items]
    return PreparedGraphInput(
        task_input=task_input,
        documents=documents,
        idf=compute_idf([task_input["query"], *documents]),
        entities_by_node_id=_entities_by_node(memory_items, config=config),
    )


def _entities_by_node(memory_items: list[MemoryItem], config: GraphBuildConfig) -> dict[NodeId, set[str]]:
    entities_by_node_id: dict[NodeId, set[str]] = {}
    for item in memory_items:
        entities = extract_entities(f'{item["source"]}. {item["text"]}', use_spacy=config.use_spacy)
        entities.update(title_aliases(item["source"]))
        entities_by_node_id[item["id"]] = entities
    return entities_by_node_id

