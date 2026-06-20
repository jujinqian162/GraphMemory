from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import NodeId
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.requests import GraphBuildNode, GraphBuildRequest
from graph_memory.text.entities import extract_entities, title_aliases
from graph_memory.text.lexical import compute_idf


@dataclass(frozen=True)
class PreparedGraphInput:
    request: GraphBuildRequest
    documents: list[str]
    idf: dict[str, float]
    entities_by_node_id: dict[NodeId, set[str]]


def prepare_graph_input(request: GraphBuildRequest, config: GraphBuildConfig) -> PreparedGraphInput:
    documents = [_node_document_text(node) for node in request.nodes]
    return PreparedGraphInput(
        request=request,
        documents=documents,
        idf=compute_idf([request.query_text, *documents]),
        entities_by_node_id=_entities_by_node(request.nodes, config=config),
    )


def _node_document_text(node: GraphBuildNode) -> str:
    if node.source_ref:
        return f"{node.source_ref}. {node.text}"
    return node.text


def _entities_by_node(nodes: Sequence[GraphBuildNode], config: GraphBuildConfig) -> dict[NodeId, set[str]]:
    entities_by_node_id: dict[NodeId, set[str]] = {}
    for node in nodes:
        entities = extract_entities(_node_document_text(node), use_spacy=config.use_spacy)
        if node.source_ref:
            entities.update(title_aliases(node.source_ref))
        entities_by_node_id[node.node_id] = entities
    return entities_by_node_id
