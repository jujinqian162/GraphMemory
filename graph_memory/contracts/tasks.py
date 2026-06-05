from __future__ import annotations

from typing import Literal, TypedDict

from typing_extensions import NotRequired

from graph_memory.contracts.common import NodeId, TaskId


class MemoryItem(TypedDict):
    id: NodeId
    node_type: Literal["document_sentence"]
    text: str
    source: str
    sentence_id: int
    position: int


class MemoryTaskInput(TypedDict):
    task_id: TaskId
    query: str
    memory_items: list[MemoryItem]
    metadata: NotRequired[dict[str, object]]
    debug: NotRequired[dict[str, object]]


class MemoryTaskLabels(TypedDict):
    task_id: TaskId
    gold_answer: str
    gold_evidence_nodes: list[NodeId]
    gold_dependency_edges: list[list[str]]
    metadata: NotRequired[dict[str, object]]
    debug: NotRequired[dict[str, object]]


class CombinedMemoryTask(MemoryTaskInput, MemoryTaskLabels):
    """Compatibility-only artifact shape containing input and label fields."""


__all__ = ["CombinedMemoryTask", "MemoryItem", "MemoryTaskInput", "MemoryTaskLabels"]
