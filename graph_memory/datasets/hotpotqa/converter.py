from __future__ import annotations

from collections.abc import Sequence

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.contracts.tasks import MemoryItem, MemoryTaskInput, MemoryTaskLabels
from graph_memory.datasets.hotpotqa.records import (
    ConvertedHotpotQAExample,
    HotpotQAConversionResult,
    HotpotQAExample,
)


def convert_hotpotqa_examples(examples: Sequence[HotpotQAExample]) -> HotpotQAConversionResult:
    converted_examples = [convert_hotpotqa_example(example) for example in examples]
    return HotpotQAConversionResult(
        task_inputs=[converted_example.task_input for converted_example in converted_examples],
        task_labels=[converted_example.task_labels for converted_example in converted_examples],
    )


def convert_hotpotqa_example(example: HotpotQAExample) -> ConvertedHotpotQAExample:
    task_id: TaskId = f"hotpot_{example.raw_id}"
    memory_items: list[MemoryItem] = []
    title_sentence_to_node_id: dict[tuple[str, int], NodeId] = {}

    position = 0
    for document in example.documents:
        for sentence_id, sentence in enumerate(document.sentences):
            node_id_from_position: NodeId = f"m{position}"
            memory_item: MemoryItem = {
                "id": node_id_from_position,
                "node_type": "document_sentence",
                "text": sentence,
                "source": document.title,
                "sentence_id": sentence_id,
                "position": position,
            }
            memory_items.append(memory_item)
            title_sentence_to_node_id[(document.title, sentence_id)] = node_id_from_position
            position += 1

    if not memory_items:
        raise ValueError(f"HotpotQA example _id={example.raw_id} contains no memory sentences.")

    gold_evidence_nodes: list[NodeId] = []
    for supporting_fact in example.supporting_facts:
        node_id = title_sentence_to_node_id.get((supporting_fact.title, supporting_fact.sentence_id))
        if node_id is None:
            raise ValueError(
                "HotpotQA example "
                f"_id={example.raw_id} supporting fact ({supporting_fact.title}, {supporting_fact.sentence_id}) "
                "cannot map to a memory node."
            )
        if node_id not in gold_evidence_nodes:
            gold_evidence_nodes.append(node_id)

    if not gold_evidence_nodes:
        raise ValueError(f"HotpotQA example _id={example.raw_id} must contain at least one supporting fact.")

    task_input: MemoryTaskInput = {
        "task_id": task_id,
        "query": example.question,
        "memory_items": memory_items,
    }
    task_labels: MemoryTaskLabels = {
        "task_id": task_id,
        "gold_answer": example.answer,
        "gold_evidence_nodes": gold_evidence_nodes,
        "gold_dependency_edges": [],
    }
    return ConvertedHotpotQAExample(task_input=task_input, task_labels=task_labels)
