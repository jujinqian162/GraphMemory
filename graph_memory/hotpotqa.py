from __future__ import annotations

from typing import Any


def convert_hotpotqa_examples(
    examples: list[dict],
    max_examples: int | None = None,
) -> tuple[list[dict], list[dict]]:
    if max_examples is not None:
        if max_examples < 0:
            raise ValueError("max_examples must be non-negative.")
        examples = examples[:max_examples]

    task_inputs: list[dict] = []
    task_labels: list[dict] = []
    for raw_example in examples:
        task_input, task_label = _convert_one_example(raw_example)
        task_inputs.append(task_input)
        task_labels.append(task_label)
    return task_inputs, task_labels


def combined_memory_tasks(task_inputs: list[dict], task_labels: list[dict]) -> list[dict]:
    labels_by_task_id = {label["task_id"]: label for label in task_labels}
    combined: list[dict] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        if task_id not in labels_by_task_id:
            raise ValueError(f"Cannot combine task_id={task_id}: matching labels are missing.")
        combined.append({**task_input, **labels_by_task_id[task_id]})
    return combined


def _convert_one_example(raw_example: dict[str, Any]) -> tuple[dict, dict]:
    raw_id = raw_example.get("_id")
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError("HotpotQA example must contain a non-empty _id.")

    question = raw_example.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError(f"HotpotQA example _id={raw_id} must contain a non-empty question.")

    answer = raw_example.get("answer")
    if not isinstance(answer, str):
        raise ValueError(f"HotpotQA example _id={raw_id} must contain a string answer.")

    context = raw_example.get("context")
    if not isinstance(context, list):
        raise ValueError(f"HotpotQA example _id={raw_id} must contain a context list.")

    task_id = f"hotpot_{raw_id}"
    memory_items: list[dict] = []
    title_sentence_to_node_id: dict[tuple[str, int], str] = {}
    position = 0

    for document in context:
        if not isinstance(document, list | tuple) or len(document) != 2:
            raise ValueError(f"HotpotQA example _id={raw_id} has invalid context document.")
        title, sentences = document
        if not isinstance(title, str) or not isinstance(sentences, list):
            raise ValueError(f"HotpotQA example _id={raw_id} has invalid context title or sentences.")
        for sentence_id, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise ValueError(f"HotpotQA example _id={raw_id} title={title} sentence_id={sentence_id} is not text.")
            node_id = f"m{position}"
            memory_items.append(
                {
                    "id": node_id,
                    "node_type": "document_sentence",
                    "text": sentence,
                    "source": title,
                    "sentence_id": sentence_id,
                    "position": position,
                }
            )
            title_sentence_to_node_id[(title, sentence_id)] = node_id
            position += 1

    if not memory_items:
        raise ValueError(f"HotpotQA example _id={raw_id} contains no memory sentences.")

    supporting_facts = raw_example.get("supporting_facts")
    if not isinstance(supporting_facts, list):
        raise ValueError(f"HotpotQA example _id={raw_id} must contain supporting_facts.")

    gold_evidence_nodes: list[str] = []
    for supporting_fact in supporting_facts:
        if not isinstance(supporting_fact, list | tuple) or len(supporting_fact) != 2:
            raise ValueError(f"HotpotQA example _id={raw_id} has invalid supporting fact.")
        title, sentence_id = supporting_fact
        if not isinstance(title, str) or not isinstance(sentence_id, int):
            raise ValueError(f"HotpotQA example _id={raw_id} has invalid supporting fact fields.")
        node_id = title_sentence_to_node_id.get((title, sentence_id))
        if node_id is None:
            raise ValueError(
                f"HotpotQA example _id={raw_id} supporting fact ({title}, {sentence_id}) cannot map to a memory node."
            )
        if node_id not in gold_evidence_nodes:
            gold_evidence_nodes.append(node_id)

    if not gold_evidence_nodes:
        raise ValueError(f"HotpotQA example _id={raw_id} must contain at least one supporting fact.")

    return (
        {
            "task_id": task_id,
            "query": question,
            "memory_items": memory_items,
        },
        {
            "task_id": task_id,
            "gold_answer": answer,
            "gold_evidence_nodes": gold_evidence_nodes,
            "gold_dependency_edges": [],
        },
    )
