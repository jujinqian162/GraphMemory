from __future__ import annotations

import re
from collections.abc import Sequence

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.datasets.musique.records import (
    ConvertedMuSiQueExample,
    MuSiQueCandidateParagraph,
    MuSiQueConversionResult,
    MuSiQueExample,
    MuSiQueLabelRecord,
    MuSiQueRankingRecord,
)

_STEP_REFERENCE_PATTERN = re.compile(r"#(\d+)")


def convert_musique_examples(examples: Sequence[MuSiQueExample]) -> MuSiQueConversionResult:
    converted_examples = [convert_musique_example(example) for example in examples]
    return MuSiQueConversionResult(
        ranking_records=[converted_example.ranking_record for converted_example in converted_examples],
        label_records=[converted_example.label_record for converted_example in converted_examples],
    )


def convert_musique_example(example: MuSiQueExample) -> ConvertedMuSiQueExample:
    task_id: TaskId = f"musique_{example.raw_id}"
    candidate_paragraphs: list[MuSiQueCandidateParagraph] = []
    paragraph_idx_to_node_id: dict[int, NodeId] = {}
    for position, paragraph in enumerate(example.paragraphs):
        node_id: NodeId = f"p{paragraph.idx}"
        candidate_paragraphs.append(
            MuSiQueCandidateParagraph(
                paragraph_id=node_id,
                title=paragraph.title,
                paragraph_index=paragraph.idx,
                position=position,
                text=paragraph.paragraph_text,
            )
        )
        paragraph_idx_to_node_id[paragraph.idx] = node_id

    gold_evidence_paragraph_ids = [
        paragraph_idx_to_node_id[paragraph.idx]
        for paragraph in example.paragraphs
        if paragraph.is_supporting
    ]
    if not gold_evidence_paragraph_ids:
        raise ValueError(f"MuSiQue example id={example.raw_id} must contain at least one supporting paragraph.")

    gold_dependency_edges, unresolved_reference_count = _dependency_edges(example, paragraph_idx_to_node_id)

    ranking_record = MuSiQueRankingRecord(
        task_id=task_id,
        question=example.question,
        candidate_paragraphs=tuple(candidate_paragraphs),
        metadata={"dataset": "musique", "raw_id": example.raw_id},
    )
    label_record = MuSiQueLabelRecord(
        task_id=task_id,
        gold_answer=example.answer,
        gold_answer_aliases=example.answer_aliases,
        gold_evidence_paragraph_ids=tuple(gold_evidence_paragraph_ids),
        gold_dependency_edges=tuple(
            (edge[0], edge[1]) for edge in gold_dependency_edges
        ),
        metadata={
            "answerable": example.answerable,
            "path_label_source": "question_decomposition",
            "path_supported": bool(gold_dependency_edges),
            "unresolved_decomposition_reference_count": unresolved_reference_count,
        },
    )
    return ConvertedMuSiQueExample(ranking_record=ranking_record, label_record=label_record)


def _dependency_edges(
    example: MuSiQueExample,
    paragraph_idx_to_node_id: dict[int, NodeId],
) -> tuple[list[list[NodeId]], int]:
    step_support_nodes = [
        paragraph_idx_to_node_id.get(step.paragraph_support_idx)
        if step.paragraph_support_idx is not None
        else None
        for step in example.question_decomposition
    ]
    edges: list[list[NodeId]] = []
    seen_edges: set[tuple[NodeId, NodeId]] = set()
    unresolved_reference_count = 0
    for target_step_index, step in enumerate(example.question_decomposition):
        target_node = step_support_nodes[target_step_index]
        if target_node is None:
            continue
        for source_step_index in _referenced_step_indices(step.question):
            if source_step_index < 0 or source_step_index >= len(step_support_nodes):
                unresolved_reference_count += 1
                continue
            source_node = step_support_nodes[source_step_index]
            if source_node is None:
                unresolved_reference_count += 1
                continue
            if source_node == target_node:
                continue
            edge = (source_node, target_node)
            if edge in seen_edges:
                continue
            edges.append([source_node, target_node])
            seen_edges.add(edge)
    return edges, unresolved_reference_count


def _referenced_step_indices(question: str) -> list[int]:
    return [int(match.group(1)) - 1 for match in _STEP_REFERENCE_PATTERN.finditer(question)]
