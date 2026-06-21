from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.datasets.twowiki.records import (
    ConvertedTwoWikiExample,
    TwoWikiCandidateSentence,
    TwoWikiConversionResult,
    TwoWikiEvidenceTriple,
    TwoWikiExample,
    TwoWikiLabelRecord,
    TwoWikiRankingRecord,
)


@dataclass(frozen=True)
class _SupportNode:
    node_id: NodeId
    title: str
    text: str
    support_order: int


@dataclass(frozen=True)
class _MappedEvidence:
    node_id: NodeId | None
    ambiguity_count: int


def convert_twowiki_examples(examples: Sequence[TwoWikiExample]) -> TwoWikiConversionResult:
    converted_examples = [convert_twowiki_example(example) for example in examples]
    return TwoWikiConversionResult(
        ranking_records=[converted_example.ranking_record for converted_example in converted_examples],
        label_records=[converted_example.label_record for converted_example in converted_examples],
    )


def convert_twowiki_example(example: TwoWikiExample) -> ConvertedTwoWikiExample:
    task_id: TaskId = f"2wiki_{example.raw_id}"
    candidate_sentences: list[TwoWikiCandidateSentence] = []
    title_sentence_to_node_id: dict[tuple[str, int], NodeId] = {}
    sentence_text_by_node_id: dict[NodeId, str] = {}

    position = 0
    for document in example.documents:
        for sentence_index, sentence in enumerate(document.sentences):
            sentence_id_from_position: NodeId = f"m{position}"
            candidate_sentence: TwoWikiCandidateSentence = {
                "sentence_id": sentence_id_from_position,
                "title": document.title,
                "sentence_index": sentence_index,
                "position": position,
                "text": sentence,
            }
            candidate_sentences.append(candidate_sentence)
            title_sentence_to_node_id[(document.title, sentence_index)] = sentence_id_from_position
            sentence_text_by_node_id[sentence_id_from_position] = sentence
            position += 1

    if not candidate_sentences:
        raise ValueError(f"2Wiki example _id={example.raw_id} contains no candidate sentences.")

    gold_evidence_sentence_ids: list[NodeId] = []
    support_nodes: list[_SupportNode] = []
    for supporting_fact in example.supporting_facts:
        node_id = title_sentence_to_node_id.get((supporting_fact.title, supporting_fact.sentence_id))
        if node_id is None:
            raise ValueError(
                "2Wiki example "
                f"_id={example.raw_id} supporting fact ({supporting_fact.title}, {supporting_fact.sentence_id}) "
                "cannot map to a candidate sentence."
            )
        if node_id not in gold_evidence_sentence_ids:
            gold_evidence_sentence_ids.append(node_id)
            support_nodes.append(
                _SupportNode(
                    node_id=node_id,
                    title=supporting_fact.title,
                    text=sentence_text_by_node_id[node_id],
                    support_order=len(support_nodes),
                )
            )

    if not gold_evidence_sentence_ids:
        raise ValueError(f"2Wiki example _id={example.raw_id} must contain at least one supporting fact.")

    path_label_source, path_label_triples = _path_label_triples(example)
    mapped_evidences = [_map_evidence_to_support_node(evidence, support_nodes) for evidence in path_label_triples]
    gold_dependency_edges = _dependency_edges(path_label_triples, mapped_evidences)
    mapping_ambiguity_count = sum(mapped.ambiguity_count for mapped in mapped_evidences)

    ranking_record: TwoWikiRankingRecord = {
        "task_id": task_id,
        "question": example.question,
        "question_type": example.question_type,
        "candidate_sentences": candidate_sentences,
        "metadata": {"dataset": "2wiki", "raw_id": example.raw_id},
    }
    label_record: TwoWikiLabelRecord = {
        "task_id": task_id,
        "gold_answer": example.answer,
        "gold_evidence_sentence_ids": gold_evidence_sentence_ids,
        "gold_dependency_edges": gold_dependency_edges,
        "metadata": {
            "question_type": example.question_type,
            "path_label_source": path_label_source,
            "path_supported": bool(gold_dependency_edges),
            "mapping_ambiguity_count": mapping_ambiguity_count,
        },
    }
    return ConvertedTwoWikiExample(ranking_record=ranking_record, label_record=label_record)


def _path_label_triples(example: TwoWikiExample) -> tuple[str, tuple[TwoWikiEvidenceTriple, ...]]:
    if example.evidences_id:
        return "evidences_id", example.evidences_id
    return "evidences", example.evidences


def _dependency_edges(
    evidences: Sequence[TwoWikiEvidenceTriple],
    mapped_evidences: Sequence[_MappedEvidence],
) -> list[list[NodeId]]:
    edges: list[list[NodeId]] = []
    seen_edges: set[tuple[NodeId, NodeId]] = set()
    for index, current in enumerate(evidences[:-1]):
        following = evidences[index + 1]
        if _normalize(current.object) != _normalize(following.subject):
            continue
        source = mapped_evidences[index].node_id
        target = mapped_evidences[index + 1].node_id
        if source is None or target is None or source == target:
            continue
        edge = (source, target)
        if edge in seen_edges:
            continue
        edges.append([source, target])
        seen_edges.add(edge)
    return edges


def _map_evidence_to_support_node(evidence: TwoWikiEvidenceTriple, support_nodes: Sequence[_SupportNode]) -> _MappedEvidence:
    scored_nodes = [
        (score, support_node.support_order, support_node.node_id)
        for support_node in support_nodes
        if (score := _evidence_support_score(evidence, support_node)) > 0
    ]
    if not scored_nodes:
        return _MappedEvidence(node_id=None, ambiguity_count=0)
    best_score = max(score for score, _, _ in scored_nodes)
    best_nodes = [(order, node_id) for score, order, node_id in scored_nodes if score == best_score]
    best_nodes.sort()
    return _MappedEvidence(
        node_id=best_nodes[0][1],
        ambiguity_count=max(0, len(best_nodes) - 1),
    )


def _evidence_support_score(evidence: TwoWikiEvidenceTriple, support_node: _SupportNode) -> int:
    title = _normalize(support_node.title)
    text = _normalize(support_node.text)
    subject = _normalize(evidence.subject)
    relation = _normalize(evidence.relation)
    object_value = _normalize(evidence.object)
    score = 0
    if subject and subject in title:
        score += 8
    if subject and subject in text:
        score += 4
    if object_value and object_value in title:
        score += 2
    if object_value and object_value in text:
        score += 2
    if relation and relation in text:
        score += 1
    return score


def _normalize(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.casefold()).strip()
    return normalized
