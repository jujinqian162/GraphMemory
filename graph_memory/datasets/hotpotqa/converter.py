from __future__ import annotations

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.datasets.hotpotqa.records import (
    HotpotQACandidateSentence,
    HotpotQAExample,
    HotpotQALabelRecord,
    HotpotQARankingRecord,
)


def convert_hotpotqa_example(
    example: HotpotQAExample,
) -> tuple[HotpotQARankingRecord, HotpotQALabelRecord]:
    task_id: TaskId = f"hotpot_{example.raw_id}"
    candidate_sentences: list[HotpotQACandidateSentence] = []
    title_sentence_to_node_id: dict[tuple[str, int], NodeId] = {}

    position = 0
    for document in example.documents:
        for sentence_index, sentence in enumerate(document.sentences):
            sentence_id_from_position: NodeId = f"m{position}"
            candidate_sentence = HotpotQACandidateSentence(
                sentence_id=sentence_id_from_position,
                title=document.title,
                sentence_index=sentence_index,
                position=position,
                text=sentence,
            )
            candidate_sentences.append(candidate_sentence)
            title_sentence_to_node_id[(document.title, sentence_index)] = sentence_id_from_position
            position += 1

    if not candidate_sentences:
        raise ValueError(f"HotpotQA example _id={example.raw_id} contains no candidate sentences.")

    gold_evidence_sentence_ids: list[NodeId] = []
    for supporting_fact in example.supporting_facts:
        node_id = title_sentence_to_node_id.get((supporting_fact.title, supporting_fact.sentence_id))
        if node_id is None:
            raise ValueError(
                "HotpotQA example "
                f"_id={example.raw_id} supporting fact ({supporting_fact.title}, {supporting_fact.sentence_id}) "
                "cannot map to a candidate sentence."
            )
        if node_id not in gold_evidence_sentence_ids:
            gold_evidence_sentence_ids.append(node_id)

    if not gold_evidence_sentence_ids:
        raise ValueError(f"HotpotQA example _id={example.raw_id} must contain at least one supporting fact.")

    ranking_record = HotpotQARankingRecord(
        task_id=task_id,
        question=example.question,
        candidate_sentences=tuple(candidate_sentences),
    )
    label_record = HotpotQALabelRecord(
        task_id=task_id,
        gold_answer=example.answer,
        gold_evidence_sentence_ids=tuple(gold_evidence_sentence_ids),
        gold_dependency_edges=(),
    )
    return ranking_record, label_record
