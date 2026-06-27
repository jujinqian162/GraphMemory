from __future__ import annotations

from collections.abc import Sequence

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.datasets.longmemeval.records import (
    ConvertedLongMemEvalExample,
    LongMemEvalConversionResult,
    LongMemEvalExample,
    LongMemEvalLabelRecord,
    LongMemEvalTurnItem,
    LongMemEvalRankingRecord,
)


def convert_longmemeval_examples(examples: Sequence[LongMemEvalExample]) -> LongMemEvalConversionResult:
    converted_examples = [convert_longmemeval_example(example) for example in examples]
    return LongMemEvalConversionResult(
        ranking_records=[converted_example.ranking_record for converted_example in converted_examples],
        label_records=[converted_example.label_record for converted_example in converted_examples],
    )


def convert_longmemeval_example(example: LongMemEvalExample) -> ConvertedLongMemEvalExample:
    task_id: TaskId = f"longmem_{example.raw_id}"
    candidate_items: list[LongMemEvalTurnItem] = []
    gold_support_item_ids: list[NodeId] = []

    global_position = 0
    for session_order, session in enumerate(example.sessions):
        for turn_index, turn in enumerate(session.turns):
            item_id: NodeId = f"m{global_position}"
            candidate_item: LongMemEvalTurnItem = {
                "item_id": item_id,
                "session_id": session.session_id,
                "session_order": session_order,
                "turn_index": turn_index,
                "global_position": global_position,
                "role": turn.role,
                "datetime": session.datetime,
                "text": turn.content,
            }
            candidate_items.append(candidate_item)
            if turn.has_answer:
                gold_support_item_ids.append(item_id)
            global_position += 1

    if not candidate_items:
        raise ValueError(f"LongMemEval example question_id={example.raw_id} contains no candidate turns.")
    if not gold_support_item_ids:
        raise ValueError(
            f"LongMemEval example question_id={example.raw_id} has no precise turn support labels."
        )

    session_ids = {session.session_id for session in example.sessions}
    missing_answer_sessions = [session_id for session_id in example.answer_session_ids if session_id not in session_ids]
    if missing_answer_sessions:
        raise ValueError(
            "LongMemEval example "
            f"question_id={example.raw_id} answer_session_ids missing from haystack: {missing_answer_sessions}."
        )

    ranking_record: LongMemEvalRankingRecord = {
        "task_id": task_id,
        "question": example.question,
        "question_datetime": example.question_datetime,
        "candidate_items": candidate_items,
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": example.raw_id,
            "question_type": example.question_type,
            "candidate_granularity": "turn",
        },
    }
    label_record: LongMemEvalLabelRecord = {
        "task_id": task_id,
        "gold_answer": example.answer,
        "gold_support_item_ids": gold_support_item_ids,
        "gold_support_session_ids": list(example.answer_session_ids),
        "gold_dependency_edges": [],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": example.raw_id,
            "question_type": example.question_type,
            "support_label_source": "has_answer",
        },
    }
    return ConvertedLongMemEvalExample(ranking_record=ranking_record, label_record=label_record)
