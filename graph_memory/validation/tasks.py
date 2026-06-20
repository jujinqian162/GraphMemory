from __future__ import annotations

from graph_memory.validation.common import (
    ContractValidationError,
    ValidationRecord,
    _reject_unknown_fields,
    _require_record_list,
    _require_record_map,
    _require_unique,
    _required_int,
    _required_string,
    validate_no_label_fields,
)

HOTPOTQA_RANKING_RECORD_FIELDS = {"task_id", "question", "candidate_sentences", "metadata", "debug"}
HOTPOTQA_CANDIDATE_SENTENCE_FIELDS = {"sentence_id", "title", "sentence_index", "position", "text"}
HOTPOTQA_LABEL_RECORD_FIELDS = {
    "task_id",
    "gold_answer",
    "gold_evidence_sentence_ids",
    "gold_dependency_edges",
    "metadata",
    "debug",
}


def validate_hotpotqa_ranking_records(records: object) -> None:
    records = _require_record_list(records, "HotpotQA ranking records")

    seen_task_ids: set[str] = set()
    for index, ranking_record in enumerate(records):
        if not isinstance(ranking_record, dict):
            raise ContractValidationError(f"Invalid HotpotQA ranking records: record index={index} is not an object.")
        task_id = _required_string(ranking_record, "task_id", "HotpotQA ranking record")
        _reject_unknown_fields(ranking_record, HOTPOTQA_RANKING_RECORD_FIELDS, "HotpotQA ranking record", task_id)
        validate_no_label_fields(ranking_record, artifact_name="HotpotQA ranking record", task_id=task_id)
        if "gold_evidence_sentence_ids" in ranking_record:
            raise ContractValidationError(
                f"Invalid HotpotQA ranking record: task_id={task_id} forbidden label field gold_evidence_sentence_ids."
            )
        _require_unique(task_id, seen_task_ids, "HotpotQA ranking record task_id")
        _required_string(ranking_record, "question", "HotpotQA ranking record", task_id)

        candidate_sentences = ranking_record.get("candidate_sentences")
        if not isinstance(candidate_sentences, list) or not candidate_sentences:
            raise ContractValidationError(
                f"Invalid HotpotQA ranking record: task_id={task_id} candidate_sentences must be non-empty."
            )

        seen_sentence_ids: set[str] = set()
        for expected_position, candidate_sentence in enumerate(candidate_sentences):
            if not isinstance(candidate_sentence, dict):
                raise ContractValidationError(
                    "Invalid HotpotQA ranking record: "
                    f"task_id={task_id} candidate sentence index={expected_position} is not an object."
                )
            _reject_unknown_fields(
                candidate_sentence,
                HOTPOTQA_CANDIDATE_SENTENCE_FIELDS,
                "HotpotQA candidate sentence",
                task_id,
            )
            sentence_id = _required_string(candidate_sentence, "sentence_id", "HotpotQA candidate sentence", task_id)
            _require_unique(sentence_id, seen_sentence_ids, f"HotpotQA candidate sentence id task_id={task_id}")
            if sentence_id != f"m{expected_position}":
                raise ContractValidationError(
                    "Invalid HotpotQA ranking record: "
                    f"task_id={task_id} sentence_id={sentence_id} expected m{expected_position}."
                )
            _required_string(candidate_sentence, "title", "HotpotQA candidate sentence", task_id)
            _required_string(candidate_sentence, "text", "HotpotQA candidate sentence", task_id)
            _required_int(candidate_sentence, "sentence_index", "HotpotQA candidate sentence", task_id, minimum=0)
            position = _required_int(candidate_sentence, "position", "HotpotQA candidate sentence", task_id, minimum=0)
            if position != expected_position:
                raise ContractValidationError(
                    "Invalid HotpotQA ranking record: "
                    f"task_id={task_id} sentence_id={sentence_id} position={position} expected {expected_position}."
                )


def validate_hotpotqa_label_records(records: object, records_by_task_id: object) -> None:
    records = _require_record_list(records, "HotpotQA label records")
    records_by_task_id = _require_record_map(records_by_task_id, "HotpotQA ranking records by task_id")

    seen_task_ids: set[str] = set()
    for index, label_record in enumerate(records):
        if not isinstance(label_record, dict):
            raise ContractValidationError(f"Invalid HotpotQA label records: record index={index} is not an object.")
        task_id = _required_string(label_record, "task_id", "HotpotQA label record")
        _reject_unknown_fields(label_record, HOTPOTQA_LABEL_RECORD_FIELDS, "HotpotQA label record", task_id)
        _require_unique(task_id, seen_task_ids, "HotpotQA label record task_id")
        if task_id not in records_by_task_id:
            raise ContractValidationError(f"Invalid HotpotQA label record: task_id={task_id} has no matching ranking record.")

        _required_string(label_record, "gold_answer", "HotpotQA label record", task_id)
        gold_sentence_ids = label_record.get("gold_evidence_sentence_ids")
        if not isinstance(gold_sentence_ids, list) or not gold_sentence_ids:
            raise ContractValidationError(
                "Invalid HotpotQA label record: "
                f"task_id={task_id} gold_evidence_sentence_ids must be a non-empty list."
            )
        if len(gold_sentence_ids) != len(set(gold_sentence_ids)):
            raise ContractValidationError(
                f"Invalid HotpotQA label record: task_id={task_id} duplicate gold evidence sentence."
            )

        valid_sentence_ids = _hotpotqa_candidate_sentence_ids(records_by_task_id[task_id])
        for sentence_id in gold_sentence_ids:
            if not isinstance(sentence_id, str) or sentence_id not in valid_sentence_ids:
                raise ContractValidationError(
                    "Invalid HotpotQA label record: "
                    f"task_id={task_id} gold sentence={sentence_id} does not exist in ranking record."
                )

        dependency_edges = label_record.get("gold_dependency_edges")
        if not isinstance(dependency_edges, list):
            raise ContractValidationError(
                f"Invalid HotpotQA label record: task_id={task_id} gold_dependency_edges must be a list."
            )
        if dependency_edges:
            raise ContractValidationError(
                f"Invalid HotpotQA label record: task_id={task_id} gold_dependency_edges must be empty for HotpotQA Phase 1."
            )


def _hotpotqa_candidate_sentence_ids(ranking_record: ValidationRecord) -> set[str]:
    candidate_sentences = ranking_record.get("candidate_sentences")
    if not isinstance(candidate_sentences, list):
        return set()
    return {
        candidate_sentence["sentence_id"]
        for candidate_sentence in candidate_sentences
        if isinstance(candidate_sentence, dict) and "sentence_id" in candidate_sentence
    }


__all__ = ["validate_hotpotqa_label_records", "validate_hotpotqa_ranking_records", "validate_no_label_fields"]
