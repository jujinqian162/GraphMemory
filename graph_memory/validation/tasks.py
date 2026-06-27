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

TWOWIKI_RANKING_RECORD_FIELDS = {"task_id", "question", "question_type", "candidate_sentences", "metadata", "debug"}
TWOWIKI_CANDIDATE_SENTENCE_FIELDS = {"sentence_id", "title", "sentence_index", "position", "text"}
TWOWIKI_LABEL_RECORD_FIELDS = {
    "task_id",
    "gold_answer",
    "gold_evidence_sentence_ids",
    "gold_dependency_edges",
    "metadata",
    "debug",
}

LONGMEMEVAL_RANKING_RECORD_FIELDS = {"task_id", "question", "question_datetime", "candidate_items", "metadata", "debug"}
LONGMEMEVAL_CANDIDATE_ITEM_FIELDS = {
    "item_id",
    "session_id",
    "session_order",
    "turn_index",
    "global_position",
    "role",
    "datetime",
    "text",
}
LONGMEMEVAL_LABEL_RECORD_FIELDS = {
    "task_id",
    "gold_answer",
    "gold_support_item_ids",
    "gold_support_session_ids",
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


def validate_twowiki_ranking_records(records: object) -> None:
    records = _require_record_list(records, "2Wiki ranking records")

    seen_task_ids: set[str] = set()
    for index, ranking_record in enumerate(records):
        if not isinstance(ranking_record, dict):
            raise ContractValidationError(f"Invalid 2Wiki ranking records: record index={index} is not an object.")
        task_id = _required_string(ranking_record, "task_id", "2Wiki ranking record")
        _reject_unknown_fields(ranking_record, TWOWIKI_RANKING_RECORD_FIELDS, "2Wiki ranking record", task_id)
        validate_no_label_fields(ranking_record, artifact_name="2Wiki ranking record", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "2Wiki ranking record task_id")
        if not task_id.startswith("2wiki_"):
            raise ContractValidationError(f"Invalid 2Wiki ranking record: task_id={task_id} must start with 2wiki_.")
        _required_string(ranking_record, "question", "2Wiki ranking record", task_id)
        _required_string(ranking_record, "question_type", "2Wiki ranking record", task_id)
        _require_metadata_object(ranking_record, "2Wiki ranking record", task_id)

        candidate_sentences = ranking_record.get("candidate_sentences")
        if not isinstance(candidate_sentences, list) or not candidate_sentences:
            raise ContractValidationError(
                f"Invalid 2Wiki ranking record: task_id={task_id} candidate_sentences must be non-empty."
            )

        seen_sentence_ids: set[str] = set()
        for expected_position, candidate_sentence in enumerate(candidate_sentences):
            if not isinstance(candidate_sentence, dict):
                raise ContractValidationError(
                    "Invalid 2Wiki ranking record: "
                    f"task_id={task_id} candidate sentence index={expected_position} is not an object."
                )
            _reject_unknown_fields(
                candidate_sentence,
                TWOWIKI_CANDIDATE_SENTENCE_FIELDS,
                "2Wiki candidate sentence",
                task_id,
            )
            sentence_id = _required_string(candidate_sentence, "sentence_id", "2Wiki candidate sentence", task_id)
            _require_unique(sentence_id, seen_sentence_ids, f"2Wiki candidate sentence id task_id={task_id}")
            if sentence_id != f"m{expected_position}":
                raise ContractValidationError(
                    "Invalid 2Wiki ranking record: "
                    f"task_id={task_id} sentence_id={sentence_id} expected m{expected_position}."
                )
            _required_string(candidate_sentence, "title", "2Wiki candidate sentence", task_id)
            _required_string(candidate_sentence, "text", "2Wiki candidate sentence", task_id)
            _required_int(candidate_sentence, "sentence_index", "2Wiki candidate sentence", task_id, minimum=0)
            position = _required_int(candidate_sentence, "position", "2Wiki candidate sentence", task_id, minimum=0)
            if position != expected_position:
                raise ContractValidationError(
                    "Invalid 2Wiki ranking record: "
                    f"task_id={task_id} sentence_id={sentence_id} position={position} expected {expected_position}."
                )


def validate_twowiki_label_records(records: object, records_by_task_id: object) -> None:
    records = _require_record_list(records, "2Wiki label records")
    records_by_task_id = _require_record_map(records_by_task_id, "2Wiki ranking records by task_id")

    seen_task_ids: set[str] = set()
    for index, label_record in enumerate(records):
        if not isinstance(label_record, dict):
            raise ContractValidationError(f"Invalid 2Wiki label records: record index={index} is not an object.")
        task_id = _required_string(label_record, "task_id", "2Wiki label record")
        _reject_unknown_fields(label_record, TWOWIKI_LABEL_RECORD_FIELDS, "2Wiki label record", task_id)
        _require_unique(task_id, seen_task_ids, "2Wiki label record task_id")
        if task_id not in records_by_task_id:
            raise ContractValidationError(f"Invalid 2Wiki label record: task_id={task_id} has no matching ranking record.")

        _required_string(label_record, "gold_answer", "2Wiki label record", task_id)
        _require_metadata_object(label_record, "2Wiki label record", task_id)
        gold_sentence_ids = label_record.get("gold_evidence_sentence_ids")
        if not isinstance(gold_sentence_ids, list) or not gold_sentence_ids:
            raise ContractValidationError(
                "Invalid 2Wiki label record: "
                f"task_id={task_id} gold_evidence_sentence_ids must be a non-empty list."
            )
        if len(gold_sentence_ids) != len(set(gold_sentence_ids)):
            raise ContractValidationError(
                f"Invalid 2Wiki label record: task_id={task_id} duplicate gold evidence sentence."
            )

        valid_sentence_ids = _hotpotqa_candidate_sentence_ids(records_by_task_id[task_id])
        for sentence_id in gold_sentence_ids:
            if not isinstance(sentence_id, str) or sentence_id not in valid_sentence_ids:
                raise ContractValidationError(
                    "Invalid 2Wiki label record: "
                    f"task_id={task_id} gold sentence={sentence_id} does not exist in ranking record."
                )

        dependency_edges = label_record.get("gold_dependency_edges")
        if not isinstance(dependency_edges, list):
            raise ContractValidationError(
                f"Invalid 2Wiki label record: task_id={task_id} gold_dependency_edges must be a list."
            )
        for edge_index, edge in enumerate(dependency_edges):
            if not isinstance(edge, list) or len(edge) != 2:
                raise ContractValidationError(
                    f"Invalid 2Wiki label record: task_id={task_id} gold_dependency_edges[{edge_index}] must be [source, target]."
                )
            source, target = edge
            if not isinstance(source, str) or source not in valid_sentence_ids:
                raise ContractValidationError(
                    f"Invalid 2Wiki label record: task_id={task_id} gold edge source={source} does not exist."
                )
            if not isinstance(target, str) or target not in valid_sentence_ids:
                raise ContractValidationError(
                    f"Invalid 2Wiki label record: task_id={task_id} gold edge target={target} does not exist."
                )


def validate_longmemeval_ranking_records(records: object) -> None:
    records = _require_record_list(records, "LongMemEval ranking records")

    seen_task_ids: set[str] = set()
    for index, ranking_record in enumerate(records):
        if not isinstance(ranking_record, dict):
            raise ContractValidationError(f"Invalid LongMemEval ranking records: record index={index} is not an object.")
        task_id = _required_string(ranking_record, "task_id", "LongMemEval ranking record")
        _reject_unknown_fields(ranking_record, LONGMEMEVAL_RANKING_RECORD_FIELDS, "LongMemEval ranking record", task_id)
        _validate_no_longmemeval_label_fields(ranking_record, artifact_name="LongMemEval ranking record", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "LongMemEval ranking record task_id")
        if not task_id.startswith("longmem_"):
            raise ContractValidationError(f"Invalid LongMemEval ranking record: task_id={task_id} must start with longmem_.")
        _required_string(ranking_record, "question", "LongMemEval ranking record", task_id)
        _required_string(ranking_record, "question_datetime", "LongMemEval ranking record", task_id)
        _require_metadata_object(ranking_record, "LongMemEval ranking record", task_id)

        candidate_items = ranking_record.get("candidate_items")
        if not isinstance(candidate_items, list) or not candidate_items:
            raise ContractValidationError(
                f"Invalid LongMemEval ranking record: task_id={task_id} candidate_items must be non-empty."
            )

        seen_item_ids: set[str] = set()
        for expected_position, candidate_item in enumerate(candidate_items):
            if not isinstance(candidate_item, dict):
                raise ContractValidationError(
                    "Invalid LongMemEval ranking record: "
                    f"task_id={task_id} candidate item index={expected_position} is not an object."
                )
            _reject_unknown_fields(
                candidate_item,
                LONGMEMEVAL_CANDIDATE_ITEM_FIELDS,
                "LongMemEval candidate item",
                task_id,
            )
            _validate_no_longmemeval_label_fields(
                candidate_item,
                artifact_name="LongMemEval candidate item",
                task_id=task_id,
            )
            item_id = _required_string(candidate_item, "item_id", "LongMemEval candidate item", task_id)
            _require_unique(item_id, seen_item_ids, f"LongMemEval candidate item id task_id={task_id}")
            if item_id != f"m{expected_position}":
                raise ContractValidationError(
                    "Invalid LongMemEval ranking record: "
                    f"task_id={task_id} item_id={item_id} expected m{expected_position}."
                )
            _required_string(candidate_item, "session_id", "LongMemEval candidate item", task_id)
            _required_string(candidate_item, "role", "LongMemEval candidate item", task_id)
            _required_string(candidate_item, "datetime", "LongMemEval candidate item", task_id)
            _required_string(candidate_item, "text", "LongMemEval candidate item", task_id)
            _required_int(candidate_item, "session_order", "LongMemEval candidate item", task_id, minimum=0)
            _required_int(candidate_item, "turn_index", "LongMemEval candidate item", task_id, minimum=0)
            global_position = _required_int(candidate_item, "global_position", "LongMemEval candidate item", task_id, minimum=0)
            if global_position != expected_position:
                raise ContractValidationError(
                    "Invalid LongMemEval ranking record: "
                    f"task_id={task_id} item_id={item_id} global_position={global_position} expected {expected_position}."
                )


def validate_longmemeval_label_records(records: object, records_by_task_id: object) -> None:
    records = _require_record_list(records, "LongMemEval label records")
    records_by_task_id = _require_record_map(records_by_task_id, "LongMemEval ranking records by task_id")

    seen_task_ids: set[str] = set()
    for index, label_record in enumerate(records):
        if not isinstance(label_record, dict):
            raise ContractValidationError(f"Invalid LongMemEval label records: record index={index} is not an object.")
        task_id = _required_string(label_record, "task_id", "LongMemEval label record")
        _reject_unknown_fields(label_record, LONGMEMEVAL_LABEL_RECORD_FIELDS, "LongMemEval label record", task_id)
        _require_unique(task_id, seen_task_ids, "LongMemEval label record task_id")
        if task_id not in records_by_task_id:
            raise ContractValidationError(f"Invalid LongMemEval label record: task_id={task_id} has no matching ranking record.")

        _required_string(label_record, "gold_answer", "LongMemEval label record", task_id)
        _require_metadata_object(label_record, "LongMemEval label record", task_id)
        gold_item_ids = label_record.get("gold_support_item_ids")
        if not isinstance(gold_item_ids, list) or not gold_item_ids:
            raise ContractValidationError(
                "Invalid LongMemEval label record: "
                f"task_id={task_id} gold_support_item_ids must be a non-empty list."
            )
        if len(gold_item_ids) != len(set(gold_item_ids)):
            raise ContractValidationError(
                f"Invalid LongMemEval label record: task_id={task_id} duplicate gold support item."
            )

        valid_item_ids = _longmemeval_candidate_item_ids(records_by_task_id[task_id])
        for item_id in gold_item_ids:
            if not isinstance(item_id, str) or item_id not in valid_item_ids:
                raise ContractValidationError(
                    "Invalid LongMemEval label record: "
                    f"task_id={task_id} gold support item={item_id} does not exist in ranking record."
                )

        gold_session_ids = label_record.get("gold_support_session_ids")
        if not isinstance(gold_session_ids, list) or not gold_session_ids:
            raise ContractValidationError(
                "Invalid LongMemEval label record: "
                f"task_id={task_id} gold_support_session_ids must be a non-empty list."
            )
        valid_session_ids = _longmemeval_candidate_session_ids(records_by_task_id[task_id])
        for session_id in gold_session_ids:
            if not isinstance(session_id, str) or session_id not in valid_session_ids:
                raise ContractValidationError(
                    "Invalid LongMemEval label record: "
                    f"task_id={task_id} gold support session={session_id} does not exist in ranking record."
                )

        dependency_edges = label_record.get("gold_dependency_edges")
        if not isinstance(dependency_edges, list):
            raise ContractValidationError(
                f"Invalid LongMemEval label record: task_id={task_id} gold_dependency_edges must be a list."
            )
        if dependency_edges:
            raise ContractValidationError(
                f"Invalid LongMemEval label record: task_id={task_id} gold_dependency_edges must be empty for LongMemEval V1."
            )


def _require_metadata_object(record: ValidationRecord, artifact_name: str, task_id: str) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} metadata must be an object.")


def _hotpotqa_candidate_sentence_ids(ranking_record: ValidationRecord) -> set[str]:
    candidate_sentences = ranking_record.get("candidate_sentences")
    if not isinstance(candidate_sentences, list):
        return set()
    return {
        candidate_sentence["sentence_id"]
        for candidate_sentence in candidate_sentences
        if isinstance(candidate_sentence, dict) and "sentence_id" in candidate_sentence
    }


def _longmemeval_candidate_item_ids(ranking_record: ValidationRecord) -> set[str]:
    candidate_items = ranking_record.get("candidate_items")
    if not isinstance(candidate_items, list):
        return set()
    return {
        candidate_item["item_id"]
        for candidate_item in candidate_items
        if isinstance(candidate_item, dict) and "item_id" in candidate_item
    }


def _longmemeval_candidate_session_ids(ranking_record: ValidationRecord) -> set[str]:
    candidate_items = ranking_record.get("candidate_items")
    if not isinstance(candidate_items, list):
        return set()
    return {
        candidate_item["session_id"]
        for candidate_item in candidate_items
        if isinstance(candidate_item, dict) and "session_id" in candidate_item
    }


def _validate_no_longmemeval_label_fields(value: object, *, artifact_name: str, task_id: str) -> None:
    validate_no_label_fields(value, artifact_name=artifact_name, task_id=task_id)
    _walk_forbidden_longmemeval_fields(value, artifact_name=artifact_name, task_id=task_id, path=artifact_name)


def _walk_forbidden_longmemeval_fields(value: object, *, artifact_name: str, task_id: str, path: str) -> None:
    forbidden = {"answer", "answer_session_ids", "has_answer", "gold_support_item_ids", "gold_support_session_ids"}
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in forbidden:
                raise ContractValidationError(
                    f"Invalid {artifact_name}: task_id={task_id} forbidden label field {key} at {path}.{key}."
                )
            _walk_forbidden_longmemeval_fields(
                nested_value,
                artifact_name=artifact_name,
                task_id=task_id,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _walk_forbidden_longmemeval_fields(
                nested_value,
                artifact_name=artifact_name,
                task_id=task_id,
                path=f"{path}[{index}]",
            )


__all__ = [
    "validate_hotpotqa_label_records",
    "validate_hotpotqa_ranking_records",
    "validate_longmemeval_label_records",
    "validate_longmemeval_ranking_records",
    "validate_no_label_fields",
    "validate_twowiki_label_records",
    "validate_twowiki_ranking_records",
]
