from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.datasets.longmemeval.records import LongMemEvalExample, LongMemEvalSession, LongMemEvalTurn

LONGMEMEVAL_RAW_FIELDS = {
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_session_ids",
    "haystack_dates",
    "haystack_sessions",
    "answer_session_ids",
}


def parse_longmemeval_examples(raw_records: Sequence[object]) -> list[LongMemEvalExample]:
    return [
        parse_longmemeval_example(raw_record, record_index=record_index)
        for record_index, raw_record in enumerate(raw_records)
    ]


def parse_longmemeval_example(raw_record: object, *, record_index: int | None = None) -> LongMemEvalExample:
    path = "LongMemEval example" if record_index is None else f"LongMemEval example index={record_index}"
    record = _required_record(raw_record, path)
    _reject_unknown_fields(record, path)

    raw_id = _required_string(record, "question_id", path)
    question_type = _required_string(record, "question_type", path)
    question = _required_string(record, "question", path)
    answer = _required_string(record, "answer", path)
    question_datetime = _required_string(record, "question_date", path)
    session_ids = _required_sequence(record, "haystack_session_ids", path)
    session_dates = _required_sequence(record, "haystack_dates", path)
    raw_sessions = _required_sequence(record, "haystack_sessions", path)
    answer_session_ids = _parse_string_sequence(
        _required_sequence(record, "answer_session_ids", path),
        field_name="answer_session_ids",
        path=path,
        allow_empty=True,
    )

    if not (len(session_ids) == len(session_dates) == len(raw_sessions)):
        raise ValueError(f"{path} parallel haystack arrays must have equal lengths.")
    if not session_ids:
        raise ValueError(f"{path} must contain at least one haystack session.")

    sessions = [
        _parse_session(
            raw_session_id=raw_session_id,
            raw_session_date=raw_session_date,
            raw_session=raw_session,
            raw_id=raw_id,
            session_order=session_order,
        )
        for session_order, (raw_session_id, raw_session_date, raw_session) in enumerate(
            zip(session_ids, session_dates, raw_sessions, strict=True)
        )
    ]
    return LongMemEvalExample(
        raw_id=raw_id,
        question_type=question_type,
        question=question,
        answer=answer,
        question_datetime=question_datetime,
        sessions=tuple(sessions),
        answer_session_ids=tuple(answer_session_ids),
    )


def _parse_session(
    *,
    raw_session_id: object,
    raw_session_date: object,
    raw_session: object,
    raw_id: str,
    session_order: int,
) -> LongMemEvalSession:
    if not isinstance(raw_session_id, str) or not raw_session_id:
        raise ValueError(f"LongMemEval example question_id={raw_id} haystack_session_ids[{session_order}] must be a non-empty string.")
    if not isinstance(raw_session_date, str) or not raw_session_date:
        raise ValueError(f"LongMemEval example question_id={raw_id} haystack_dates[{session_order}] must be a non-empty string.")
    if not isinstance(raw_session, list) or not raw_session:
        raise ValueError(f"LongMemEval example question_id={raw_id} haystack_sessions[{session_order}] must be a non-empty list.")
    turns = [
        _parse_turn(raw_turn, raw_id=raw_id, session_order=session_order, turn_index=turn_index)
        for turn_index, raw_turn in enumerate(raw_session)
    ]
    return LongMemEvalSession(session_id=raw_session_id, datetime=raw_session_date, turns=tuple(turns))


def _parse_turn(raw_turn: object, *, raw_id: str, session_order: int, turn_index: int) -> LongMemEvalTurn:
    if not isinstance(raw_turn, Mapping):
        raise ValueError(
            "LongMemEval example "
            f"question_id={raw_id} haystack_sessions[{session_order}][{turn_index}] must be an object."
        )
    path = f"question_id={raw_id} haystack_sessions[{session_order}][{turn_index}]"
    role = _required_string(raw_turn, "role", path)
    content = _required_string(raw_turn, "content", path)
    raw_has_answer = raw_turn.get("has_answer", False)
    if not isinstance(raw_has_answer, bool):
        raise ValueError(f"LongMemEval example {path} has_answer must be a boolean when present.")
    return LongMemEvalTurn(role=role, content=content, has_answer=raw_has_answer)


def _required_record(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    return value


def _required_sequence(record: Mapping[str, object], field_name: str, path: str) -> Sequence[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a {field_name} list.")
    return value


def _required_string(record: Mapping[str, object], field_name: str, path: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must contain a non-empty string {field_name}.")
    return value


def _parse_string_sequence(
    values: Sequence[object],
    *,
    field_name: str,
    path: str,
    allow_empty: bool,
) -> list[str]:
    if not allow_empty and not values:
        raise ValueError(f"{path} {field_name} must be non-empty.")
    strings: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path} {field_name}[{index}] must be a non-empty string.")
        strings.append(value)
    return strings


def _reject_unknown_fields(record: Mapping[str, object], path: str) -> None:
    unknown = sorted(set(record) - LONGMEMEVAL_RAW_FIELDS)
    if unknown:
        raise ValueError(f"{path} unknown fields={unknown}.")
