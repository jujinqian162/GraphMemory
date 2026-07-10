from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.datasets.musique.records import (
    MuSiQueDecompositionStep,
    MuSiQueExample,
    MuSiQueParagraph,
)

MUSIQUE_RAW_FIELDS = {
    "id",
    "paragraphs",
    "question",
    "question_decomposition",
    "answer",
    "answer_aliases",
    "answerable",
}


def parse_musique_examples(raw_records: Sequence[object]) -> list[MuSiQueExample]:
    return [
        parse_musique_example(raw_record, record_index=record_index)
        for record_index, raw_record in enumerate(raw_records)
    ]


def parse_musique_example(raw_record: object, *, record_index: int | None = None) -> MuSiQueExample:
    path = "MuSiQue example" if record_index is None else f"MuSiQue example index={record_index}"
    record = _required_record(raw_record, path)
    _reject_unknown_fields(record, path)
    raw_id = _required_string(record, "id", path)
    question = _required_string(record, "question", path)
    answer = _required_string(record, "answer", path)
    answer_aliases = _parse_answer_aliases(_required_sequence(record, "answer_aliases", path), raw_id=raw_id)
    answerable = _required_bool(record, "answerable", path)
    if not answerable:
        raise ValueError(f"MuSiQue example id={raw_id} answerable must be true for MuSiQue-Ans.")
    paragraphs = _parse_paragraphs(_required_sequence(record, "paragraphs", path), raw_id=raw_id)
    question_decomposition = _parse_question_decomposition(
        _required_sequence(record, "question_decomposition", path),
        raw_id=raw_id,
    )
    return MuSiQueExample(
        raw_id=raw_id,
        question=question,
        answer=answer,
        answer_aliases=tuple(answer_aliases),
        answerable=answerable,
        paragraphs=tuple(paragraphs),
        question_decomposition=tuple(question_decomposition),
    )


def _parse_answer_aliases(raw_aliases: Sequence[object], *, raw_id: str) -> list[str]:
    aliases: list[str] = []
    for alias_index, raw_alias in enumerate(raw_aliases):
        if not isinstance(raw_alias, str):
            raise ValueError(f"MuSiQue example id={raw_id} answer_aliases[{alias_index}] must be text.")
        if raw_alias:
            aliases.append(raw_alias)
    return aliases


def _parse_paragraphs(raw_paragraphs: Sequence[object], *, raw_id: str) -> list[MuSiQueParagraph]:
    if not raw_paragraphs:
        raise ValueError(f"MuSiQue example id={raw_id} paragraphs must be non-empty.")
    paragraphs: list[MuSiQueParagraph] = []
    seen_indices: set[int] = set()
    for paragraph_position, raw_paragraph in enumerate(raw_paragraphs):
        if not isinstance(raw_paragraph, Mapping):
            raise ValueError(f"MuSiQue example id={raw_id} paragraphs[{paragraph_position}] must be an object.")
        paragraph_record = cast(Mapping[str, object], raw_paragraph)
        idx = _required_int(paragraph_record, "idx", f"MuSiQue example id={raw_id} paragraph", minimum=0)
        if idx in seen_indices:
            raise ValueError(f"MuSiQue example id={raw_id} duplicate paragraph idx={idx}.")
        seen_indices.add(idx)
        paragraphs.append(
            MuSiQueParagraph(
                idx=idx,
                title=_required_string(paragraph_record, "title", f"MuSiQue example id={raw_id} paragraph idx={idx}"),
                paragraph_text=_required_string(
                    paragraph_record,
                    "paragraph_text",
                    f"MuSiQue example id={raw_id} paragraph idx={idx}",
                ),
                is_supporting=_required_bool(
                    paragraph_record,
                    "is_supporting",
                    f"MuSiQue example id={raw_id} paragraph idx={idx}",
                ),
            )
        )
    return paragraphs


def _parse_question_decomposition(
    raw_steps: Sequence[object],
    *,
    raw_id: str,
) -> list[MuSiQueDecompositionStep]:
    if not raw_steps:
        raise ValueError(f"MuSiQue example id={raw_id} question_decomposition must be non-empty.")
    steps: list[MuSiQueDecompositionStep] = []
    for step_position, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"MuSiQue example id={raw_id} question_decomposition[{step_position}] must be an object.")
        step_record = cast(Mapping[str, object], raw_step)
        paragraph_support_idx = _optional_int(
            step_record,
            "paragraph_support_idx",
            f"MuSiQue example id={raw_id} question_decomposition[{step_position}]",
            minimum=0,
        )
        steps.append(
            MuSiQueDecompositionStep(
                step_id=_required_step_id(
                    step_record,
                    "id",
                    f"MuSiQue example id={raw_id} question_decomposition[{step_position}]",
                ),
                question=_required_string(
                    step_record,
                    "question",
                    f"MuSiQue example id={raw_id} question_decomposition[{step_position}]",
                ),
                answer=_required_string(
                    step_record,
                    "answer",
                    f"MuSiQue example id={raw_id} question_decomposition[{step_position}]",
                ),
                paragraph_support_idx=paragraph_support_idx,
            )
        )
    return steps


def _required_record(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _required_sequence(record: Mapping[str, object], field_name: str, path: str) -> Sequence[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a {field_name} list.")
    return cast(Sequence[object], value)


def _required_string(record: Mapping[str, object], field_name: str, path: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must contain a non-empty string {field_name}.")
    return value


def _required_step_id(record: Mapping[str, object], field_name: str, path: str) -> str:
    value = record.get(field_name)
    if isinstance(value, bool):
        raise ValueError(f"{path} must contain a non-empty string or int {field_name}.")
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{path} must contain a non-empty string or int {field_name}.")
        return value
    if isinstance(value, int):
        return str(value)
    raise ValueError(f"{path} must contain a non-empty string or int {field_name}.")


def _required_bool(record: Mapping[str, object], field_name: str, path: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{path} must contain a boolean {field_name}.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, path: str, *, minimum: int) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must contain an int {field_name} >= {minimum}.")
    return value


def _optional_int(record: Mapping[str, object], field_name: str, path: str, *, minimum: int) -> int | None:
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} field={field_name} must be an int >= {minimum} when present.")
    return value


def _reject_unknown_fields(record: Mapping[str, object], path: str) -> None:
    unknown = sorted(set(record) - MUSIQUE_RAW_FIELDS)
    if unknown:
        raise ValueError(f"{path} unknown fields={unknown}.")
