from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.datasets.hotpotqa.records import HotpotQADocument, HotpotQAExample, HotpotQASupportingFact


def parse_hotpotqa_examples(raw_records: Sequence[object]) -> list[HotpotQAExample]:
    return [
        parse_hotpotqa_example(raw_record, record_index=record_index)
        for record_index, raw_record in enumerate(raw_records)
    ]


def parse_hotpotqa_example(raw_record: object, *, record_index: int | None = None) -> HotpotQAExample:
    path = "HotpotQA example" if record_index is None else f"HotpotQA example index={record_index}"
    record = _required_record(raw_record, path)
    raw_id = _required_string(record, "_id", path)
    question = _required_string(record, "question", path)
    answer = _required_string(record, "answer", path, allow_empty=True)
    documents = _parse_documents(_required_sequence(record, "context", path), raw_id=raw_id)
    supporting_facts = _parse_supporting_facts(_required_sequence(record, "supporting_facts", path), raw_id=raw_id)
    return HotpotQAExample(
        raw_id=raw_id,
        question=question,
        answer=answer,
        documents=tuple(documents),
        supporting_facts=tuple(supporting_facts),
    )


def _parse_documents(raw_documents: Sequence[object], *, raw_id: str) -> list[HotpotQADocument]:
    documents: list[HotpotQADocument] = []
    for document_index, raw_document in enumerate(raw_documents):
        if not isinstance(raw_document, (list, tuple)) or len(raw_document) != 2:
            raise ValueError(f"HotpotQA example _id={raw_id} context[{document_index}] must be [title, sentences].")
        raw_title, raw_sentences = raw_document
        if not isinstance(raw_title, str) or not raw_title:
            raise ValueError(f"HotpotQA example _id={raw_id} context[{document_index}] title must be a non-empty string.")
        if not isinstance(raw_sentences, list):
            raise ValueError(f"HotpotQA example _id={raw_id} context[{document_index}] sentences must be a list.")

        sentences: list[str] = []
        for sentence_id, raw_sentence in enumerate(raw_sentences):
            if not isinstance(raw_sentence, str):
                raise ValueError(
                    f"HotpotQA example _id={raw_id} title={raw_title} sentence_id={sentence_id} must be text."
                )
            sentences.append(raw_sentence)
        documents.append(HotpotQADocument(title=raw_title, sentences=tuple(sentences)))
    return documents


def _parse_supporting_facts(raw_supporting_facts: Sequence[object], *, raw_id: str) -> list[HotpotQASupportingFact]:
    supporting_facts: list[HotpotQASupportingFact] = []
    for fact_index, raw_supporting_fact in enumerate(raw_supporting_facts):
        if not isinstance(raw_supporting_fact, (list, tuple)) or len(raw_supporting_fact) != 2:
            raise ValueError(f"HotpotQA example _id={raw_id} supporting_facts[{fact_index}] must be [title, sentence_id].")
        raw_title, raw_sentence_id = raw_supporting_fact
        if not isinstance(raw_title, str):
            raise ValueError(f"HotpotQA example _id={raw_id} supporting_facts[{fact_index}] title must be a string.")
        if isinstance(raw_sentence_id, bool) or not isinstance(raw_sentence_id, int):
            raise ValueError(f"HotpotQA example _id={raw_id} supporting_facts[{fact_index}] sentence_id must be an int.")
        supporting_facts.append(HotpotQASupportingFact(title=raw_title, sentence_id=raw_sentence_id))
    return supporting_facts


def _required_record(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    return value


def _required_sequence(record: Mapping[str, object], field_name: str, path: str) -> Sequence[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a {field_name} list.")
    return value


def _required_string(
    record: Mapping[str, object], field_name: str, path: str, *, allow_empty: bool = False
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{path} must contain a non-empty string {field_name}.")
    return value

