from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.datasets.twowiki.records import (
    TwoWikiDocument,
    TwoWikiEvidenceTriple,
    TwoWikiExample,
    TwoWikiSupportingFact,
)

TWOWIKI_RAW_FIELDS = {
    "_id",
    "type",
    "question",
    "context",
    "supporting_facts",
    "evidences",
    "evidences_id",
    "answer_id",
    "answer",
}


def parse_twowiki_examples(raw_records: Sequence[object]) -> list[TwoWikiExample]:
    return [
        parse_twowiki_example(raw_record, record_index=record_index)
        for record_index, raw_record in enumerate(raw_records)
    ]


def parse_twowiki_example(raw_record: object, *, record_index: int | None = None) -> TwoWikiExample:
    path = "2Wiki example" if record_index is None else f"2Wiki example index={record_index}"
    record = _required_record(raw_record, path)
    _reject_unknown_fields(record, path)
    raw_id = _required_string(record, "_id", path)
    question = _required_string(record, "question", path)
    answer = _required_string(record, "answer", path)
    question_type = _required_string(record, "type", path)
    documents = _parse_documents(_required_sequence(record, "context", path), raw_id=raw_id)
    supporting_facts = _parse_supporting_facts(_required_sequence(record, "supporting_facts", path), raw_id=raw_id)
    evidences = _parse_evidence_triples(_required_sequence(record, "evidences", path), raw_id=raw_id, field_name="evidences")
    evidences_id = _parse_evidences_id(record, raw_id=raw_id)
    answer_id = _optional_string(record, "answer_id", path)
    return TwoWikiExample(
        raw_id=raw_id,
        question=question,
        answer=answer,
        question_type=question_type,
        documents=tuple(documents),
        supporting_facts=tuple(supporting_facts),
        evidences=tuple(evidences),
        evidences_id=tuple(evidences_id),
        answer_id=answer_id,
    )


def _parse_documents(raw_documents: Sequence[object], *, raw_id: str) -> list[TwoWikiDocument]:
    documents: list[TwoWikiDocument] = []
    for document_index, raw_document in enumerate(raw_documents):
        if not isinstance(raw_document, (list, tuple)) or len(raw_document) != 2:
            raise ValueError(f"2Wiki example _id={raw_id} context[{document_index}] must be [title, sentences].")
        raw_title, raw_sentences = raw_document
        if not isinstance(raw_title, str) or not raw_title:
            raise ValueError(f"2Wiki example _id={raw_id} context[{document_index}] title must be a non-empty string.")
        if not isinstance(raw_sentences, list) or not raw_sentences:
            raise ValueError(
                f"2Wiki example _id={raw_id} context[{document_index}] sentences must be a non-empty list."
            )

        sentences: list[str] = []
        for sentence_id, raw_sentence in enumerate(raw_sentences):
            if not isinstance(raw_sentence, str):
                raise ValueError(
                    f"2Wiki example _id={raw_id} title={raw_title} sentence_id={sentence_id} must be text."
                )
            sentences.append(raw_sentence)
        documents.append(TwoWikiDocument(title=raw_title, sentences=tuple(sentences)))
    return documents


def _parse_supporting_facts(raw_supporting_facts: Sequence[object], *, raw_id: str) -> list[TwoWikiSupportingFact]:
    supporting_facts: list[TwoWikiSupportingFact] = []
    for fact_index, raw_supporting_fact in enumerate(raw_supporting_facts):
        if not isinstance(raw_supporting_fact, (list, tuple)) or len(raw_supporting_fact) != 2:
            raise ValueError(f"2Wiki example _id={raw_id} supporting_facts[{fact_index}] must be [title, sentence_id].")
        raw_title, raw_sentence_id = raw_supporting_fact
        if not isinstance(raw_title, str) or not raw_title:
            raise ValueError(f"2Wiki example _id={raw_id} supporting_facts[{fact_index}] title must be a non-empty string.")
        if isinstance(raw_sentence_id, bool) or not isinstance(raw_sentence_id, int):
            raise ValueError(f"2Wiki example _id={raw_id} supporting_facts[{fact_index}] sentence_id must be an int.")
        supporting_facts.append(TwoWikiSupportingFact(title=raw_title, sentence_id=raw_sentence_id))
    return supporting_facts


def _parse_evidences_id(record: Mapping[str, object], *, raw_id: str) -> list[TwoWikiEvidenceTriple]:
    value = record.get("evidences_id")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"2Wiki example _id={raw_id} evidences_id must be a list when present.")
    return _parse_evidence_triples(value, raw_id=raw_id, field_name="evidences_id")


def _parse_evidence_triples(
    raw_evidences: Sequence[object],
    *,
    raw_id: str,
    field_name: str,
) -> list[TwoWikiEvidenceTriple]:
    triples: list[TwoWikiEvidenceTriple] = []
    for evidence_index, raw_evidence in enumerate(raw_evidences):
        if not isinstance(raw_evidence, (list, tuple)) or len(raw_evidence) != 3:
            raise ValueError(f"2Wiki example _id={raw_id} {field_name}[{evidence_index}] must be [subject, relation, object].")
        raw_subject, raw_relation, raw_object = raw_evidence
        if not isinstance(raw_subject, str) or not raw_subject:
            raise ValueError(f"2Wiki example _id={raw_id} {field_name}[{evidence_index}] subject must be a non-empty string.")
        if not isinstance(raw_relation, str) or not raw_relation:
            raise ValueError(f"2Wiki example _id={raw_id} {field_name}[{evidence_index}] relation must be a non-empty string.")
        if not isinstance(raw_object, str) or not raw_object:
            raise ValueError(f"2Wiki example _id={raw_id} {field_name}[{evidence_index}] object must be a non-empty string.")
        triples.append(TwoWikiEvidenceTriple(subject=raw_subject, relation=raw_relation, object=raw_object))
    return triples


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


def _optional_string(record: Mapping[str, object], field_name: str, path: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} field={field_name} must be a non-empty string when present.")
    return value


def _reject_unknown_fields(record: Mapping[str, object], path: str) -> None:
    unknown = sorted(set(record) - TWOWIKI_RAW_FIELDS)
    if unknown:
        raise ValueError(f"{path} unknown fields={unknown}.")
