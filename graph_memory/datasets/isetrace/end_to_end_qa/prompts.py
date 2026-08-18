from __future__ import annotations

from graph_memory.datasets.isetrace.end_to_end_qa.contracts import (
    AnswerResponse,
    Condition,
    JudgeResponse,
    PreparedRecord,
)

ANSWER_PROMPT_VERSION = "isetrace-e10-answer-v2-no-evidence-control"
JUDGE_PROMPT_VERSION = "isetrace-e10-judge-v1"

ANSWER_SYSTEM_PROMPT = """You answer a memory question using only the supplied evidence.
Do not use outside knowledge or unsupported inference. Give the shortest complete answer
that resolves every part of the question. If the evidence is insufficient to answer the
question completely, set abstained=true and answer exactly INSUFFICIENT_EVIDENCE.
If abstained=false, the answer must be fully supported by the evidence."""

NO_EVIDENCE_SYSTEM_PROMPT = """You answer a memory question without access to memory evidence.
Answer only from information stated in the question or knowledge you already have; do not
use tools or invent details. Give the shortest complete answer that resolves every part of
the question. If you cannot answer completely, set abstained=true and answer exactly
INSUFFICIENT_EVIDENCE."""

JUDGE_SYSTEM_PROMPT = """You are an independent evaluator of an answer to a memory question.
Use the frozen gold evidence to determine factual completeness and correctness. Use the
retrieved evidence to determine whether every substantive answer claim is supported.
Do not infer the retrieval method from wording or metadata. Apply these labels exactly:
- correctness=correct: all requested facts are correct and complete.
- correctness=partial: at least one requested fact is correct, but the answer is incomplete
  or contains a limited error.
- correctness=incorrect: the answer fails to provide a correct substantive answer.
- correctness=not_answered: the answer abstains.
- faithfulness=faithful: every substantive answer claim is supported by retrieved evidence.
- faithfulness=mixed: some but not all substantive claims are supported.
- faithfulness=unsupported: no substantive answer claim is supported.
- faithfulness=not_applicable: the answer abstains and makes no substantive claim.
The abstained field must describe the answer itself."""


def answer_system_prompt(record: PreparedRecord) -> str:
    return (
        NO_EVIDENCE_SYSTEM_PROMPT
        if record.condition is Condition.NO_EVIDENCE
        else ANSWER_SYSTEM_PROMPT
    )


def answer_payload(record: PreparedRecord) -> dict[str, object]:
    return {
        "query": record.query,
        "evidence": [item.text for item in record.evidence],
    }


def judge_payload(
    prepared: PreparedRecord,
    answer: AnswerResponse,
) -> dict[str, object]:
    return {
        "query": prepared.query,
        "gold_evidence": prepared.gold_quotes,
        "retrieved_evidence": [item.text for item in prepared.evidence],
        "answer": answer.answer,
        "answer_declared_abstention": answer.abstained,
    }


def validate_answer(value: object) -> dict[str, object]:
    return AnswerResponse.model_validate(value).model_dump(mode="json")


def validate_judgment(value: object) -> dict[str, object]:
    return JudgeResponse.model_validate(value).model_dump(mode="json")


__all__ = [
    "ANSWER_PROMPT_VERSION",
    "ANSWER_SYSTEM_PROMPT",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "NO_EVIDENCE_SYSTEM_PROMPT",
    "answer_payload",
    "answer_system_prompt",
    "judge_payload",
    "validate_answer",
    "validate_judgment",
]
