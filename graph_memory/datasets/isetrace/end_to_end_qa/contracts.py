from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from graph_memory.trajectories import SourceSpan


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class Condition(str, Enum):
    FLAT_DENSE_FT = "flat_dense_ft"
    PU_DENSE_FT = "pu_dense_ft"
    RESIDUAL_RGCN = "residual_rgcn"
    GOLD_ORACLE = "gold_oracle"
    NO_EVIDENCE = "no_evidence"


class AnswerCorrectness(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NOT_ANSWERED = "not_answered"


class AnswerFaithfulness(str, Enum):
    FAITHFUL = "faithful"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


class AnswerResponse(StrictModel):
    answer: str = Field(min_length=1, max_length=4000)
    abstained: bool

    @model_validator(mode="after")
    def validate_abstention(self) -> AnswerResponse:
        sentinel = "INSUFFICIENT_EVIDENCE"
        if self.abstained != (self.answer == sentinel):
            raise ValueError(
                "abstained must be true exactly when answer is INSUFFICIENT_EVIDENCE"
            )
        return self


class _JudgeLabels(StrictModel):
    correctness: AnswerCorrectness
    faithfulness: AnswerFaithfulness


class JudgeResponse(_JudgeLabels):
    reason: str | None = Field(default=None, min_length=1, max_length=2000)


class EvidenceItem(StrictModel):
    item_id: str = Field(min_length=1)
    text: str
    token_count: PositiveInt
    source_spans: tuple[SourceSpan, ...] = ()


class RankedEvidence(StrictModel):
    node_id: str = Field(min_length=1)
    score: float
    token_count: PositiveInt
    source_spans: tuple[SourceSpan, ...] = ()


class CandidateMetadata(BaseModel):
    token_count: PositiveInt


class CandidateRecord(StrictModel):
    item_id: str = Field(min_length=1)
    text: str
    metadata: CandidateMetadata
    source_spans: tuple[SourceSpan, ...] = ()

    def evidence_item(self) -> EvidenceItem:
        return EvidenceItem(
            item_id=self.item_id,
            text=self.text,
            token_count=self.metadata.token_count,
            source_spans=self.source_spans,
        )


class BenchmarkTask(StrictModel):
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    flat_candidates: tuple[CandidateRecord, ...]
    provenance_candidates: tuple[CandidateRecord, ...]


class LabelRecord(StrictModel):
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    gold_evidence_spans: tuple[SourceSpan, ...]


class RankingRecord(StrictModel):
    task_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    ranked_nodes: tuple[RankedEvidence, ...]


class HumanMetadataRecord(StrictModel):
    query_id: str = Field(min_length=1)
    memory_mode: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)


class ReviewContext(StrictModel):
    handle: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    text: str
    gold_quotes: tuple[str, ...]


class ReviewPacket(StrictModel):
    audit_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    context: tuple[ReviewContext, ...]

    @property
    def gold_quotes(self) -> tuple[str, ...]:
        return tuple(
            quote
            for context in self.context
            for quote in context.gold_quotes
            if quote.strip()
        )


class PreparedRecord(StrictModel):
    record_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    memory_mode: str = Field(min_length=1)
    condition: Condition
    query: str = Field(min_length=1)
    gold_quotes: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[EvidenceItem, ...]
    evidence_token_count: int = Field(ge=0, le=2048)
    full_support: bool

    @model_validator(mode="after")
    def validate_identity_and_budget(self) -> PreparedRecord:
        expected = f"{self.task_id}::{self.condition.value}"
        if self.record_id != expected:
            raise ValueError(f"record_id must be {expected}")
        if self.evidence_token_count != sum(item.token_count for item in self.evidence):
            raise ValueError("evidence_token_count does not match evidence items")
        return self


class AnswerArtifact(StrictModel):
    record_id: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    answer: AnswerResponse
    request_digest: str = Field(min_length=1)
    response_id: str | None
    usage: dict[str, object]
    cached: bool


class JudgmentArtifact(StrictModel):
    record_id: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    judgment: JudgeResponse
    request_digest: str = Field(min_length=1)
    response_id: str | None
    usage: dict[str, object]
    cached: bool


ANSWER_SCHEMA = AnswerResponse.model_json_schema()
JUDGE_SCHEMA = _JudgeLabels.model_json_schema()


__all__ = [
    "ANSWER_SCHEMA",
    "JUDGE_SCHEMA",
    "AnswerArtifact",
    "AnswerCorrectness",
    "AnswerFaithfulness",
    "AnswerResponse",
    "BenchmarkTask",
    "CandidateRecord",
    "Condition",
    "EvidenceItem",
    "HumanMetadataRecord",
    "JudgeResponse",
    "JudgmentArtifact",
    "LabelRecord",
    "PreparedRecord",
    "RankedEvidence",
    "RankingRecord",
    "ReviewPacket",
]
