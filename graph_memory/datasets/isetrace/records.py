from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr, NonNegativeInt


class ISETraceIntent(DomainModel):
    intent_id: NonEmptyStr
    natural_language_intent: NonEmptyStr
    task_type: NonEmptyStr | None = None


class ISETraceFunctionDefinition(DomainModel):
    name: NonEmptyStr
    description: str = ""
    parameters: str


class ISETraceToolDefinition(DomainModel):
    type: Literal["function"]
    function: ISETraceFunctionDefinition


class ISETraceFunctionCall(DomainModel):
    name: NonEmptyStr
    arguments: str


class ISETraceToolCall(DomainModel):
    id: NonEmptyStr
    type: Literal["function"]
    function: ISETraceFunctionCall


class ISETraceSystemMessage(DomainModel):
    role: Literal["system"]
    content: str


class ISETraceUserMessage(DomainModel):
    role: Literal["user"]
    content: str


class ISETraceAssistantMessage(DomainModel):
    role: Literal["assistant"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ISETraceToolCall, ...] = ()


class ISETraceToolMessage(DomainModel):
    role: Literal["tool"]
    name: NonEmptyStr
    tool_call_id: NonEmptyStr
    content: str
    success: bool | None = None


ISETraceMessage: TypeAlias = Annotated[
    ISETraceSystemMessage
    | ISETraceUserMessage
    | ISETraceAssistantMessage
    | ISETraceToolMessage,
    Field(discriminator="role"),
]


class ISETraceRecord(DomainModel):
    status: NonEmptyStr
    session_id: NonEmptyStr
    source_intent_count: NonNegativeInt
    source_intent_ids: tuple[NonEmptyStr, ...]
    source_intents: tuple[ISETraceIntent, ...] = Field(min_length=1)
    session_finalized_by_intent_id: NonEmptyStr
    total_steps: NonNegativeInt
    enable_thinking: bool
    messages: tuple[ISETraceMessage, ...] = Field(min_length=1)
    tools: tuple[ISETraceToolDefinition, ...]
    final_output: str
    intent_id: NonEmptyStr
    metadata: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_source_intents(self) -> "ISETraceRecord":
        embedded_ids = tuple(intent.intent_id for intent in self.source_intents)
        if self.source_intent_count != len(self.source_intent_ids):
            raise ValueError("source_intent_count does not match source_intent_ids")
        if self.source_intent_ids != embedded_ids:
            raise ValueError("source_intent_ids do not match embedded source_intents")
        if len(set(self.source_intent_ids)) != len(self.source_intent_ids):
            raise ValueError("source_intent_ids must be unique")
        return self


__all__ = [
    "ISETraceAssistantMessage",
    "ISETraceFunctionCall",
    "ISETraceFunctionDefinition",
    "ISETraceIntent",
    "ISETraceMessage",
    "ISETraceRecord",
    "ISETraceSystemMessage",
    "ISETraceToolCall",
    "ISETraceToolDefinition",
    "ISETraceToolMessage",
    "ISETraceUserMessage",
]
