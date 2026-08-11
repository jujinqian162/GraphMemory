from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.provenance.contracts import NamespacedIdentifier

MotifType: TypeAlias = Literal[
    "call_result",
    "value_flow",
    "artifact_lifecycle",
    "multi_hop_flow",
    "multi_source_join",
]
QueryIntent: TypeAlias = Literal[
    "call_result",
    "upstream_source",
    "downstream_result",
    "complete_chain",
    "artifact_origin",
    "artifact_use",
    "contributing_sources",
]


class LogicalDependency(DomainModel):
    source_output_id: NonEmptyStr
    target_output_id: NonEmptyStr
    relation: NamespacedIdentifier

    @model_validator(mode="after")
    def _reject_self_edge(self) -> "LogicalDependency":
        if self.source_output_id == self.target_output_id:
            raise ValueError("logical dependency cannot be a self edge")
        return self


class MotifAuthoringTarget(DomainModel):
    """Internal target used only to select source material for v7 authoring."""

    query_intent: QueryIntent
    focus_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    participant_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_target(self) -> "MotifAuthoringTarget":
        if len(set(self.focus_output_ids)) != len(self.focus_output_ids):
            raise ValueError("focus output IDs must be unique")
        if len(set(self.participant_output_ids)) != len(self.participant_output_ids):
            raise ValueError("participant output IDs must be unique")
        if not set(self.focus_output_ids).issubset(self.participant_output_ids):
            raise ValueError(
                "focus output IDs must be included in participant output IDs"
            )
        return self


class MotifSpec(DomainModel):
    schema_version: Literal[1] = 1
    motif_id: NonEmptyStr
    motif_type: MotifType
    graph_id: NonEmptyStr
    participant_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    dependencies: tuple[LogicalDependency, ...]
    targets: tuple[MotifAuthoringTarget, ...] = Field(min_length=1)
    hidden_metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_motif(self) -> "MotifSpec":
        participants = set(self.participant_output_ids)
        if len(participants) != len(self.participant_output_ids):
            raise ValueError("motif participant output IDs must be unique")
        target_intents = [target.query_intent for target in self.targets]
        if len(target_intents) != len(set(target_intents)):
            raise ValueError("motif query intents must be unique")
        for dependency in self.dependencies:
            if {
                dependency.source_output_id,
                dependency.target_output_id,
            } - participants:
                raise ValueError("motif dependency endpoint is not a participant")
        for target in self.targets:
            if set(target.participant_output_ids) - participants:
                raise ValueError("motif target participant is not in the motif")
        return self

    def target_for(self, query_intent: QueryIntent) -> MotifAuthoringTarget:
        for target in self.targets:
            if target.query_intent == query_intent:
                return target
        raise ValueError(
            f"motif={self.motif_id} does not support query_intent={query_intent}"
        )


def authoring_target_key(
    motif: MotifSpec,
    target: MotifAuthoringTarget,
    *,
    seed: int,
) -> str:
    identity = f"{seed}\0{motif.motif_id}\0{target.query_intent}"
    return f"T-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"


def motif_target_source_ids(target: MotifAuthoringTarget) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*target.participant_output_ids, *target.focus_output_ids))
    )


def motif_id(
    motif_type: str,
    participants: Iterable[str],
    dependencies: Iterable[LogicalDependency],
) -> str:
    payload = {
        "motif_type": motif_type,
        "participants": list(participants),
        "dependencies": [
            dependency.model_dump(mode="json") for dependency in dependencies
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"motif:{motif_type}:{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"


__all__ = [
    "LogicalDependency",
    "MotifAuthoringTarget",
    "MotifSpec",
    "MotifType",
    "QueryIntent",
    "authoring_target_key",
    "motif_id",
    "motif_target_source_ids",
]
