from __future__ import annotations

import re
from collections import defaultdict
from string import Formatter
from typing import Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr, PositiveInt
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


class MotifQueryTarget(DomainModel):
    query_intent: QueryIntent
    answer_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    support_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    safe_slots: dict[NonEmptyStr, NonEmptyStr]

    @model_validator(mode="after")
    def _validate_target(self) -> "MotifQueryTarget":
        if len(set(self.answer_output_ids)) != len(self.answer_output_ids):
            raise ValueError("answer output IDs must be unique")
        if len(set(self.support_output_ids)) != len(self.support_output_ids):
            raise ValueError("support output IDs must be unique")
        if not set(self.answer_output_ids).issubset(self.support_output_ids):
            raise ValueError("answer output IDs must be included in support output IDs")
        return self


class MotifSpec(DomainModel):
    schema_version: Literal[1] = 1
    motif_id: NonEmptyStr
    motif_type: MotifType
    graph_id: NonEmptyStr
    participant_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    dependencies: tuple[LogicalDependency, ...]
    targets: tuple[MotifQueryTarget, ...] = Field(min_length=1)
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
            if set(target.support_output_ids) - participants:
                raise ValueError("motif target support is not a participant")
        return self

    def target_for(self, query_intent: QueryIntent) -> MotifQueryTarget:
        for target in self.targets:
            if target.query_intent == query_intent:
                return target
        raise ValueError(
            f"motif={self.motif_id} does not support query_intent={query_intent}"
        )


class QueryTemplate(DomainModel):
    template_id: NonEmptyStr
    catalog_version: NonEmptyStr
    motif_type: MotifType
    query_intent: QueryIntent
    style_tags: tuple[NonEmptyStr, ...] = Field(min_length=1)
    text: NonEmptyStr
    required_slots: tuple[NonEmptyStr, ...]


class SyntheticQueryRecord(DomainModel):
    query_id: NonEmptyStr
    graph_id: NonEmptyStr
    query_text: NonEmptyStr


class SyntheticQueryLabel(DomainModel):
    query_id: NonEmptyStr
    motif_id: NonEmptyStr
    motif_type: MotifType
    query_intent: QueryIntent
    template_id: NonEmptyStr
    template_catalog_version: NonEmptyStr
    style_tags: tuple[NonEmptyStr, ...]
    answer_output_ids: tuple[NonEmptyStr, ...]
    support_output_ids: tuple[NonEmptyStr, ...]
    dependencies: tuple[LogicalDependency, ...]
    generation_seed: int


class SyntheticQueryExample(DomainModel):
    query: SyntheticQueryRecord
    label: SyntheticQueryLabel

    @model_validator(mode="after")
    def _validate_alignment(self) -> "SyntheticQueryExample":
        if self.query.query_id != self.label.query_id:
            raise ValueError("synthetic query and label IDs must align")
        forbidden = {
            self.label.motif_id,
            *self.label.answer_output_ids,
            *self.label.support_output_ids,
        }
        if any(value and value in self.query.query_text for value in forbidden):
            raise ValueError("synthetic query text leaks an internal motif or node ID")
        return self


class TemplateCatalog(DomainModel):
    version: NonEmptyStr
    minimum_templates_per_pair: PositiveInt = 6
    minimum_styles_per_pair: PositiveInt = 3
    templates: tuple[QueryTemplate, ...]

    @model_validator(mode="after")
    def _validate_catalog(self) -> "TemplateCatalog":
        template_ids = [template.template_id for template in self.templates]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("query template IDs must be unique")
        grouped: dict[tuple[str, str], list[QueryTemplate]] = defaultdict(list)
        for template in self.templates:
            if template.catalog_version != self.version:
                raise ValueError(
                    f"template={template.template_id} catalog version mismatch"
                )
            fields = {
                field_name
                for _, field_name, _, _ in Formatter().parse(template.text)
                if field_name is not None
            }
            if fields != set(template.required_slots):
                raise ValueError(
                    f"template={template.template_id} placeholder/slot mismatch"
                )
            grouped[(template.motif_type, template.query_intent)].append(template)
        for pair, templates in grouped.items():
            if len(templates) < self.minimum_templates_per_pair:
                raise ValueError(f"template pair={pair} has too few templates")
            normalized = {
                re.sub(r"\s+", " ", template.text.strip().lower())
                for template in templates
            }
            if len(normalized) != len(templates):
                raise ValueError(f"template pair={pair} contains duplicate text")
            styles = {tag for template in templates for tag in template.style_tags}
            if len(styles) < self.minimum_styles_per_pair:
                raise ValueError(f"template pair={pair} has too few style tags")
        return self


__all__ = [
    "LogicalDependency",
    "MotifQueryTarget",
    "MotifSpec",
    "MotifType",
    "QueryIntent",
    "QueryTemplate",
    "SyntheticQueryExample",
    "SyntheticQueryLabel",
    "SyntheticQueryRecord",
    "TemplateCatalog",
]
