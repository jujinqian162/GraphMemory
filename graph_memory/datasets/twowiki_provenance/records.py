from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import Field, StrictBool, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveInt,
    reject_label_fields,
)
from graph_memory.datasets.twowiki_provenance.scoring import (
    ProvenanceGraphConstructionConfig,
)
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
    binding_matches_endpoints,
)

TWOWIKI_PROVENANCE_SCHEMA_VERSION = 3

ProvenanceBindingRecord: TypeAlias = FieldBinding
ProvenanceNodeRecord: TypeAlias = ExecutionProvenanceNode
ProvenanceEdgeRecord: TypeAlias = ExecutionProvenanceEdge
ProvenanceGraphRecord: TypeAlias = ExecutionProvenanceGraph


class ProvenanceCandidateRecord(DomainModel):
    output_id: NonEmptyStr
    call_id: NonEmptyStr
    title: NonEmptyStr
    sentence_index: NonNegativeInt
    position: NonNegativeInt
    text: NonEmptyStr


class TwoWikiProvenanceRankingMetadata(DomainModel):
    dataset: Literal["twowiki_provenance"]
    source_dataset: Literal["2wiki"]
    source_raw_id: NonEmptyStr
    synthetic_execution_graph: Literal[True]
    schema_version: Literal[3]
    graph_construction: ProvenanceGraphConstructionConfig


class TwoWikiProvenanceLabelMetadata(DomainModel):
    dataset: Literal["twowiki_provenance"]
    question_type: NonEmptyStr
    source_path_label_source: NonEmptyStr
    gold_edge_semantic_rank: PositiveInt
    gold_edge_rank_bucket: NonEmptyStr
    gold_edge_branch_role: Literal["semantic_head", "rank_banded_branch"]
    gold_edge_is_head: StrictBool
    gold_edge_calibrated_weight: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def _validate_head_flag(self) -> "TwoWikiProvenanceLabelMetadata":
        if self.gold_edge_is_head != (
            self.gold_edge_branch_role == "semantic_head"
        ):
            raise ValueError("gold edge head flag must match branch role")
        return self


class ProvenanceFeedEdgeMetadataRecord(DomainModel):
    synthetic: Literal[True]
    semantic_scorer: Literal["bm25", "dense", "hybrid"]
    scorer_identity: NonEmptyStr
    query_template_version: Literal["question_source_v1"]
    semantic_rank: PositiveInt
    semantic_score: FiniteFloat
    normalized_score: FiniteFloat = Field(ge=0.0, le=1.0)
    source_probability: FiniteFloat = Field(ge=0.0, le=1.0)
    calibrated_weight: NonNegativeFiniteFloat
    branch_role: Literal["semantic_head", "rank_banded_branch"]
    rank_bucket: Literal["head", "near", "mid", "tail"]


class TwoWikiProvenanceRankingRecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    question_type: NonEmptyStr
    candidates: tuple[ProvenanceCandidateRecord, ...] = Field(min_length=4)
    graph: ExecutionProvenanceGraph
    metadata: TwoWikiProvenanceRankingMetadata

    @model_validator(mode="after")
    def _validate_ranking(self) -> "TwoWikiProvenanceRankingRecord":
        if not self.task_id.startswith("2wiki_provenance_"):
            raise ValueError("2Wiki provenance task_id has an invalid prefix")
        reject_label_fields(
            self.model_dump(mode="python", exclude_none=True),
            path="2Wiki provenance ranking record",
        )
        output_ids: list[str] = []
        call_ids: list[str] = []
        for expected, candidate in enumerate(self.candidates):
            output_ids.append(candidate.output_id)
            call_ids.append(candidate.call_id)
            if candidate.position != expected:
                raise ValueError("provenance candidate position is not contiguous")
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("candidate output IDs must be unique")
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("candidate call IDs must be unique")
        if self.graph.task_id != self.task_id:
            raise ValueError("provenance graph task_id must match ranking task_id")

        candidate_pairs = {
            candidate.output_id: candidate.call_id for candidate in self.candidates
        }
        node_by_id = {node.node_id: node for node in self.graph.nodes}
        output_nodes = {
            node.node_id
            for node in self.graph.nodes
            if node.node_type is ProvenanceNodeType.TOOL_OUTPUT
        }
        if output_nodes != set(candidate_pairs):
            raise ValueError("candidate outputs must match tool-output graph nodes")
        for output_id, call_id in candidate_pairs.items():
            call = node_by_id.get(call_id)
            if call is None or call.node_type is not ProvenanceNodeType.TOOL_CALL:
                raise ValueError(f"missing tool-call node={call_id}")
            if not any(
                edge.source == call_id
                and edge.target == output_id
                and edge.edge_type is ProvenanceEdgeType.RETURNS
                for edge in self.graph.edges
            ):
                raise ValueError(f"missing returns pair={call_id}->{output_id}")

        feeds_by_source: defaultdict[str, list[ExecutionProvenanceEdge]] = (
            defaultdict(list)
        )
        for edge in self.graph.edges:
            if edge.edge_type is ProvenanceEdgeType.FEEDS:
                feeds_by_source[edge.source].append(edge)
            elif edge.weight != 1.0:
                raise ValueError("non-feed edge weight must be 1.0")
        if set(feeds_by_source) != set(candidate_pairs) or {
            len(edges) for edges in feeds_by_source.values()
        } != {2}:
            raise ValueError("output branch degree must be exactly two")
        _validate_feed_confidence(
            feeds_by_source,
            construction=self.metadata.graph_construction,
        )
        inconsistent_bindings = [
            f"{edge.source}->{edge.target}"
            for edge in self.graph.edges
            if edge.edge_type is ProvenanceEdgeType.FEEDS
            and not binding_matches_endpoints(edge, node_by_id)
        ]
        if inconsistent_bindings:
            raise ValueError(
                f"inconsistent feed bindings={sorted(inconsistent_bindings)}"
            )
        return self

    @property
    def candidate_by_output(self) -> dict[str, ProvenanceCandidateRecord]:
        return {candidate.output_id: candidate for candidate in self.candidates}


class TwoWikiProvenanceLabelRecord(DomainModel):
    task_id: NonEmptyStr
    gold_answer: NonEmptyStr
    gold_evidence_output_ids: tuple[NonEmptyStr, NonEmptyStr]
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: TwoWikiProvenanceLabelMetadata

    @model_validator(mode="after")
    def _validate_label(self) -> "TwoWikiProvenanceLabelRecord":
        source, target = self.gold_evidence_output_ids
        if source == target:
            raise ValueError("gold evidence outputs must be distinct")
        if self.gold_dependency_edges != ((source, target),):
            raise ValueError("gold dependency edge must be the ordered gold outputs")
        return self


class TwoWikiProvenanceRawRecord(DomainModel):
    schema_version: Literal[3]
    ranking: TwoWikiProvenanceRankingRecord
    label: TwoWikiProvenanceLabelRecord

    @model_validator(mode="after")
    def _validate_join(self) -> "TwoWikiProvenanceRawRecord":
        if self.ranking.task_id != self.label.task_id:
            raise ValueError("provenance ranking and label task IDs must match")
        source, target = self.label.gold_evidence_output_ids
        candidates = self.ranking.candidate_by_output
        if source not in candidates or target not in candidates:
            raise ValueError("gold output is missing from candidates")
        target_call = candidates[target].call_id
        has_feeds = any(
            edge.source == source
            and edge.target == target_call
            and edge.edge_type is ProvenanceEdgeType.FEEDS
            and edge.binding is not None
            for edge in self.ranking.graph.edges
        )
        has_returns = any(
            edge.source == target_call
            and edge.target == target
            and edge.edge_type is ProvenanceEdgeType.RETURNS
            for edge in self.ranking.graph.edges
        )
        if not has_feeds or not has_returns:
            raise ValueError("gold dependency path is missing from provenance graph")
        return self


class TwoWikiProvenancePreparedSplit(DomainModel):
    records: tuple[TwoWikiProvenanceRawRecord, ...]

    @model_validator(mode="after")
    def _validate_task_ids(self) -> "TwoWikiProvenancePreparedSplit":
        task_ids = [record.ranking.task_id for record in self.records]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("2Wiki provenance task IDs must be unique")
        return self


@dataclass(frozen=True)
class ConvertedTwoWikiProvenanceExample:
    raw_record: TwoWikiProvenanceRawRecord


@dataclass(frozen=True)
class TwoWikiProvenanceConversionResult:
    records: list[TwoWikiProvenanceRawRecord]
    rejected_reason_counts: dict[str, int]


def _validate_feed_confidence(
    feeds_by_source: dict[str, list[ExecutionProvenanceEdge]],
    *,
    construction: ProvenanceGraphConstructionConfig,
) -> None:
    floor = construction.weight_floor
    expected_mass = 2.0 * floor + (1.0 - floor)
    for source, edges in feeds_by_source.items():
        roles: Counter[str] = Counter()
        probability_sum = 0.0
        weight_sum = 0.0
        for edge in edges:
            metadata = ProvenanceFeedEdgeMetadataRecord.model_validate(edge.metadata)
            if (
                metadata.semantic_scorer != construction.strategy
                or metadata.scorer_identity != construction.scorer_identity
                or metadata.query_template_version
                != construction.query_template_version
            ):
                raise ValueError(f"source={source} feed metadata/config mismatch")
            if not floor <= edge.weight <= 1.0 or not math.isclose(
                edge.weight,
                metadata.calibrated_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"source={source} feed weight mismatch")
            if metadata.branch_role == "semantic_head":
                valid_role = (
                    metadata.semantic_rank == 1 and metadata.rank_bucket == "head"
                )
            elif metadata.rank_bucket != "head":
                lower, upper = construction.bucket_bounds(metadata.rank_bucket)
                valid_role = metadata.semantic_rank >= lower and (
                    upper is None or metadata.semantic_rank <= upper
                )
            else:
                valid_role = False
            if not valid_role:
                raise ValueError(f"source={source} rank bucket does not match rank")
            roles[metadata.branch_role] += 1
            probability_sum += metadata.source_probability
            weight_sum += edge.weight
        if roles != {"semantic_head": 1, "rank_banded_branch": 1}:
            raise ValueError(f"source={source} branch roles are invalid")
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"source={source} source probability mass is invalid")
        if not math.isclose(weight_sum, expected_mass, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"source={source} feed mass is invalid")


__all__ = [
    "ConvertedTwoWikiProvenanceExample",
    "ProvenanceBindingRecord",
    "ProvenanceCandidateRecord",
    "ProvenanceEdgeRecord",
    "ProvenanceFeedEdgeMetadataRecord",
    "ProvenanceGraphRecord",
    "ProvenanceNodeRecord",
    "TWOWIKI_PROVENANCE_SCHEMA_VERSION",
    "TwoWikiProvenanceConversionResult",
    "TwoWikiProvenanceLabelRecord",
    "TwoWikiProvenancePreparedSplit",
    "TwoWikiProvenanceRankingRecord",
    "TwoWikiProvenanceRawRecord",
]
