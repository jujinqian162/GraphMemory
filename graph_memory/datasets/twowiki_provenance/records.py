from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from graph_memory.contracts.common import JsonValue

TWOWIKI_PROVENANCE_SCHEMA_VERSION = 3


class ProvenanceCandidateRecord(TypedDict):
    output_id: str
    call_id: str
    title: str
    sentence_index: int
    position: int
    text: str


class ProvenanceBindingRecord(TypedDict):
    output_field: str
    input_parameter: str
    binding_value_hash: str
    binding_kind: str


class ProvenanceFeedEdgeMetadataRecord(TypedDict):
    synthetic: bool
    semantic_scorer: str
    scorer_identity: str
    query_template_version: str
    semantic_rank: int
    semantic_score: float
    normalized_score: float
    source_probability: float
    calibrated_weight: float
    branch_role: str
    rank_bucket: str


class ProvenanceNodeRecord(TypedDict):
    node_id: str
    node_type: str
    text: str
    metadata: dict[str, JsonValue]


class ProvenanceEdgeRecord(TypedDict):
    source: str
    target: str
    edge_type: str
    binding: ProvenanceBindingRecord | None
    weight: float
    metadata: dict[str, JsonValue]


class ProvenanceGraphRecord(TypedDict):
    task_id: str
    nodes: list[ProvenanceNodeRecord]
    edges: list[ProvenanceEdgeRecord]


class TwoWikiProvenanceRankingRecord(TypedDict):
    task_id: str
    question: str
    question_type: str
    candidates: list[ProvenanceCandidateRecord]
    graph: ProvenanceGraphRecord
    metadata: dict[str, JsonValue]


class TwoWikiProvenanceLabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_evidence_output_ids: list[str]
    gold_dependency_edges: list[list[str]]
    metadata: dict[str, JsonValue]


class TwoWikiProvenanceRawRecord(TypedDict):
    schema_version: int
    ranking: TwoWikiProvenanceRankingRecord
    label: TwoWikiProvenanceLabelRecord


@dataclass(frozen=True)
class ConvertedTwoWikiProvenanceExample:
    raw_record: TwoWikiProvenanceRawRecord


@dataclass(frozen=True)
class TwoWikiProvenanceConversionResult:
    records: list[TwoWikiProvenanceRawRecord]
    rejected_reason_counts: dict[str, int]


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
    "TwoWikiProvenanceRankingRecord",
    "TwoWikiProvenanceRawRecord",
]
