"""Frozen configuration for the single non-trained EPGM retriever.

``ppr_steiner`` is the reported default. It builds query-conditioned typed
transitions, diffuses Dense relevance with Personalized PageRank, and extracts
a budgeted connected evidence subgraph. ``typed_beam`` and
``dependency_path`` remain reproducibility diagnostics only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Literal, Mapping

from graph_memory.graphs.provenance import ProvenanceEdgeType

EpgmVariant = Literal["ppr_steiner", "typed_beam", "dependency_path"]
EpgmTraversal = Literal["directed", "bidirectional"]
EpgmEdgeScope = Literal["typed", "dependency"]
EpgmPathScore = Literal["type_prior", "feeds_geometric"]
EpgmGating = Literal["none", "schema"]
EpgmFusion = Literal["additive", "stable_insert"]

EPGM_VARIANTS: tuple[EpgmVariant, ...] = (
    "ppr_steiner",
    "typed_beam",
    "dependency_path",
)
RELATION_DESCRIPTION_VERSION = "provenance-relations-v1"
# Legacy diagnostic threshold; the default architecture never applies global
# per-type min-max weights.
WEIGHT_INFORMATIVE_VARIANCE: float = 1e-9

DEFAULT_RELATION_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        ProvenanceEdgeType.CONTAINS.value: "an execution scope contains an agent or artifact",
        ProvenanceEdgeType.INVOKES.value: "an agent or task invokes a tool call",
        ProvenanceEdgeType.RETURNS.value: "a tool call returns an output",
        ProvenanceEdgeType.FEEDS.value: "an output is consumed as context by a downstream tool call",
        ProvenanceEdgeType.GROUNDS.value: "source evidence directly grounds a semantic conclusion",
        ProvenanceEdgeType.PRECEDES.value: "an execution artifact temporally precedes another artifact",
        ProvenanceEdgeType.SUPPORTS.value: "evidence supports a claim decision or answer",
        ProvenanceEdgeType.VERIFIES.value: "an independent verification checks a conclusion",
        ProvenanceEdgeType.DEPENDS_ON.value: "a downstream execution artifact depends on an upstream artifact",
        ProvenanceEdgeType.CONTRADICTS.value: "evidence contradicts or overturns an earlier conclusion",
        ProvenanceEdgeType.INVALIDATES.value: "a later revision invalidates an earlier claim or decision",
        ProvenanceEdgeType.AFFECTS.value: "a revision affects a downstream execution artifact",
        ProvenanceEdgeType.SUPERSEDES.value: "a newer conclusion supersedes an older conclusion",
    }
)

DEFAULT_EDGE_PRIORS: Mapping[str, float] = MappingProxyType(
    {
        ProvenanceEdgeType.GROUNDS.value: 1.0,
        ProvenanceEdgeType.SUPPORTS.value: 0.9,
        ProvenanceEdgeType.VERIFIES.value: 0.9,
        ProvenanceEdgeType.CONTRADICTS.value: 0.85,
        ProvenanceEdgeType.INVALIDATES.value: 0.85,
        ProvenanceEdgeType.SUPERSEDES.value: 0.85,
        ProvenanceEdgeType.DEPENDS_ON.value: 0.7,
        ProvenanceEdgeType.RETURNS.value: 0.6,
        ProvenanceEdgeType.FEEDS.value: 0.6,
        ProvenanceEdgeType.AFFECTS.value: 0.6,
        ProvenanceEdgeType.PRECEDES.value: 0.4,
        ProvenanceEdgeType.INVOKES.value: 0.4,
        ProvenanceEdgeType.CONTAINS.value: 0.0,
    }
)

DEPENDENCY_EDGE_TYPES: frozenset[ProvenanceEdgeType] = frozenset(
    {
        ProvenanceEdgeType.RETURNS,
        ProvenanceEdgeType.FEEDS,
        ProvenanceEdgeType.GROUNDS,
        ProvenanceEdgeType.SUPPORTS,
        ProvenanceEdgeType.DEPENDS_ON,
    }
)
REVISION_EDGE_TYPES: frozenset[ProvenanceEdgeType] = frozenset(
    {ProvenanceEdgeType.INVALIDATES, ProvenanceEdgeType.SUPERSEDES}
)
INVALID_LIFECYCLE_STATES: frozenset[str] = frozenset(
    {"invalid", "invalidated", "superseded", "obsolete"}
)
NON_TRAVERSABLE_EDGE_TYPES: frozenset[str] = frozenset(
    {ProvenanceEdgeType.CONTAINS.value}
)
HUB_NODE_TYPES: frozenset[str] = frozenset({"task", "agent"})


@dataclass(frozen=True)
class EpgmRetrieverConfig:
    """No-train EPGM configuration; every behavior field is cache-relevant."""

    variant: EpgmVariant = "ppr_steiner"

    # Legacy bounded-path diagnostics.
    seed_top_s: int = 8
    beam_width: int = 32
    max_hops: int = 4
    max_path_expansions: int = 4000
    traversal: EpgmTraversal = "bidirectional"
    edge_scope: EpgmEdgeScope = "typed"
    path_score: EpgmPathScore = "type_prior"
    gating: EpgmGating = "none"
    fusion: EpgmFusion = "additive"
    graph_weight: float = 0.25
    hop_decay: float = 0.6
    reverse_edge_factor: float = 0.85
    min_edge_prior: float = 0.05
    edge_priors: Mapping[str, float] = field(default=DEFAULT_EDGE_PRIORS)
    # Global path multiplication is a measured negative diagnostic and is off.
    weight_aware: bool = False
    min_effective_weight: float = 0.1
    hop_penalty: float = 0.0
    min_path_confidence: float = 0.0
    preserve_dense_top_n: int = 0

    # Query-conditioned typed transition model.
    relation_description_version: str = RELATION_DESCRIPTION_VERSION
    relation_descriptions: Mapping[str, str] = field(
        default=DEFAULT_RELATION_DESCRIPTIONS
    )
    relation_temperature: float = 0.2
    hub_degree_exponent: float = 0.5

    # Personalized PageRank.
    ppr_alpha: float = 0.85
    ppr_tolerance: float = 1e-9
    ppr_max_iterations: int = 200

    # Budgeted connected-subgraph objective.
    dense_prize_weight: float = 0.55
    ppr_prize_weight: float = 0.45
    candidate_inclusion_cost: float = 0.2
    selection_edge_cost_weight: float = 0.08
    connector_hop_cost: float = 0.05
    selection_min_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in EPGM_VARIANTS:
            raise ValueError(f"unknown EPGM variant={self.variant!r}.")
        for name in (
            "seed_top_s",
            "beam_width",
            "max_hops",
            "max_path_expansions",
            "ppr_max_iterations",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.preserve_dense_top_n < 0:
            raise ValueError("preserve_dense_top_n must be non-negative.")
        if not self.relation_description_version.strip():
            raise ValueError("relation_description_version must be non-empty.")
        if set(self.relation_descriptions) != {
            edge_type.value for edge_type in ProvenanceEdgeType
        }:
            raise ValueError("relation_descriptions must cover every edge type exactly.")
        if any(not description.strip() for description in self.relation_descriptions.values()):
            raise ValueError("relation descriptions must be non-empty.")
        for name in (
            "graph_weight",
            "hop_decay",
            "reverse_edge_factor",
            "hop_penalty",
            "min_path_confidence",
            "relation_temperature",
            "hub_degree_exponent",
            "ppr_tolerance",
            "dense_prize_weight",
            "ppr_prize_weight",
            "candidate_inclusion_cost",
            "selection_edge_cost_weight",
            "connector_hop_cost",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.relation_temperature <= 0.0:
            raise ValueError("relation_temperature must be positive.")
        if not 0.0 < self.ppr_alpha < 1.0:
            raise ValueError("ppr_alpha must be in (0, 1).")
        if not math.isfinite(self.selection_min_gain):
            raise ValueError("selection_min_gain must be finite.")
        if self.dense_prize_weight + self.ppr_prize_weight <= 0.0:
            raise ValueError("at least one candidate prize weight must be positive.")
        object.__setattr__(
            self, "edge_priors", MappingProxyType(dict(self.edge_priors))
        )
        object.__setattr__(
            self,
            "relation_descriptions",
            MappingProxyType(dict(self.relation_descriptions)),
        )
        if not 0.0 <= self.min_edge_prior <= 1.0:
            raise ValueError("min_edge_prior must be in [0, 1].")
        if not 0.0 <= self.min_effective_weight <= 1.0:
            raise ValueError("min_effective_weight must be in [0, 1].")

    @classmethod
    def for_variant(
        cls, variant: EpgmVariant, **overrides: object
    ) -> "EpgmRetrieverConfig":
        if variant not in EPGM_VARIANTS:
            raise ValueError(f"unknown EPGM variant={variant!r}.")
        if variant == "ppr_steiner":
            base = cls()
        elif variant == "typed_beam":
            base = _TYPED_BEAM_PRESET
        else:
            base = _DEPENDENCY_PATH_PRESET
        return base if not overrides else replace(base, **overrides)  # type: ignore[arg-type]

    def edge_prior(self, edge_type: str) -> float:
        return self.edge_priors.get(edge_type, self.min_edge_prior)

    def cache_fingerprint(self) -> str:
        """Digest every frozen behavior field for Prefect ranking identity."""

        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = (
                dict(sorted(value.items())) if isinstance(value, Mapping) else value
            )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


_TYPED_BEAM_PRESET = EpgmRetrieverConfig(
    variant="typed_beam",
    weight_aware=False,
)
_DEPENDENCY_PATH_PRESET = EpgmRetrieverConfig(
    variant="dependency_path",
    seed_top_s=5,
    beam_width=8,
    max_hops=2,
    max_path_expansions=256,
    traversal="directed",
    edge_scope="dependency",
    path_score="feeds_geometric",
    gating="schema",
    fusion="stable_insert",
    hop_penalty=0.04,
    min_path_confidence=0.2,
    preserve_dense_top_n=2,
    weight_aware=False,
)


__all__ = [
    "DEFAULT_EDGE_PRIORS",
    "DEFAULT_RELATION_DESCRIPTIONS",
    "DEPENDENCY_EDGE_TYPES",
    "EPGM_VARIANTS",
    "EpgmEdgeScope",
    "EpgmFusion",
    "EpgmGating",
    "EpgmPathScore",
    "EpgmRetrieverConfig",
    "EpgmTraversal",
    "EpgmVariant",
    "HUB_NODE_TYPES",
    "INVALID_LIFECYCLE_STATES",
    "NON_TRAVERSABLE_EDGE_TYPES",
    "RELATION_DESCRIPTION_VERSION",
    "REVISION_EDGE_TYPES",
    "WEIGHT_INFORMATIVE_VARIANCE",
]
