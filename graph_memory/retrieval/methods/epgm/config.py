"""Configuration for the single non-trained EPGM provenance retriever.

There is exactly one non-trained EPGM implementation. Its behaviour is
selected by explicit strategy axes, and two frozen presets are exposed as
``EpgmVariant``:

``typed_beam``
    The default. Bidirectional typed beam search over the full EPGM schema
    with additive dense/graph fusion and no schema gating. This is the
    variant reported as "EPGM (non-trained)".

``dependency_path``
    A *restriction* of the same search: directed traversal, dependency edges
    only, hard schema gating (exactly one ``feeds`` + one ``returns``, valid
    binding, no invalidated node), and stable insertion that preserves the
    dense score multiset. Reported only as an ablation of the default.

Every axis below is shared by both presets, so ``dependency_path`` is a
configuration of ``typed_beam`` rather than a second method.

Recorded edge weights are consumed through a single ``w_eff`` factor that is
normalised per edge type and collapses to exactly ``1.0`` when the recorded
weights have no variance. A scorer-calibrated synthetic graph therefore gets
full weight sensitivity while a recorded agent trace that stores a constant
placeholder weight falls back to pure type priors, with no dataset-specific
branch anywhere in the method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Mapping

from graph_memory.graphs.provenance import ProvenanceEdgeType

EpgmVariant = Literal["typed_beam", "dependency_path"]
EpgmTraversal = Literal["directed", "bidirectional"]
EpgmEdgeScope = Literal["typed", "dependency"]
EpgmPathScore = Literal["type_prior", "feeds_geometric"]
EpgmGating = Literal["none", "schema"]
EpgmFusion = Literal["additive", "stable_insert"]

# Minimum per-edge-type weight variance for recorded weights to count as a
# real measurement rather than a constant placeholder. Synthetic graphs whose
# weights come from a semantic scorer clear this easily; a recorded agent trace
# that fills `weight=1.0` everywhere does not.
WEIGHT_INFORMATIVE_VARIANCE: float = 1e-9

EPGM_VARIANTS: tuple[EpgmVariant, ...] = ("typed_beam", "dependency_path")

# Per-edge-type semantic prior. Higher = the edge carries stronger evidentiary
# signal for propagating relevance between two memory nodes. `contains` is
# intentionally 0.0 and additionally treated as non-traversable (see
# NON_TRAVERSABLE_EDGE_TYPES) so the task super-node cannot act as a hub.
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

# Directed data-dependency edges. `edge_scope="dependency"` restricts the walk
# to these, which is what makes the dependency-path restriction blind to audit
# structure (contradicts/invalidates/verifies).
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

# `contains` links the task to (almost) every node; traversing it turns the
# task into a hub that connects arbitrary pairs, so it is never walked.
NON_TRAVERSABLE_EDGE_TYPES: frozenset[str] = frozenset(
    {ProvenanceEdgeType.CONTAINS.value}
)

# task/agent nodes may be a path endpoint (so an agent-identity question can be
# answered), but the search never expands *through* them, again to avoid hubs.
HUB_NODE_TYPES: frozenset[str] = frozenset({"task", "agent"})


@dataclass(frozen=True)
class EpgmRetrieverConfig:
    """No-train configuration for the EPGM provenance retriever.

    Dense retrieval seeds the ranking, then bounded typed path search reranks
    candidates. Nothing here is learned. Use :meth:`for_variant` to obtain a
    frozen preset instead of hand-assembling the axes.
    """

    variant: EpgmVariant = "typed_beam"

    # search budget
    seed_top_s: int = 8
    beam_width: int = 32
    max_hops: int = 4
    max_path_expansions: int = 4000

    # strategy axes
    traversal: EpgmTraversal = "bidirectional"
    edge_scope: EpgmEdgeScope = "typed"
    path_score: EpgmPathScore = "type_prior"
    gating: EpgmGating = "none"
    fusion: EpgmFusion = "additive"

    # type-prior scoring / additive fusion
    graph_weight: float = 0.25
    hop_decay: float = 0.6
    reverse_edge_factor: float = 0.85
    min_edge_prior: float = 0.05
    edge_priors: Mapping[str, float] = field(default=DEFAULT_EDGE_PRIORS)

    # Recorded-weight sensitivity. When True, a step's prior is multiplied by
    # the edge's effective recorded weight w_eff. w_eff is computed per edge
    # type and collapses to exactly 1.0 whenever the recorded weights carry no
    # information (zero variance), so this axis is an identity operation on
    # traces that fill a constant placeholder weight. No dataset switch.
    weight_aware: bool = True
    min_effective_weight: float = 0.1

    # feeds-geometric scoring / stable-insert fusion
    hop_penalty: float = 0.0
    min_path_confidence: float = 0.0
    preserve_dense_top_n: int = 0

    def __post_init__(self) -> None:
        if self.variant not in EPGM_VARIANTS:
            raise ValueError(f"unknown EPGM variant={self.variant!r}.")
        for name in (
            "seed_top_s",
            "beam_width",
            "max_hops",
            "max_path_expansions",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.preserve_dense_top_n < 0:
            raise ValueError("preserve_dense_top_n must be non-negative.")
        for name in (
            "graph_weight",
            "hop_decay",
            "reverse_edge_factor",
            "hop_penalty",
            "min_path_confidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0.0 <= self.min_edge_prior <= 1.0:
            raise ValueError("min_edge_prior must be in [0, 1].")
        if not 0.0 <= self.min_effective_weight <= 1.0:
            raise ValueError("min_effective_weight must be in [0, 1].")

    @classmethod
    def for_variant(cls, variant: EpgmVariant, **overrides: object) -> "EpgmRetrieverConfig":
        if variant not in EPGM_VARIANTS:
            raise ValueError(f"unknown EPGM variant={variant!r}.")
        base = cls() if variant == "typed_beam" else _DEPENDENCY_PATH_PRESET
        if not overrides:
            return base
        return replace(base, **overrides)  # type: ignore[arg-type]

    def edge_prior(self, edge_type: str) -> float:
        return self.edge_priors.get(edge_type, self.min_edge_prior)


# Frozen restriction of the default: what the paper reports as the
# "+ schema gating" ablation of EPGM (non-trained).
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
)


__all__ = [
    "DEFAULT_EDGE_PRIORS",
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
    "REVISION_EDGE_TYPES",
    "WEIGHT_INFORMATIVE_VARIANCE",
]
