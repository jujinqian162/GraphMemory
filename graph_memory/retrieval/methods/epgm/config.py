from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from graph_memory.graphs.provenance import ProvenanceEdgeType

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

    The method is a deterministic dense seed + typed bidirectional graph
    propagation reranker. Nothing here is learned.
    """

    seed_top_s: int = 8
    beam_width: int = 32
    max_hops: int = 4
    max_path_expansions: int = 4000
    graph_weight: float = 0.25
    hop_decay: float = 0.6
    reverse_edge_factor: float = 0.85
    min_edge_prior: float = 0.05
    edge_priors: Mapping[str, float] = field(default=DEFAULT_EDGE_PRIORS)

    def __post_init__(self) -> None:
        if self.seed_top_s <= 0:
            raise ValueError("seed_top_s must be positive.")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.max_hops <= 0:
            raise ValueError("max_hops must be positive.")
        if self.max_path_expansions <= 0:
            raise ValueError("max_path_expansions must be positive.")
        for name in ("graph_weight", "hop_decay", "reverse_edge_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0.0 <= self.min_edge_prior <= 1.0:
            raise ValueError("min_edge_prior must be in [0, 1].")

    def edge_prior(self, edge_type: str) -> float:
        return self.edge_priors.get(edge_type, self.min_edge_prior)


__all__ = [
    "DEFAULT_EDGE_PRIORS",
    "EpgmRetrieverConfig",
    "HUB_NODE_TYPES",
    "NON_TRAVERSABLE_EDGE_TYPES",
]
