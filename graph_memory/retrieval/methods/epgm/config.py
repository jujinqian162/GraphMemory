"""Frozen configuration for the single non-trained EPGM retriever.

The method is *typed partner completion*: from each leading Dense anchor, walk
query-relevant typed provenance edges a bounded number of hops and promote the
anchor's evidence partner to just after it. There is no variant axis; one
configuration serves both synthetic dependency graphs and real agent traces.

Design note: hand-authored per-type priors and schema gates are deliberately
absent. They were the mechanism by which earlier variants overfitted a single
dataset. The only typed signal is the runtime query-to-relation affinity, which
reads the query and the schema but never dataset identity or labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Mapping

from graph_memory.graphs.provenance import ProvenanceEdgeType

RELATION_DESCRIPTION_VERSION = "provenance-relations-v1"

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

# `contains` is scope membership, not evidential dependency. Traversing it would
# make every node in an execution scope a bounded-hop neighbour of every other,
# destroying the precision of the hop bound. This is a schema fact, not a knob.
NON_TRAVERSABLE_EDGE_TYPES: frozenset[str] = frozenset(
    {ProvenanceEdgeType.CONTAINS.value}
)

# Candidate-level dependency relations. A promotion whose path carries one of
# these may emit a shared logical edge; structural relations such as `invokes`
# or `precedes` may rank evidence without claiming an output dependency.
EMITTABLE_EDGE_TYPES: frozenset[str] = frozenset(
    {
        ProvenanceEdgeType.FEEDS.value,
        ProvenanceEdgeType.DEPENDS_ON.value,
        ProvenanceEdgeType.GROUNDS.value,
        ProvenanceEdgeType.SUPPORTS.value,
        ProvenanceEdgeType.VERIFIES.value,
        ProvenanceEdgeType.CONTRADICTS.value,
        ProvenanceEdgeType.INVALIDATES.value,
        ProvenanceEdgeType.SUPERSEDES.value,
        ProvenanceEdgeType.AFFECTS.value,
    }
)


@dataclass(frozen=True)
class EpgmRetrieverConfig:
    """No-train EPGM configuration; every field is cache-relevant."""

    # How many Dense heads may propose a partner. Measured on RQ3: widening to 5
    # adds proposals but halves promotion precision, so the head stays narrow.
    anchor_top_a: int = 3
    # Bounded walk length. RQ3 results are identical for 1, 2, and 3 because
    # every accepted promotion there is a direct typed edge; 2 is kept so a
    # non-candidate connector (for example a tool call joining two outputs) can
    # still bridge two candidates.
    max_hops: int = 2
    # Penalty applied per hop beyond the first.
    hop_decay: float = 0.6
    # Single acceptance threshold. RQ3 is flat across 0.35-0.55; 0.4 sits inside
    # that plateau rather than on its edge.
    min_partner_confidence: float = 0.4
    # Dense prefix that promotions may never displace; protects MRR/Recall@2.
    preserve_dense_top_n: int = 2

    # Query-conditioned relation semantics.
    relation_description_version: str = RELATION_DESCRIPTION_VERSION
    relation_descriptions: Mapping[str, str] = field(
        default=DEFAULT_RELATION_DESCRIPTIONS
    )
    relation_temperature: float = 0.2

    def __post_init__(self) -> None:
        for name in ("anchor_top_a", "max_hops"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.preserve_dense_top_n < 0:
            raise ValueError("preserve_dense_top_n must be non-negative.")
        if not self.relation_description_version.strip():
            raise ValueError("relation_description_version must be non-empty.")
        if set(self.relation_descriptions) != {
            edge_type.value for edge_type in ProvenanceEdgeType
        }:
            raise ValueError(
                "relation_descriptions must cover every edge type exactly."
            )
        if any(
            not description.strip()
            for description in self.relation_descriptions.values()
        ):
            raise ValueError("relation descriptions must be non-empty.")
        for name in ("hop_decay", "min_partner_confidence", "relation_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.relation_temperature <= 0.0:
            raise ValueError("relation_temperature must be positive.")
        if not 0.0 < self.hop_decay <= 1.0:
            raise ValueError("hop_decay must be in (0, 1].")
        object.__setattr__(
            self,
            "relation_descriptions",
            MappingProxyType(dict(self.relation_descriptions)),
        )

    def cache_fingerprint(self) -> str:
        """Digest every frozen behavior field for ranking cache identity."""

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


__all__ = [
    "DEFAULT_RELATION_DESCRIPTIONS",
    "EMITTABLE_EDGE_TYPES",
    "EpgmRetrieverConfig",
    "NON_TRAVERSABLE_EDGE_TYPES",
    "RELATION_DESCRIPTION_VERSION",
]
