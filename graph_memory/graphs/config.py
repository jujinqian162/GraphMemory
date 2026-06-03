from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphBuildConfig:
    max_query_overlap: int = 20
    max_entity_neighbors: int = 10
    max_bridge_edges: int = 50
    use_spacy: bool = False

