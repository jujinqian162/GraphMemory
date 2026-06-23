from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

FastGraphRAGExtractorType = Literal["regex_english", "syntactic_parser", "cfg"]


def _empty_noun_phrase_grammars() -> Mapping[str, str]:
    return {}


@dataclass(frozen=True)
class FastGraphRAGExtractionConfig:
    extractor_type: FastGraphRAGExtractorType = "regex_english"
    normalize_edge_weights: bool = True
    max_word_length: int = 15
    word_delimiter: str = " "
    include_named_entities: bool = True
    exclude_nouns: tuple[str, ...] | None = None
    exclude_entity_tags: tuple[str, ...] = ()
    exclude_pos_tags: tuple[str, ...] = ()
    noun_phrase_tags: tuple[str, ...] = ()
    noun_phrase_grammars: Mapping[str, str] = field(default_factory=_empty_noun_phrase_grammars)
    model_name: str = "en_core_web_md"


@dataclass(frozen=True)
class FastGraphRAGPruningConfig:
    min_node_freq: int = 1
    max_node_freq_std: float | None = None
    min_node_degree: int = 0
    max_node_degree_std: float | None = None
    min_edge_weight_pct: float = 0.0
    remove_ego_nodes: bool = False
    lcc_only: bool = False


@dataclass(frozen=True)
class FastGraphRAGScoringConfig:
    lambda_entity: float = 1.0
    lambda_relation: float = 1.0
    lambda_dense_fallback: float = 0.05


@dataclass(frozen=True)
class FastGraphRAGConfig:
    extraction: FastGraphRAGExtractionConfig = field(default_factory=FastGraphRAGExtractionConfig)
    pruning: FastGraphRAGPruningConfig = field(default_factory=FastGraphRAGPruningConfig)
    scoring: FastGraphRAGScoringConfig = field(default_factory=FastGraphRAGScoringConfig)
    entity_seed_top_k: int = 32
    query_link_seed_score: float = 1.0
    dense_entity_seed_weight: float = 1.0
    lexical_substring_match_score: float = 0.5
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-8


__all__ = [
    "FastGraphRAGConfig",
    "FastGraphRAGExtractionConfig",
    "FastGraphRAGExtractorType",
    "FastGraphRAGPruningConfig",
    "FastGraphRAGScoringConfig",
]
