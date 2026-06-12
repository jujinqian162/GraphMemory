from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from graph_memory.registry.retrieval import RetrievalMethodId


@dataclass(frozen=True)
class AblationVariantPatch:
    identifier: str
    changed_dimensions: frozenset[str]
    training_config_override: Mapping[str, Any] = field(default_factory=dict)
    baseline_alias: bool = False


@dataclass(frozen=True)
class AblationSuitePatch:
    method: str
    variants: tuple[AblationVariantPatch, ...]


RGCN_ABLATION_PATCHES = (
    AblationVariantPatch("full_rgcn", frozenset(), baseline_alias=True),
    AblationVariantPatch(
        "wo_bridge",
        frozenset({"model_graph_view"}),
        {"train": {"model": {"ablation": "wo_bridge"}}},
    ),
    AblationVariantPatch(
        "wo_entity_overlap",
        frozenset({"model_graph_view"}),
        {"train": {"model": {"ablation": "wo_entity_overlap"}}},
    ),
    AblationVariantPatch(
        "wo_sequential",
        frozenset({"model_graph_view"}),
        {"train": {"model": {"ablation": "wo_sequential"}}},
    ),
    AblationVariantPatch(
        "wo_query_overlap",
        frozenset({"model_graph_view"}),
        {"train": {"model": {"ablation": "wo_query_overlap"}}},
    ),
    AblationVariantPatch(
        "wo_graph",
        frozenset({"model_structure"}),
        {"train": {"model": {"ablation": "wo_graph", "num_layers": 0}}},
    ),
    AblationVariantPatch(
        "wo_edge_type",
        frozenset({"model_structure"}),
        {"train": {"model": {"ablation": "wo_edge_type"}}},
    ),
    AblationVariantPatch(
        "wo_edge_weight",
        frozenset({"model_structure"}),
        {"train": {"model": {"ablation": "wo_edge_weight"}}},
    ),
    AblationVariantPatch(
        "wo_seed_score",
        frozenset({"model_structure"}),
        {"train": {"model": {"ablation": "wo_seed_score"}}},
    ),
    AblationVariantPatch(
        "wo_hard_negatives",
        frozenset({"pair_sampling"}),
        {
            "pairs": {
                "hard_bm25_per_positive": 0,
                "hard_dense_per_positive": 0,
                "hard_graph_neighbor_per_positive": 0,
            }
        },
    ),
)

RGCN_ABLATION_PATCH_SUITE = AblationSuitePatch(
    method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value,
    variants=RGCN_ABLATION_PATCHES,
)

ABLATION_SUITE_PATCHES = {
    RGCN_ABLATION_PATCH_SUITE.method: RGCN_ABLATION_PATCH_SUITE,
}


__all__ = [
    "ABLATION_SUITE_PATCHES",
    "AblationSuitePatch",
    "AblationVariantPatch",
    "RGCN_ABLATION_PATCHES",
    "RGCN_ABLATION_PATCH_SUITE",
]
