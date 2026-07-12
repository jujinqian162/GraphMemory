from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from graph_memory.registry.retrieval import RetrievalMethodId

AblationInvalidationStage: TypeAlias = Literal["pairs", "train"]


@dataclass(frozen=True)
class RgcnModelPatch:
    ablation: str

    def updates(self) -> dict[str, str | int]:
        return {"ablation": self.ablation}


@dataclass(frozen=True)
class NoGraphModelPatch(RgcnModelPatch):
    num_layers: Literal[0] = 0

    def updates(self) -> dict[str, str | int]:
        return {"ablation": self.ablation, "num_layers": self.num_layers}


@dataclass(frozen=True)
class PairSamplingPatch:
    hard_bm25_per_positive: Literal[0] = 0
    hard_dense_per_positive: Literal[0] = 0
    hard_graph_neighbor_per_positive: Literal[0] = 0

    def updates(self) -> dict[str, int]:
        return {
            "hard_bm25_per_positive": self.hard_bm25_per_positive,
            "hard_dense_per_positive": self.hard_dense_per_positive,
            "hard_graph_neighbor_per_positive": self.hard_graph_neighbor_per_positive,
        }


AblationConfigPatch: TypeAlias = RgcnModelPatch | NoGraphModelPatch | PairSamplingPatch


@dataclass(frozen=True)
class BaselineAblationVariant:
    identifier: Literal["full_rgcn"] = "full_rgcn"
    changed_dimensions: frozenset[str] = frozenset()
    baseline_alias: Literal[True] = True


@dataclass(frozen=True)
class ExecutableAblationVariant:
    identifier: str
    changed_dimensions: frozenset[str]
    earliest_invalidated_stage: AblationInvalidationStage
    config_patch: AblationConfigPatch
    baseline_alias: Literal[False] = False


AblationVariantPatch: TypeAlias = BaselineAblationVariant | ExecutableAblationVariant


@dataclass(frozen=True)
class AblationSuitePatch:
    method: RetrievalMethodId
    variants: tuple[AblationVariantPatch, ...]


def _model_variant(identifier: str, dimension: str) -> ExecutableAblationVariant:
    return ExecutableAblationVariant(
        identifier=identifier,
        changed_dimensions=frozenset({dimension}),
        earliest_invalidated_stage="train",
        config_patch=RgcnModelPatch(ablation=identifier),
    )


RGCN_ABLATION_PATCHES: tuple[AblationVariantPatch, ...] = (
    BaselineAblationVariant(),
    _model_variant("wo_bridge", "model_graph_view"),
    _model_variant("wo_entity_overlap", "model_graph_view"),
    _model_variant("wo_sequential", "model_graph_view"),
    _model_variant("wo_query_overlap", "model_graph_view"),
    ExecutableAblationVariant(
        identifier="wo_graph",
        changed_dimensions=frozenset({"model_structure"}),
        earliest_invalidated_stage="train",
        config_patch=NoGraphModelPatch(ablation="wo_graph"),
    ),
    _model_variant("wo_edge_type", "model_structure"),
    _model_variant("wo_edge_weight", "model_structure"),
    _model_variant("wo_seed_score", "model_structure"),
    ExecutableAblationVariant(
        identifier="wo_hard_negatives",
        changed_dimensions=frozenset({"pair_sampling"}),
        earliest_invalidated_stage="pairs",
        config_patch=PairSamplingPatch(),
    ),
)

RGCN_ABLATION_PATCH_SUITE = AblationSuitePatch(
    method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
    variants=RGCN_ABLATION_PATCHES,
)
DENSE_FT_RGCN_ABLATION_PATCH_SUITE = AblationSuitePatch(
    method=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    variants=RGCN_ABLATION_PATCHES,
)

ABLATION_SUITE_PATCHES = {
    RGCN_ABLATION_PATCH_SUITE.method: RGCN_ABLATION_PATCH_SUITE,
    DENSE_FT_RGCN_ABLATION_PATCH_SUITE.method: DENSE_FT_RGCN_ABLATION_PATCH_SUITE,
}


__all__ = [
    "ABLATION_SUITE_PATCHES",
    "AblationConfigPatch",
    "AblationInvalidationStage",
    "AblationSuitePatch",
    "AblationVariantPatch",
    "BaselineAblationVariant",
    "DENSE_FT_RGCN_ABLATION_PATCH_SUITE",
    "ExecutableAblationVariant",
    "NoGraphModelPatch",
    "PairSamplingPatch",
    "RGCN_ABLATION_PATCHES",
    "RGCN_ABLATION_PATCH_SUITE",
    "RgcnModelPatch",
]
