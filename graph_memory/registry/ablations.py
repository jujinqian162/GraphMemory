from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from graph_memory.registry.retrieval import RetrievalMethodId

AblationInvalidationStage: TypeAlias = Literal["pairs", "train"]


class AblationVariantId(str, Enum):
    WO_BRIDGE = "wo_bridge"
    WO_ENTITY_OVERLAP = "wo_entity_overlap"
    WO_SEQUENTIAL = "wo_sequential"
    WO_QUERY_OVERLAP = "wo_query_overlap"
    WO_GRAPH = "wo_graph"
    WO_EDGE_TYPE = "wo_edge_type"
    WO_EDGE_WEIGHT = "wo_edge_weight"
    WO_SEED_SCORE = "wo_seed_score"
    WO_HARD_NEGATIVES = "wo_hard_negatives"


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
    identifier: AblationVariantId
    changed_dimensions: frozenset[str]
    earliest_invalidated_stage: AblationInvalidationStage
    config_patch: AblationConfigPatch
    baseline_alias: Literal[False] = False


AblationVariantPatch: TypeAlias = BaselineAblationVariant | ExecutableAblationVariant


@dataclass(frozen=True)
class AblationSuitePatch:
    method: RetrievalMethodId
    variants: tuple[AblationVariantPatch, ...]


def _model_variant(
    identifier: AblationVariantId,
    dimension: str,
) -> ExecutableAblationVariant:
    return ExecutableAblationVariant(
        identifier=identifier,
        changed_dimensions=frozenset({dimension}),
        earliest_invalidated_stage="train",
        config_patch=RgcnModelPatch(ablation=identifier),
    )


RGCN_ABLATION_PATCHES: tuple[AblationVariantPatch, ...] = (
    BaselineAblationVariant(),
    _model_variant(AblationVariantId.WO_BRIDGE, "model_graph_view"),
    _model_variant(AblationVariantId.WO_ENTITY_OVERLAP, "model_graph_view"),
    _model_variant(AblationVariantId.WO_SEQUENTIAL, "model_graph_view"),
    _model_variant(AblationVariantId.WO_QUERY_OVERLAP, "model_graph_view"),
    ExecutableAblationVariant(
        identifier=AblationVariantId.WO_GRAPH,
        changed_dimensions=frozenset({"model_structure"}),
        earliest_invalidated_stage="train",
        config_patch=NoGraphModelPatch(ablation="wo_graph"),
    ),
    _model_variant(AblationVariantId.WO_EDGE_TYPE, "model_structure"),
    _model_variant(AblationVariantId.WO_EDGE_WEIGHT, "model_structure"),
    _model_variant(AblationVariantId.WO_SEED_SCORE, "model_structure"),
    ExecutableAblationVariant(
        identifier=AblationVariantId.WO_HARD_NEGATIVES,
        changed_dimensions=frozenset({"pair_sampling"}),
        earliest_invalidated_stage="pairs",
        config_patch=PairSamplingPatch(),
    ),
)

EXECUTION_PROVENANCE_RGCN_ABLATION_PATCHES: tuple[AblationVariantPatch, ...] = (
    BaselineAblationVariant(),
    ExecutableAblationVariant(
        identifier=AblationVariantId.WO_GRAPH,
        changed_dimensions=frozenset({"model_structure"}),
        earliest_invalidated_stage="train",
        config_patch=NoGraphModelPatch(ablation="wo_graph"),
    ),
    _model_variant(AblationVariantId.WO_EDGE_TYPE, "model_structure"),
    _model_variant(AblationVariantId.WO_EDGE_WEIGHT, "model_structure"),
    ExecutableAblationVariant(
        identifier=AblationVariantId.WO_HARD_NEGATIVES,
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
EXECUTION_PROVENANCE_RGCN_ABLATION_PATCH_SUITE = AblationSuitePatch(
    method=RetrievalMethodId.EXECUTION_PROVENANCE_RGCN_RETRIEVER,
    variants=EXECUTION_PROVENANCE_RGCN_ABLATION_PATCHES,
)

ABLATION_SUITE_PATCHES = {
    RGCN_ABLATION_PATCH_SUITE.method: RGCN_ABLATION_PATCH_SUITE,
    DENSE_FT_RGCN_ABLATION_PATCH_SUITE.method: DENSE_FT_RGCN_ABLATION_PATCH_SUITE,
    EXECUTION_PROVENANCE_RGCN_ABLATION_PATCH_SUITE.method: (
        EXECUTION_PROVENANCE_RGCN_ABLATION_PATCH_SUITE
    ),
}


__all__ = [
    "ABLATION_SUITE_PATCHES",
    "AblationConfigPatch",
    "AblationInvalidationStage",
    "AblationSuitePatch",
    "AblationVariantId",
    "AblationVariantPatch",
    "BaselineAblationVariant",
    "DENSE_FT_RGCN_ABLATION_PATCH_SUITE",
    "EXECUTION_PROVENANCE_RGCN_ABLATION_PATCHES",
    "EXECUTION_PROVENANCE_RGCN_ABLATION_PATCH_SUITE",
    "ExecutableAblationVariant",
    "NoGraphModelPatch",
    "PairSamplingPatch",
    "RGCN_ABLATION_PATCHES",
    "RGCN_ABLATION_PATCH_SUITE",
    "RgcnModelPatch",
]
