from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from graph_memory.contracts.common import EdgeType
from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveInt,
)

_KNOWN_NODE_FEATURES = frozenset(
    {"seed_score", "seed_rank_percentile", "is_question_node"}
)


class NodeFeatureConfig(DomainModel):
    node_feature_names: tuple[NonEmptyStr, ...] = (
        "seed_score",
        "seed_rank_percentile",
        "is_question_node",
    )
    scorer_feature_names: tuple[NonEmptyStr, ...] = (
        "seed_score",
        "seed_rank_percentile",
    )

    @model_validator(mode="after")
    def _validate_features(self) -> "NodeFeatureConfig":
        for field_name in ("node_feature_names", "scorer_feature_names"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
            unknown = sorted(set(values) - _KNOWN_NODE_FEATURES)
            if unknown:
                raise ValueError(f"unsupported {field_name}={unknown}")
        if not set(self.scorer_feature_names).issubset(self.node_feature_names):
            raise ValueError("scorer features must be enabled node features")
        return self


class RgcnModelConfig(DomainModel):
    method_name: NonEmptyStr
    encoder_model: NonEmptyStr
    encoder_dim: PositiveInt
    query_prefix: str
    passage_prefix: str
    encoder_batch_size: PositiveInt
    hidden_dim: PositiveInt
    num_layers: NonNegativeInt
    dropout: FiniteFloat = Field(ge=0.0, lt=1.0)
    feature_config: NodeFeatureConfig
    relation_vocab: tuple[NonEmptyStr, ...] = Field(min_length=1)
    graph_encoder_type: Literal["identity", "rgcn"]
    message_transform_type: Literal["typed", "shared"]
    edge_weight_policy: Literal["artifact", "uniform"]
    enabled_edge_types: tuple[EdgeType, ...]
    enabled_provenance_relations: tuple[NonEmptyStr, ...] = ()
    message_topology: Literal["native", "degree_preserving_random_v1"] = "native"
    message_topology_seed: int = 13
    ablation_name: NonEmptyStr
    scoring_mode: Literal[
        "seed_residual", "seed_passthrough", "residual_only"
    ] = "seed_residual"

    @model_validator(mode="after")
    def _validate_vocab(self) -> "RgcnModelConfig":
        if len(self.relation_vocab) != len(set(self.relation_vocab)):
            raise ValueError("relation_vocab must be unique")
        if len(self.enabled_edge_types) != len(set(self.enabled_edge_types)):
            raise ValueError("enabled_edge_types must be unique")
        if len(self.enabled_provenance_relations) != len(
            set(self.enabled_provenance_relations)
        ):
            raise ValueError("enabled_provenance_relations must be unique")
        if self.method_name != "provenance_rgcn" and (
            self.enabled_provenance_relations
            or self.message_topology != "native"
        ):
            raise ValueError(
                "provenance relation/topology policies require method_name='provenance_rgcn'"
            )
        has_seed_score = "seed_score" in self.feature_config.scorer_feature_names
        if self.scoring_mode in {"seed_residual", "seed_passthrough"} and not has_seed_score:
            raise ValueError(
                f"scoring_mode={self.scoring_mode!r} requires scorer feature 'seed_score'"
            )
        if self.scoring_mode == "residual_only" and has_seed_score:
            raise ValueError(
                "scoring_mode='residual_only' must not expose scorer feature 'seed_score'"
            )
        return self


class RgcnTrainingConfig(DomainModel):
    optimizer_name: Literal["AdamW"] = "AdamW"
    learning_rate: NonNegativeFiniteFloat = 1e-4
    per_device_graph_batch_size: PositiveInt = 1
    max_grad_norm: NonNegativeFiniteFloat = 1.0
    random_seed: int = 13
    pos_weight_enabled: StrictBool = False
    epochs: PositiveInt = 1


__all__ = [
    "NodeFeatureConfig",
    "RgcnModelConfig",
    "RgcnTrainingConfig",
]
