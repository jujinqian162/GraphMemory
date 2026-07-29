from __future__ import annotations

from typing import Literal, TypeAlias, cast

from pydantic import Field, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveFiniteFloat,
    PositiveInt,
)
from graph_memory.graphs.provenance import ProvenanceEdgeType, ProvenanceNodeType

PROVENANCE_RGCN_CHECKPOINT_FAMILY = "execution_provenance_rgcn"
PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION = 4
DEFAULT_NODE_TYPE_VOCAB = tuple(node_type.value for node_type in ProvenanceNodeType)
DEFAULT_FEEDS_BINDING_RELATIONS = (
    "feeds:evidence:context:semantic_reference",
    "feeds:bound_other",
)
DEFAULT_RELATION_VOCAB = tuple(
    relation
    for edge_type in ProvenanceEdgeType
    if edge_type is not ProvenanceEdgeType.FEEDS
    for relation in (f"{edge_type.value}_forward", f"{edge_type.value}_reverse")
) + tuple(
    relation
    for binding_relation in DEFAULT_FEEDS_BINDING_RELATIONS
    for relation in (f"{binding_relation}_forward", f"{binding_relation}_reverse")
)

ProvenanceRgcnMessageTransformType: TypeAlias = Literal["typed", "shared"]
ProvenanceRgcnEdgeWeightPolicy: TypeAlias = Literal["artifact", "uniform"]
ProvenanceRgcnAblation: TypeAlias = Literal[
    "full_rgcn",
    "wo_graph",
    "wo_edge_type",
    "wo_edge_weight",
    "diagnostic_shuffled_feed",
]
SUPPORTED_PROVENANCE_RGCN_MODEL_ABLATIONS = frozenset(
    {
        "full_rgcn",
        "wo_graph",
        "wo_edge_type",
        "wo_edge_weight",
        "diagnostic_shuffled_feed",
    }
)


class ProvenanceRgcnModelConfig(DomainModel):
    encoder_model: NonEmptyStr = "intfloat/e5-base-v2"
    encoder_dim: PositiveInt = 768
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    encoder_batch_size: PositiveInt = 64
    hidden_dim: PositiveInt = 128
    node_type_dim: PositiveInt = 16
    num_layers: NonNegativeInt = 2
    dropout: FiniteFloat = Field(default=0.1, ge=0.0, lt=1.0)
    ablation_name: ProvenanceRgcnAblation = Field(
        default="full_rgcn", json_schema_extra={"checkpoint_required": True}
    )
    message_transform_type: ProvenanceRgcnMessageTransformType = Field(
        default="typed", json_schema_extra={"checkpoint_required": True}
    )
    edge_weight_policy: ProvenanceRgcnEdgeWeightPolicy = Field(
        default="artifact", json_schema_extra={"checkpoint_required": True}
    )
    structured_pool_size: PositiveInt = 16
    structured_seed_top_s: PositiveInt = 5
    preserve_node_top_n: NonNegativeInt = 2
    edge_accept_threshold: FiniteFloat = Field(default=0.5, ge=0.0, le=1.0)
    feed_message_topology: Literal["native", "shuffled"] = "native"
    feed_message_shuffle_seed: int = 13
    node_type_vocab: tuple[NonEmptyStr, ...] = DEFAULT_NODE_TYPE_VOCAB
    relation_vocab: tuple[NonEmptyStr, ...] = DEFAULT_RELATION_VOCAB

    @model_validator(mode="after")
    def _validate_model(self) -> "ProvenanceRgcnModelConfig":
        if self.structured_seed_top_s > self.structured_pool_size:
            raise ValueError("structured_seed_top_s cannot exceed structured_pool_size")
        if self.preserve_node_top_n > self.structured_pool_size:
            raise ValueError("preserve_node_top_n cannot exceed structured_pool_size")
        if (self.ablation_name == "diagnostic_shuffled_feed") != (
            self.feed_message_topology == "shuffled"
        ):
            raise ValueError(
                "diagnostic_shuffled_feed must exactly select shuffled feed topology"
            )
        if len(self.node_type_vocab) != len(set(self.node_type_vocab)):
            raise ValueError("node_type_vocab must be unique")
        if len(self.relation_vocab) != len(set(self.relation_vocab)):
            raise ValueError("relation_vocab must be unique")
        return self


class ProvenanceRgcnTrainingConfig(DomainModel):
    learning_rate: PositiveFiniteFloat = 1e-4
    per_device_graph_batch_size: PositiveInt = 1
    epochs: PositiveInt = 1
    max_grad_norm: NonNegativeFiniteFloat = 1.0
    random_seed: int = 13
    candidate_loss_weight: NonNegativeFiniteFloat = 1.0
    edge_loss_weight: NonNegativeFiniteFloat = 0.5


def default_provenance_rgcn_model_config(
    *,
    encoder_model: str,
    encoder_dim: int,
    query_prefix: str,
    passage_prefix: str,
    encoder_batch_size: int,
    hidden_dim: int = 128,
    node_type_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.1,
    ablation_name: str = "full_rgcn",
    structured_pool_size: int = 16,
    structured_seed_top_s: int = 5,
    preserve_node_top_n: int = 2,
    edge_accept_threshold: float = 0.5,
    feed_message_shuffle_seed: int = 13,
) -> ProvenanceRgcnModelConfig:
    if ablation_name not in SUPPORTED_PROVENANCE_RGCN_MODEL_ABLATIONS:
        raise ValueError(
            f"unsupported provenance R-GCN model ablation: {ablation_name!r}."
        )
    effective_ablation = cast(ProvenanceRgcnAblation, ablation_name)
    effective_layers = num_layers
    message_transform_type: ProvenanceRgcnMessageTransformType = "typed"
    edge_weight_policy: ProvenanceRgcnEdgeWeightPolicy = "artifact"
    if ablation_name == "wo_graph" or num_layers == 0:
        effective_ablation = "wo_graph"
        effective_layers = 0
    elif ablation_name == "wo_edge_type":
        message_transform_type = "shared"
    elif ablation_name == "wo_edge_weight":
        edge_weight_policy = "uniform"
    feed_message_topology: Literal["native", "shuffled"] = (
        "shuffled" if ablation_name == "diagnostic_shuffled_feed" else "native"
    )
    return ProvenanceRgcnModelConfig(
        encoder_model=encoder_model,
        encoder_dim=encoder_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_batch_size=encoder_batch_size,
        hidden_dim=hidden_dim,
        node_type_dim=node_type_dim,
        num_layers=effective_layers,
        dropout=dropout,
        ablation_name=effective_ablation,
        message_transform_type=message_transform_type,
        edge_weight_policy=edge_weight_policy,
        structured_pool_size=structured_pool_size,
        structured_seed_top_s=structured_seed_top_s,
        preserve_node_top_n=preserve_node_top_n,
        edge_accept_threshold=edge_accept_threshold,
        feed_message_topology=feed_message_topology,
        feed_message_shuffle_seed=feed_message_shuffle_seed,
    )


__all__ = [
    "DEFAULT_NODE_TYPE_VOCAB",
    "DEFAULT_FEEDS_BINDING_RELATIONS",
    "DEFAULT_RELATION_VOCAB",
    "PROVENANCE_RGCN_CHECKPOINT_FAMILY",
    "PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION",
    "ProvenanceRgcnEdgeWeightPolicy",
    "ProvenanceRgcnMessageTransformType",
    "ProvenanceRgcnModelConfig",
    "ProvenanceRgcnTrainingConfig",
    "SUPPORTED_PROVENANCE_RGCN_MODEL_ABLATIONS",
    "default_provenance_rgcn_model_config",
]
