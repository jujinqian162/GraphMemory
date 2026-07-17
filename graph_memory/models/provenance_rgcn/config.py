from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, cast

from graph_memory.graphs.provenance import ProvenanceEdgeType, ProvenanceNodeType

PROVENANCE_RGCN_CHECKPOINT_FAMILY = "execution_provenance_rgcn"
PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION = 2
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
SUPPORTED_PROVENANCE_RGCN_MODEL_ABLATIONS = frozenset(
    {"full_rgcn", "wo_graph", "wo_edge_type", "wo_edge_weight"}
)


@dataclass(frozen=True)
class ProvenanceRgcnModelConfig:
    encoder_model: str = "intfloat/e5-base-v2"
    encoder_dim: int = 768
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    encoder_batch_size: int = 64
    hidden_dim: int = 128
    node_type_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.1
    ablation_name: str = "full_rgcn"
    message_transform_type: ProvenanceRgcnMessageTransformType = "typed"
    edge_weight_policy: ProvenanceRgcnEdgeWeightPolicy = "artifact"
    node_type_vocab: tuple[str, ...] = DEFAULT_NODE_TYPE_VOCAB
    relation_vocab: tuple[str, ...] = DEFAULT_RELATION_VOCAB

    def __post_init__(self) -> None:
        for name in (
            "encoder_dim",
            "encoder_batch_size",
            "hidden_dim",
            "node_type_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.ablation_name not in SUPPORTED_PROVENANCE_RGCN_MODEL_ABLATIONS:
            raise ValueError(
                "unsupported provenance R-GCN model ablation: "
                f"{self.ablation_name!r}."
            )
        if self.message_transform_type not in {"typed", "shared"}:
            raise ValueError("message_transform_type must be 'typed' or 'shared'.")
        if self.edge_weight_policy not in {"artifact", "uniform"}:
            raise ValueError("edge_weight_policy must be 'artifact' or 'uniform'.")
        if len(set(self.node_type_vocab)) != len(self.node_type_vocab):
            raise ValueError("node_type_vocab must be unique.")
        if len(set(self.relation_vocab)) != len(self.relation_vocab):
            raise ValueError("relation_vocab must be unique.")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["node_type_vocab"] = list(self.node_type_vocab)
        result["relation_vocab"] = list(self.relation_vocab)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProvenanceRgcnModelConfig:
        return cls(
            encoder_model=cast(str, value["encoder_model"]),
            encoder_dim=cast(int, value["encoder_dim"]),
            query_prefix=cast(str, value["query_prefix"]),
            passage_prefix=cast(str, value["passage_prefix"]),
            encoder_batch_size=cast(int, value["encoder_batch_size"]),
            hidden_dim=cast(int, value["hidden_dim"]),
            node_type_dim=cast(int, value["node_type_dim"]),
            num_layers=cast(int, value["num_layers"]),
            dropout=cast(float, value["dropout"]),
            ablation_name=cast(str, value.get("ablation_name", "full_rgcn")),
            message_transform_type=cast(
                ProvenanceRgcnMessageTransformType,
                value.get("message_transform_type", "typed"),
            ),
            edge_weight_policy=cast(
                ProvenanceRgcnEdgeWeightPolicy,
                value.get("edge_weight_policy", "artifact"),
            ),
            node_type_vocab=tuple(cast(Sequence[str], value["node_type_vocab"])),
            relation_vocab=tuple(cast(Sequence[str], value["relation_vocab"])),
        )


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
) -> ProvenanceRgcnModelConfig:
    if ablation_name not in SUPPORTED_PROVENANCE_RGCN_MODEL_ABLATIONS:
        raise ValueError(
            f"unsupported provenance R-GCN model ablation: {ablation_name!r}."
        )
    effective_ablation = ablation_name
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
    )


@dataclass(frozen=True)
class ProvenanceRgcnTrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 1
    epochs: int = 1
    max_grad_norm: float = 1.0
    random_seed: int = 13
    candidate_loss_weight: float = 1.0
    edge_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "max_grad_norm",
            "candidate_loss_weight",
            "edge_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProvenanceRgcnTrainingConfig:
        return cls(
            learning_rate=cast(float, value["learning_rate"]),
            batch_size=cast(int, value["batch_size"]),
            epochs=cast(int, value["epochs"]),
            max_grad_norm=cast(float, value["max_grad_norm"]),
            random_seed=cast(int, value["random_seed"]),
            candidate_loss_weight=cast(float, value["candidate_loss_weight"]),
            edge_loss_weight=cast(float, value["edge_loss_weight"]),
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
