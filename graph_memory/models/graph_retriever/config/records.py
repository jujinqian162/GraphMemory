from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.common import MethodName


@dataclass(frozen=True)
class NodeFeatureConfig:
    """
    Ordered numeric node feature configuration.
    有序的节点数值特征配置。
    """

    node_feature_names: tuple[str, ...] = ("seed_score", "seed_rank_percentile", "is_question_node")
    scorer_feature_names: tuple[str, ...] = ("seed_score", "seed_rank_percentile")


@dataclass(frozen=True)
class TrainableModelConfig:
    """
    Minimal model reconstruction config saved in every trainable checkpoint.
    每个可训练 checkpoint 中保存的最小模型重建配置。
    """

    method_name: MethodName
    encoder_model: str
    encoder_dim: int
    query_prefix: str
    passage_prefix: str
    hidden_dim: int
    num_layers: int
    dropout: float
    feature_config: NodeFeatureConfig
    relation_vocab: tuple[str, ...]
    graph_encoder_type: str
    message_transform_type: str
    edge_weight_policy: str
    enabled_edge_types: tuple[str, ...]
    ablation_name: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "encoder_model": self.encoder_model,
            "encoder_dim": self.encoder_dim,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "feature_config": {
                "node_feature_names": list(self.feature_config.node_feature_names),
                "scorer_feature_names": list(self.feature_config.scorer_feature_names),
            },
            "relation_vocab": list(self.relation_vocab),
            "graph_encoder_type": self.graph_encoder_type,
            "message_transform_type": self.message_transform_type,
            "edge_weight_policy": self.edge_weight_policy,
            "enabled_edge_types": list(self.enabled_edge_types),
            "ablation_name": self.ablation_name,
        }


@dataclass(frozen=True)
class TrainableTrainingConfig:
    """
    Minimal training config needed to resume or audit a trainable run.
    用于恢复或审计可训练运行的最小训练配置。
    """

    optimizer_name: str = "AdamW"
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_grad_norm: float = 1.0
    random_seed: int = 13
    pos_weight_enabled: bool = False
    epochs: int = 1

    def to_json_dict(self) -> dict[str, object]:
        return {
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "random_seed": self.random_seed,
            "pos_weight_enabled": self.pos_weight_enabled,
            "epochs": self.epochs,
        }


__all__ = [
    "NodeFeatureConfig",
    "TrainableModelConfig",
    "TrainableTrainingConfig",
]
