from __future__ import annotations

from dataclasses import dataclass, field

from graph_memory.contracts.common import MethodName


@dataclass(frozen=True)
class NodeFeatureConfig:
    """
    Ordered numeric node feature configuration.
    有序的节点数值特征配置。
    """

    node_feature_names: tuple[str, ...] = (
        "seed_score",
        "seed_rank_percentile",
        "is_question_node",
    )
    scorer_feature_names: tuple[str, ...] = ("seed_score", "seed_rank_percentile")


@dataclass(frozen=True)
class BeamDecoderConfig:
    hidden_dim: int = 256
    step_embedding_dim: int = 16
    frontier_relation_dim: int = 4

    def to_json_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "step_embedding_dim": self.step_embedding_dim,
            "frontier_relation_dim": self.frontier_relation_dim,
        }


@dataclass(frozen=True)
class BeamSearchConfig:
    training_beam_size: int = 2
    inference_beam_size: int = 2
    max_steps: int = 5
    length_penalty_alpha: float = 1.0
    deduplicate_selected_sets: bool = True

    def to_json_dict(self) -> dict[str, object]:
        return {
            "training_beam_size": self.training_beam_size,
            "inference_beam_size": self.inference_beam_size,
            "max_steps": self.max_steps,
            "length_penalty_alpha": self.length_penalty_alpha,
            "deduplicate_selected_sets": self.deduplicate_selected_sets,
        }


@dataclass(frozen=True)
class BeamLossConfig:
    next_action_loss_weight: float = 1.0
    stop_loss_weight: float = 1.0
    aux_node_loss_weight: float = 0.2

    def to_json_dict(self) -> dict[str, object]:
        return {
            "next_action_loss_weight": self.next_action_loss_weight,
            "stop_loss_weight": self.stop_loss_weight,
            "aux_node_loss_weight": self.aux_node_loss_weight,
        }


@dataclass(frozen=True)
class OptimizerPhaseConfig:
    decoder_warmup_epochs: int = 0
    decoder_learning_rate: float = 1e-4
    rgcn_learning_rate: float = 1e-4

    def to_json_dict(self) -> dict[str, object]:
        return {
            "decoder_warmup_epochs": self.decoder_warmup_epochs,
            "decoder_learning_rate": self.decoder_learning_rate,
            "rgcn_learning_rate": self.rgcn_learning_rate,
        }


@dataclass(frozen=True)
class RgcnModelConfig:
    """
    Minimal model reconstruction config saved in every trainable checkpoint.
    每个可训练 checkpoint 中保存的最小模型重建配置。
    """

    method_name: MethodName
    encoder_model: str
    encoder_dim: int
    query_prefix: str
    passage_prefix: str
    encoder_batch_size: int
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
    decoder_config: BeamDecoderConfig = field(default_factory=BeamDecoderConfig)
    beam_search_config: BeamSearchConfig = field(default_factory=BeamSearchConfig)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "encoder_model": self.encoder_model,
            "encoder_dim": self.encoder_dim,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "encoder_batch_size": self.encoder_batch_size,
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
            "decoder_config": self.decoder_config.to_json_dict(),
            "beam_search_config": self.beam_search_config.to_json_dict(),
        }


@dataclass(frozen=True)
class RgcnTrainingConfig:
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
    beam_loss_config: BeamLossConfig = field(default_factory=BeamLossConfig)
    optimizer_phase_config: OptimizerPhaseConfig = field(
        default_factory=OptimizerPhaseConfig
    )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "random_seed": self.random_seed,
            "pos_weight_enabled": self.pos_weight_enabled,
            "epochs": self.epochs,
            "beam_loss_config": self.beam_loss_config.to_json_dict(),
            "optimizer_phase_config": self.optimizer_phase_config.to_json_dict(),
        }


__all__ = [
    "BeamDecoderConfig",
    "BeamLossConfig",
    "BeamSearchConfig",
    "NodeFeatureConfig",
    "OptimizerPhaseConfig",
    "RgcnModelConfig",
    "RgcnTrainingConfig",
]
