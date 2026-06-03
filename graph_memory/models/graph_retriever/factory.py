from __future__ import annotations

from dataclasses import dataclass

from graph_memory.models.graph_retriever.config.records import TrainableModelConfig
from graph_memory.models.graph_retriever.internals.neural import (
    EvidenceScoringModel,
    IdentityGraphEncoder,
    RGCNGraphEncoder,
    SharedRelationTransform,
    TypedRelationTransform,
)
from graph_memory.validation import validate_trainable_model_config


@dataclass(frozen=True)
class GraphScoringModelFactory:
    """
    Factory for reconstructing trainable graph scoring models.
    可训练图评分模型的重建工厂。
    """

    def build(self, model_config: TrainableModelConfig) -> EvidenceScoringModel:
        return build_model_from_config(model_config)


def build_model_from_config(model_config: TrainableModelConfig) -> EvidenceScoringModel:
    """
    Reconstruct an EvidenceScoringModel from saved model config.
    根据保存的 model config 重建 EvidenceScoringModel。
    """

    validate_trainable_model_config(model_config)
    if model_config.graph_encoder_type == "identity" or model_config.num_layers == 0:
        graph_encoder = IdentityGraphEncoder()
    elif model_config.graph_encoder_type == "rgcn":
        if model_config.message_transform_type == "typed":
            def transform_factory() -> TypedRelationTransform:
                return TypedRelationTransform(
                    hidden_dim=model_config.hidden_dim,
                    num_relations=len(model_config.relation_vocab),
                )
        elif model_config.message_transform_type == "shared":
            def transform_factory() -> SharedRelationTransform:
                return SharedRelationTransform(hidden_dim=model_config.hidden_dim)
        else:
            raise ValueError(f"Unsupported message_transform_type: {model_config.message_transform_type}")
        graph_encoder = RGCNGraphEncoder(
            hidden_dim=model_config.hidden_dim,
            num_relations=len(model_config.relation_vocab),
            num_layers=model_config.num_layers,
            message_transform_factory=transform_factory,
            dropout=model_config.dropout,
        )
    else:
        raise ValueError(f"Unsupported graph_encoder_type: {model_config.graph_encoder_type}")

    return EvidenceScoringModel(
        encoder_dim=model_config.encoder_dim,
        node_feature_dim=len(model_config.feature_config.node_feature_names),
        hidden_dim=model_config.hidden_dim,
        graph_encoder=graph_encoder,
        scorer_feature_dim=len(model_config.feature_config.scorer_feature_names),
        dropout=model_config.dropout,
    )
