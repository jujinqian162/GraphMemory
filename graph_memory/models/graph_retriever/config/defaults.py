from __future__ import annotations

from graph_memory.models.graph_retriever.config.records import NodeFeatureConfig, RgcnModelConfig


def default_model_config(
    *,
    method_name: str,
    encoder_model: str,
    encoder_dim: int,
    query_prefix: str,
    passage_prefix: str,
    encoder_batch_size: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    ablation_name: str = "full_rgcn",
) -> RgcnModelConfig:
    """
    Build the default trainable model config for one ablation name.
    为一个 ablation 名称构造默认可训练模型配置。
    """

    feature_config = NodeFeatureConfig()
    graph_encoder_type = "rgcn"
    message_transform_type = "typed"
    edge_weight_policy = "artifact"
    enabled_edge_types = ("bridge", "entity_overlap", "query_overlap", "sequential")
    canonical_ablation = ablation_name
    layer_count = num_layers

    if ablation_name in {"identity", "wo_graph", "num_layers_0"} or num_layers == 0:
        graph_encoder_type = "identity"
        layer_count = 0
        canonical_ablation = "wo_graph"
    elif ablation_name == "wo_edge_type":
        message_transform_type = "shared"
    elif ablation_name == "wo_bridge":
        enabled_edge_types = ("entity_overlap", "query_overlap", "sequential")
    elif ablation_name == "wo_entity_overlap":
        enabled_edge_types = ("bridge", "query_overlap", "sequential")
    elif ablation_name == "wo_sequential":
        enabled_edge_types = ("bridge", "entity_overlap", "query_overlap")
    elif ablation_name == "wo_query_overlap":
        enabled_edge_types = ("bridge", "entity_overlap", "sequential")
    elif ablation_name == "wo_edge_weight":
        edge_weight_policy = "uniform"
    elif ablation_name == "wo_seed_score":
        feature_config = NodeFeatureConfig(node_feature_names=("is_question_node",), scorer_feature_names=())

    return RgcnModelConfig(
        method_name=method_name,
        encoder_model=encoder_model,
        encoder_dim=encoder_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_batch_size=encoder_batch_size,
        hidden_dim=hidden_dim,
        num_layers=layer_count,
        dropout=dropout,
        feature_config=feature_config,
        relation_vocab=(
            "query_overlap_forward",
            "sequential_forward",
            "sequential_reverse",
            "entity_overlap_forward",
            "entity_overlap_reverse",
            "bridge_forward",
            "bridge_reverse",
        ),
        graph_encoder_type=graph_encoder_type,
        message_transform_type=message_transform_type,
        edge_weight_policy=edge_weight_policy,
        enabled_edge_types=enabled_edge_types,
        ablation_name=canonical_ablation,
    )
