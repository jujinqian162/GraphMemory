from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_memory.evaluation.service import evaluate_results
from graph_memory.learned.batching import build_full_ranking_batches, build_training_batches, move_training_batch
from graph_memory.learned.features import SeedSignalProvider, TextEmbeddingProvider
from graph_memory.learned.model import (
    EvidenceScoringModel,
    IdentityGraphEncoder,
    RGCNGraphEncoder,
    SharedRelationTransform,
    TypedRelationTransform,
)
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.graphs.views import induced_retrieved_subgraph, model_visible_graph
from graph_memory.types import (
    NodeFeatureConfig,
    RankedNode,
    TrainableModelConfig,
    TrainableTrainingConfig,
)
from graph_memory.validation import (
    validate_graphs,
    validate_memory_task_inputs,
    validate_memory_task_labels,
    validate_train_pairs,
    validate_trainable_model_config,
    validate_trainable_training_config,
)


MetricRecord: TypeAlias = dict[str, object]
CheckpointCallback: TypeAlias = Callable[["TrainableTrainingResult"], None]


@dataclass(frozen=True)
class TrainableTrainingResult:
    """
    In-memory result of one trainable retriever training run.
    一次可训练检索器训练运行的内存结果。

    Fields / 字段:
    - model_config: Effective model config used by the run.
      model_config：本次运行使用的实际模型配置。
    - training_config: Effective training config used by the run.
      training_config：本次运行使用的实际训练配置。
    - metric_records: Epoch-level metric records ready for JSONL output.
      metric_records：可写入 JSONL 的 epoch 级 metric 记录。
    - best_model_state_dict: CPU state dict for the best dev checkpoint.
      best_model_state_dict：最佳 dev checkpoint 对应的 CPU state dict。
    - optimizer_state_dict: Optimizer state dict after training.
      optimizer_state_dict：训练结束后的 optimizer state dict。
    - scheduler_state_dict: Scheduler state dict after training.
      scheduler_state_dict：训练结束后的 scheduler state dict。
    - best_epoch: Epoch number with the best dev metric.
      best_epoch：dev metric 最好的 epoch 编号。
    - global_step: Total optimizer steps.
      global_step：optimizer step 总数。
    - best_dev_metric: Best checkpoint selection metric.
      best_dev_metric：最佳 checkpoint 选择指标。
    """

    model_config: TrainableModelConfig
    training_config: TrainableTrainingConfig
    metric_records: list[MetricRecord]
    best_model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object]
    best_epoch: int
    global_step: int
    best_dev_metric: float


def default_model_config(
    *,
    encoder_model: str,
    encoder_dim: int,
    query_prefix: str,
    passage_prefix: str,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    ablation_name: str = "full_rgcn",
) -> TrainableModelConfig:
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

    return TrainableModelConfig(
        method_name="dense_rgcn_graph_retriever",
        encoder_model=encoder_model,
        encoder_dim=encoder_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
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


def train_graph_retriever(
    *,
    train_task_inputs: list[MemoryTaskInput],
    train_graphs: list[MemoryGraph],
    train_pairs: list[TrainPairRecord],
    dev_task_inputs: list[MemoryTaskInput],
    dev_labels: list[MemoryTaskLabels],
    dev_graphs: list[MemoryGraph],
    model_config: TrainableModelConfig,
    training_config: TrainableTrainingConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    train_labels: list[MemoryTaskLabels] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    device: str | torch.device = "cpu",
) -> TrainableTrainingResult:
    """
    Train a frozen-encoder R-GCN binary node scorer.
    训练一个 frozen-encoder R-GCN 二分类节点 scorer。
    """

    validate_trainable_model_config(model_config)
    validate_trainable_training_config(training_config)
    validate_memory_task_inputs(train_task_inputs)
    validate_memory_task_inputs(dev_task_inputs)
    train_inputs_by_task_id = {task_input["task_id"]: task_input for task_input in train_task_inputs}
    dev_inputs_by_task_id = {task_input["task_id"]: task_input for task_input in dev_task_inputs}
    validate_graphs(train_graphs, train_inputs_by_task_id)
    validate_graphs(dev_graphs, dev_inputs_by_task_id)
    validate_memory_task_labels(dev_labels, dev_inputs_by_task_id)
    if train_labels is not None:
        train_labels_by_task_id = {label["task_id"]: label for label in train_labels}
        validate_memory_task_labels(train_labels, train_inputs_by_task_id)
        validate_train_pairs(
            train_pairs,
            train_inputs_by_task_id,
            train_labels_by_task_id,
            {graph["task_id"]: graph for graph in train_graphs},
        )

    _ = torch.manual_seed(training_config.random_seed)
    device = torch.device(device)
    model = build_model_from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    train_batches = build_training_batches(
        task_inputs=train_task_inputs,
        graphs=train_graphs,
        pairs=train_pairs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=training_config.batch_size,
    )
    if not train_batches:
        raise ValueError("Training requires at least one non-empty training batch.")

    pos_weight = _pos_weight(train_pairs, device) if training_config.pos_weight_enabled else None
    metric_records: list[MetricRecord] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    global_step = 0
    negative_count_by_type = _negative_count_by_type(train_pairs)
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_sample_count = 0
        last_grad_norm = 0.0
        for batch in train_batches:
            moved_batch = move_training_batch(batch, device)
            logits = model(moved_batch)
            loss = F.binary_cross_entropy_with_logits(logits, moved_batch.labels, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            sample_count = int(moved_batch.labels.shape[0])
            train_loss_total += float(loss.detach().cpu()) * sample_count
            train_sample_count += sample_count
            last_grad_norm = float(grad_norm.detach().cpu() if isinstance(grad_norm, Tensor) else grad_norm)

        dev_predictions, dev_loss = _predict_dev(
            model=model,
            task_inputs=dev_task_inputs,
            labels=dev_labels,
            graphs=dev_graphs,
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            batch_size=training_config.batch_size,
            device=device,
        )
        dev_rows = evaluate_results(dev_predictions, dev_labels, dev_graphs)
        dev_row = dev_rows[0]
        dev_metric = _best_metric(dev_row)
        if dev_metric > best_metric:
            best_metric = dev_metric
            best_epoch = epoch
            best_state = _cpu_state_dict(model)

        metric_records.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss_total / train_sample_count if train_sample_count else 0.0,
                "dev_loss": dev_loss,
                "dev_recall_at_5": float(dev_row["Recall@5"]),
                "dev_full_support_at_5": float(dev_row["Full Support@5"]),
                "dev_full_support_at_10": float(dev_row["Full Support@10"]),
                "dev_mrr": float(dev_row["MRR"]),
                "best_dev_metric": best_metric,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": last_grad_norm,
                "positive_count": positive_count,
                "negative_count_by_type": negative_count_by_type,
            }
        )

    result = TrainableTrainingResult(
        model_config=model_config,
        training_config=training_config,
        metric_records=metric_records,
        best_model_state_dict=best_state,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        best_epoch=best_epoch,
        global_step=global_step,
        best_dev_metric=best_metric,
    )
    if checkpoint_callback is not None:
        checkpoint_callback(result)
    return result


def _predict_dev(
    *,
    model: EvidenceScoringModel,
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    model_config: TrainableModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    batch_size: int,
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    batches = build_full_ranking_batches(
        task_inputs=task_inputs,
        graphs=graphs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=batch_size,
        labels=labels,
    )
    labels_by_task_id = {label["task_id"]: label for label in labels}
    graph_by_task_id = {graph["task_id"]: graph for graph in graphs}
    logits_by_task_id: dict[str, list[RankedNode]] = defaultdict(list)
    loss_total = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved_batch = move_training_batch(batch, device)
            logits = model(moved_batch)
            loss = F.binary_cross_entropy_with_logits(logits, moved_batch.labels)
            loss_total += float(loss.detach().cpu()) * int(moved_batch.labels.shape[0])
            sample_count += int(moved_batch.labels.shape[0])
            for task_id, node_id, score in zip(batch.sample_task_ids, batch.sample_node_ids, logits.detach().cpu().tolist()):
                logits_by_task_id[task_id].append(RankedNode(node_id=node_id, score=float(score)))

    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        ranked_nodes = sorted(logits_by_task_id[task_id], key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:10]]
        visible_graph = model_visible_graph(graph_by_task_id[task_id], frozenset(model_config.enabled_edge_types))
        subgraph = induced_retrieved_subgraph(visible_graph, top_node_ids)
        predictions.append(
            {
                "task_id": task_id,
                "method": model_config.method_name,
                "ranked_nodes": [{"node_id": node.node_id, "score": node.score} for node in ranked_nodes],
                "retrieved_subgraph": subgraph,
                "latency_ms": 0.0,
                "input_tokens": 0,
            }
        )
        if not set(labels_by_task_id[task_id]["gold_evidence_nodes"]):
            raise ValueError(f"Dev labels must contain gold evidence nodes for task_id={task_id}.")
    return predictions, loss_total / sample_count if sample_count else 0.0


def _best_metric(row: MetricRow) -> float:
    return 0.50 * _metric_float(row, "Full Support@5") + 0.30 * _metric_float(row, "Recall@5") + 0.20 * _metric_float(row, "MRR")


def _metric_float(row: MetricRow, key: str) -> float:
    value = row[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Metric column must be numeric for best checkpoint selection: {key}")
    return float(value)


def _pos_weight(train_pairs: list[TrainPairRecord], device: torch.device) -> Tensor:
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)
    negative_count = sum(1 for pair in train_pairs if pair["label"] == 0)
    if positive_count == 0:
        raise ValueError("pos_weight requires at least one positive sample.")
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)


def _negative_count_by_type(train_pairs: list[TrainPairRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter(pair["sample_type"] for pair in train_pairs if pair["label"] == 0)
    return dict(sorted(counter.items()))


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in deepcopy(model.state_dict()).items()}
