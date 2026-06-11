# Model Contracts

Status: Maintained project-level reference.

This document defines trainable model configs, tensor batch contracts, relation vocab rules, checkpoint metadata, and training evaluation behavior. Disk artifacts live in `data-contracts.md`; public retrieval method dispatch lives in `retrieval-contracts.md`.

## Scope

Model code may consume:

- validated memory task inputs.
- validated graphs.
- validated train pair records during training.
- explicit config dataclasses.
- tensors built by named tensorizer functions.

Model code must not consume:

- raw JSON dictionaries directly.
- labels during retrieval inference.
- script-level path or CLI config objects.

## Trainable Model Config

```python
@dataclass(frozen=True)
class TrainableModelConfig:
    """
    Minimal model reconstruction config saved in every trainable checkpoint.
    每个可训练 checkpoint 中保存的最小模型重建配置。

    Fields / 字段:
    - method_name: Public retrieval method name.
      method_name：公开检索方法名。
    - encoder_model: Frozen text encoder model name.
      encoder_model：冻结文本 encoder 的模型名。
    - encoder_dim: Frozen text embedding dimension used by the model input projection.
      encoder_dim：模型 input projection 使用的冻结文本 embedding 维度。
    - query_prefix: Prefix applied to query text before encoding.
      query_prefix：编码 query 文本前添加的前缀。
    - passage_prefix: Prefix applied to memory text before encoding.
      passage_prefix：编码 memory 文本前添加的前缀。
    - hidden_dim: Hidden dimension used by graph encoder and scorer.
      hidden_dim：graph encoder 和 scorer 使用的隐藏维度。
    - num_layers: Number of R-GCN layers; 0 means identity graph encoder.
      num_layers：R-GCN 层数；0 表示 identity graph encoder。
    - dropout: Dropout probability.
      dropout：dropout 概率。
    - feature_config: Ordered node and scorer feature names.
      feature_config：有序的 node 和 scorer 特征名。
    - relation_vocab: Ordered relation names used by relation ids.
      relation_vocab：relation id 使用的有序 relation 名称。
    - graph_encoder_type: Graph encoder component name, such as `rgcn` or `identity`.
      graph_encoder_type：graph encoder 组件名，例如 `rgcn` 或 `identity`。
    - message_transform_type: Relation transform component name, such as `typed` or `shared`.
      message_transform_type：relation transform 组件名，例如 `typed` 或 `shared`。
    - edge_weight_policy: Edge weight policy name, such as `artifact` or `uniform`.
      edge_weight_policy：edge weight policy 名称，例如 `artifact` 或 `uniform`。
    - enabled_edge_types: Ordered graph artifact edge types enabled during tensorization.
      enabled_edge_types：tensorization 时启用的有序 graph artifact edge type。
    - ablation_name: Canonical experiment or ablation name.
      ablation_name：规范化的实验或 ablation 名称。
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
```

Rules:

- Any field that changes tensor dimensions or score semantics must be saved in checkpoint metadata.
- `method_name` must match the public retrieval method used for inference.
- `num_layers=0` is the canonical identity graph encoder ablation.
- `ablation_name` is a stable experiment label, not an arbitrary run note.
- `encoder_dim` and `enabled_edge_types` are required because they affect model reconstruction and graph tensorization.

## Training Config

```python
@dataclass(frozen=True)
class TrainableTrainingConfig:
    """
    Minimal training config needed to resume or audit a trainable run.
    用于恢复或审计可训练运行的最小训练配置。

    Fields / 字段:
    - optimizer_name: Optimizer name, default `AdamW`.
      optimizer_name：优化器名称，默认 `AdamW`。
    - learning_rate: Graph/scorer learning rate.
      learning_rate：graph/scorer 学习率。
    - batch_size: Number of task graphs per training batch.
      batch_size：每个 training batch 中的 task graph 数量。
    - max_grad_norm: Gradient clipping maximum norm.
      max_grad_norm：梯度裁剪最大 norm。
    - random_seed: Run-level random seed.
      random_seed：运行级随机种子。
    - pos_weight_enabled: Whether BCE positive weighting was enabled.
      pos_weight_enabled：是否启用 BCE 正例权重。
    - epochs: Number of training epochs.
      epochs：训练 epoch 数量。
    """

    optimizer_name: str
    learning_rate: float
    batch_size: int
    max_grad_norm: float
    random_seed: int
    pos_weight_enabled: bool
    epochs: int
```

Rules:

- Training config records effective values after defaults and CLI overrides.
- Training config is not a replacement for run summary; run summary still records paths, counts, timings, and environment notes.
- `TrainableTrainingConfig.batch_size` counts task graphs. It is independent from `DenseEncoderSettings.batch_size`, which controls sentence-transformer text mini-batches.

## Negative Sampling Config

Negative sampling config belongs to artifact production, but its typed representation is consumed by training preparation code.

```python
@dataclass(frozen=True)
class NegativeSamplingConfig:
    """
    Configuration for deterministic train pair negative sampling.
    确定性训练 pair 负采样配置。

    Fields / 字段:
    - random_seed: Seed used by random negative sampling and tie-breaking.
      random_seed：随机负采样和 tie-breaking 使用的种子。
    - easy_random_per_positive: Number of easy random negatives per positive node.
      easy_random_per_positive：每个正例对应的 easy random 负例数量。
    - hard_bm25_per_positive: Number of hard BM25 negatives per positive node.
      hard_bm25_per_positive：每个正例对应的 hard BM25 负例数量。
    - hard_dense_per_positive: Number of hard dense negatives per positive node.
      hard_dense_per_positive：每个正例对应的 hard dense 负例数量。
    - hard_graph_neighbor_per_positive: Number of graph-neighbor negatives per positive node.
      hard_graph_neighbor_per_positive：每个正例对应的 graph-neighbor 负例数量。
    - hard_pool_size: Top-ranked non-gold candidate pool size for hard retriever negatives.
      hard_pool_size：hard retriever 负例采样时使用的非 gold top-ranked 候选池大小。
    """

    random_seed: int = 13
    easy_random_per_positive: int = 2
    hard_bm25_per_positive: int = 2
    hard_dense_per_positive: int = 2
    hard_graph_neighbor_per_positive: int = 1
    hard_pool_size: int = 30
```

Rules:

- Counts must be non-negative integers.
- `hard_pool_size` must be positive when any hard retriever negative count is positive.
- Sampling must be deterministic for a fixed config and input artifact set.

## Feature Config

Feature order is part of the model contract. Any config that changes feature names changes tensor dimensions and must be saved in checkpoints.

```python
@dataclass(frozen=True)
class NodeFeatureConfig:
    """
    Ordered numeric node feature configuration.
    有序的节点数值特征配置。

    Fields / 字段:
    - node_feature_names: Ordered features concatenated with text embeddings before input projection.
      node_feature_names：input projection 前与文本 embedding 拼接的有序特征名。
    - scorer_feature_names: Ordered direct numeric features passed to the evidence scorer.
      scorer_feature_names：直接传入 evidence scorer 的有序数值特征名。
    """

    node_feature_names: tuple[str, ...]
    scorer_feature_names: tuple[str, ...]
```

Default full model feature names:

```text
node_feature_names = ("seed_score", "seed_rank_percentile", "is_question_node")
scorer_feature_names = ("seed_score", "seed_rank_percentile")
```

For `w/o seed score`:

```text
node_feature_names = ("is_question_node",)
scorer_feature_names = ()
```

Rules:

- `node_feature_names` controls `GraphBatch.node_features` column order.
- `scorer_feature_names` controls `TrainingBatch.sample_node_features` column order.
- Unknown feature names must fail validation.
- Feature builders should consume `SeedSignalProvider` from `retrieval-contracts.md` instead of recomputing seed ranks independently.

## Graph Batch

All tensor indexes in `GraphBatch` and `TrainingBatch` are batch-flattened global indexes unless explicitly documented otherwise.

```python
@dataclass(frozen=True)
class GraphBatch:
    """
    Tensorized batch of one or more task graphs for message passing.
    一个或多个 task graph 的 message passing 张量化 batch。

    Fields / 字段:
    - node_embeddings: Tensor `[total_nodes, encoder_dim]` with frozen query and memory text embeddings.
      node_embeddings：形状为 `[total_nodes, encoder_dim]` 的冻结 query 和 memory 文本 embedding。
    - node_features: Tensor `[total_nodes, node_feature_dim]` with ordered numeric node features.
      node_features：形状为 `[total_nodes, node_feature_dim]` 的有序节点数值特征。
    - edge_index: Long tensor `[2, num_message_edges]`; row 0 is source, row 1 is target.
      edge_index：Long tensor，形状为 `[2, num_message_edges]`；第 0 行是 source，第 1 行是 target。
    - relation_ids: Long tensor `[num_message_edges]` indexing the saved relation vocab.
      relation_ids：Long tensor，形状为 `[num_message_edges]`，索引保存的 relation vocab。
    - edge_weights: Float tensor `[num_message_edges]` with tensorizer-produced message edge weights.
      edge_weights：Float tensor，形状为 `[num_message_edges]`，来自 tensorizer 的 message edge 权重。
    - query_node_indices: Long tensor `[num_tasks]` containing each task question node global index.
      query_node_indices：Long tensor，形状为 `[num_tasks]`，包含每个 task 的问题节点全局 index。
    - task_node_offsets: Python list of length `num_tasks + 1`; start inclusive, end exclusive.
      task_node_offsets：长度为 `num_tasks + 1` 的 Python list；起点包含，终点不包含。
    - task_ids: Task ids in the same order as `task_node_offsets`.
      task_ids：与 `task_node_offsets` 顺序一致的 task id。
    - node_ids_by_task: Per-task node ids in local tensorization order; `q` must be present.
      node_ids_by_task：每个 task 内按本地张量化顺序排列的 node id；必须包含 `q`。
    """

    node_embeddings: Tensor
    node_features: Tensor
    edge_index: Tensor
    relation_ids: Tensor
    edge_weights: Tensor
    query_node_indices: Tensor
    task_node_offsets: list[int]
    task_ids: list[TaskId]
    node_ids_by_task: list[list[NodeId]]
```

Rules:

- The question node `q` is included for message passing but is never a candidate ranked memory node.
- `task_node_offsets[i] <= global_node_index < task_node_offsets[i + 1]` defines task membership.
- `node_ids_by_task[i]` has length `task_node_offsets[i + 1] - task_node_offsets[i]`.
- No raw artifact dictionary should be passed into the model forward path.
- All tasks in one graph batch request frozen text features through one provider bulk operation.
- When the same joint dense provider supplies embeddings and seed signals, both values come from one normalized encoder result.
- When providers are different objects or lack the joint capability, each provider is called through its own bulk-or-single compatibility path.

## Training Batch

```python
@dataclass(frozen=True)
class TrainingBatch:
    """
    Supervised sample batch over a tensorized graph batch.
    基于张量化 graph batch 的监督样本 batch。

    Fields / 字段:
    - graph_batch: Shared graph tensor batch used once for message passing.
      graph_batch：共享的 graph tensor batch，用于一次 message passing。
    - sample_node_indices: Long tensor `[num_samples]` with batch-flattened memory node indexes.
      sample_node_indices：Long tensor，形状为 `[num_samples]`，包含 batch-flattened memory node index。
    - sample_query_indices: Long tensor `[num_samples]` with matching question node indexes.
      sample_query_indices：Long tensor，形状为 `[num_samples]`，包含匹配的问题节点 index。
    - sample_node_features: Float tensor `[num_samples, scorer_feature_dim]` for direct scorer features.
      sample_node_features：Float tensor，形状为 `[num_samples, scorer_feature_dim]`，用于 scorer 直接特征。
    - labels: Float tensor `[num_samples]` with binary labels.
      labels：Float tensor，形状为 `[num_samples]`，包含二分类标签。
    - sample_task_ids: Task id for each sample, used for debug and metric grouping.
      sample_task_ids：每个 sample 对应的 task id，用于 debug 和 metric 分组。
    - sample_node_ids: Memory node id for each sample, used for debug and validation.
      sample_node_ids：每个 sample 对应的 memory node id，用于 debug 和 validation。
    - sample_types: Sampling source for each sample.
      sample_types：每个 sample 的采样来源。
    """

    graph_batch: GraphBatch
    sample_node_indices: Tensor
    sample_query_indices: Tensor
    sample_node_features: Tensor
    labels: Tensor
    sample_task_ids: list[TaskId]
    sample_node_ids: list[NodeId]
    sample_types: list[TrainPairSampleType]
```

Rules:

- `sample_node_indices` must point to memory nodes, never `q`.
- `sample_query_indices` must point to the matching task's `q` node.
- `labels` uses float dtype for BCE compatibility but only contains `0.0` or `1.0`.
- `sample_*` metadata is kept for debug and validation; model math should use tensor fields.

## Relation Vocab

Relation vocab is ordered and must be saved in checkpoints.

Default relation vocab:

```text
(
  "query_overlap_forward",
  "sequential_forward",
  "sequential_reverse",
  "entity_overlap_forward",
  "entity_overlap_reverse",
  "bridge_forward",
  "bridge_reverse",
)
```

Rules:

- `edge_index[0]` is message source and `edge_index[1]` is message target.
- A `directed=false` graph edge emits forward and reverse message edges.
- A `directed=true` graph edge emits only a forward message edge in the first trainable implementation.
- `relation_id` indexes the ordered relation vocab; it is not the same thing as graph artifact `edge_type`.
- Disabled edge types are filtered before message edge expansion.

## Checkpoint Contract

Checkpoint files are PyTorch checkpoint dictionaries, not JSON artifacts. Their metadata still follows a documented schema.

Required top-level keys:

```text
checkpoint_version
method_name
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
global_step
best_dev_metric
model_config
training_config
created_at
```

Rules:

- `model_config.feature_config` and `model_config.relation_vocab` are required for inference.
- Loading must fail if checkpoint `method_name` does not match the requested retrieval method.
- Loading must fail if config values needed to reconstruct dimensions are missing.
- `best.pt` is used for retrieval inference.
- Epoch checkpoints may be used for resume and debugging.
- Checkpoint loading belongs in `graph_memory.models.graph_retriever.checkpoint` or trainable retrieval builders, not in generic artifact IO.

## Dev Evaluation During Training

The first trainable implementation should not create a formal `dev_pairs.json` artifact.

Dev evaluation during training should:

1. Score every memory node for each dev task.
2. Build in-memory ranked result records.
3. Use existing retrieval metrics against dev labels and graphs.
4. Select `best.pt` by the configured retrieval metric.

Frozen full-ranking dev batches are constructed once before the epoch loop and retained as CPU values for the training invocation. Every epoch still moves a separate batch value to the target device and recomputes model logits, loss, retrieval metrics, and checkpoint selection. This reuse is invocation-scoped: it does not create a process-global cache or a persistent embedding artifact.

If dev BCE loss is needed, compute it from full-node labels derived in memory from `dev_memory_tasks.labels.json`, not from a separate dev pairs artifact.

## Validators

Recommended validators:

```text
validate_negative_sampling_config(config)
validate_trainable_model_config(config)
validate_trainable_training_config(config)
validate_graph_batch(batch)
validate_training_batch(batch)
validate_trainable_checkpoint_metadata(checkpoint)
```

Validation rules:

- Validators do not mutate tensors or records.
- Validators do not infer missing config fields.
- Shape checks must mention the field name and expected shape in error messages.
- Config validators should run before expensive model or encoder initialization.
