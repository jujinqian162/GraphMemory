# Phase 2 Trainable Retriever Contracts

Date: 2026-05-27

Status: Historical Phase 2 planning contract. The maintained project-level contracts now live in `data-contracts.md`, `retrieval-contracts.md`, and `model-contracts.md`.

This document is preserved for Phase 2 design provenance. Do not treat it as the current source of truth for new work; promote stable schema, retrieval, or model changes into the project-level contract documents.

## Contract Principles

- Scripts own file IO. Library modules receive parsed records, validated records, config objects, or tensors.
- JSON artifacts use `TypedDict` records plus fail-fast validators.
- Internal training and tensorization state uses small frozen dataclasses.
- Torch model code must not receive raw JSON dictionaries.
- Field order that affects tensor meaning must be explicit, stable, and saved in checkpoint config.
- Training-only artifacts may contain labels; retrieval-only code must not read labels or train pairs.
- Any schema or config violation should fail fast with a clear exception.

## Bilingual Type Documentation Rule

Every concrete Phase 2 type must include a complete bilingual docstring using Python triple-quoted docstrings.

This applies to:

- `TypedDict` artifact records.
- frozen dataclass configs and internal records.
- concrete model, tensorizer, feature builder, sampler, registry, and retriever classes.
- public `Protocol` interfaces that define a replaceable behavior boundary.

Each docstring must describe:

- the type purpose in English and Chinese.
- every field or public method in English and Chinese.
- index semantics for tensors when relevant.
- whether the type belongs to disk artifact, library-core state, or model input.

Required style:

```python
class TrainPairRecord(TypedDict):
    """
    One training pair artifact row for a query-node supervision example.
    一个 query-node 监督样本对应的训练 pair artifact 行。

    Fields / 字段:
    - task_id: Task join key matching memory task, label, and graph artifacts.
      task_id：任务 join key，必须匹配 memory task、label 和 graph artifact。
    - node_id: Memory node id being supervised; must not be the question node `q`.
      node_id：被监督的 memory node id；不能是问题节点 `q`。
    - label: Binary evidence label, where 1 means gold evidence and 0 means sampled negative.
      label：二分类 evidence 标签，1 表示 gold evidence，0 表示采样负例。
    - sample_type: Sampling source used to create this row.
      sample_type：生成该样本行时使用的采样来源。
    """

    task_id: TaskId
    node_id: NodeId
    label: Literal[0, 1]
    sample_type: TrainPairSampleType
```

Do not rely on ad hoc inline comments as the only field explanation. Inline comments may clarify implementation details, but the complete field meaning belongs in the docstring and this contract document.

## Retrieval Method Registry

Phase 2 must use the existing static lightweight registry in `graph_memory/retrieval_registry.py`; scattered `method in {...}` checks are not allowed. This is not a plugin system and should not do dynamic discovery.

```python
@dataclass(frozen=True)
class RetrievalMethodSpec:
    """
    Static metadata for one public retrieval method.
    一个公开检索方法的静态元数据。

    Fields / 字段:
    - name: Public method name written into ranked result artifacts.
      name：写入 ranked result artifact 的公开方法名。
    - requires_graphs: Whether this method requires `*_graphs.json`.
      requires_graphs：该方法是否需要 `*_graphs.json`。
    - requires_graph_config: Whether this method requires graph rerank config.
      requires_graph_config：该方法是否需要 graph rerank config。
    - requires_checkpoint: Whether this method requires a trainable model checkpoint.
      requires_checkpoint：该方法是否需要可训练模型 checkpoint。
    - requires_dense_encoder: Whether this method needs dense encoder runtime args.
      requires_dense_encoder：该方法是否需要 dense encoder 运行参数。
    - seed_method: Optional flat seed method used by this method, such as `dense`.
      seed_method：该方法使用的可选 flat seed method，例如 `dense`。
    - builder_id: Local runtime builder selected by `graph_memory.retrieval.factory`.
      builder_id：由 `graph_memory.retrieval.factory` 选择的本地运行时 builder。
    """

    name: MethodName
    requires_graphs: bool
    requires_graph_config: bool
    requires_checkpoint: bool
    requires_dense_encoder: bool
    seed_method: MethodName | None
    builder_id: str
```

Registry rules:

- supported methods and CLI `choices` are derived from `METHOD_REGISTRY.keys()`.
- Graph-rerank and dense-encoder method sets are derived from registry capability fields, not method-name string matching.
- `dense_rgcn_graph_retriever` must be registered through the same registry as Phase 1 methods.
- Runtime builders live in `graph_memory/retrieval/factory.py`; trainable graph retrieval is adapted through `graph_memory/retrieval/methods/trainable_graph.py`.
- Registry entries declare required inputs; scripts should reject missing graphs, graph configs, or checkpoints before invoking core retrieval.

Current registry entries before Phase 2:

| Method | Graphs | Graph config | Checkpoint | Dense encoder args | Seed method |
|---|---:|---:|---:|---:|---|
| `bm25` | no | no | no | no | none |
| `dense` | no | no | no | yes | none |
| `bm25_graph_rerank` | yes | yes | no | no | `bm25` |
| `dense_graph_rerank` | yes | yes | no | yes | `dense` |

Phase 2 adds:

| Method | Graphs | Graph config | Checkpoint | Dense encoder args | Seed method |
|---|---:|---:|---:|---:|---|
| `dense_rgcn_graph_retriever` | yes | no | yes | yes | `dense` |

## Train Pair Artifact

File pattern:

```text
data/hotpotqa/processed/{split}_pairs.json
```

Producer:

```text
scripts/build_train_pairs.py
```

Consumers:

```text
scripts/train_graph_retriever.py
graph_memory.training_pairs
```

Shape:

```json
[
  {
    "task_id": "hotpot_000001",
    "node_id": "m7",
    "label": 1,
    "sample_type": "positive"
  }
]
```

Type definitions:

```python
TrainPairSampleType = Literal[
    "positive",
    "easy_random",
    "hard_bm25",
    "hard_dense",
    "hard_graph_neighbor",
]


class TrainPairRecord(TypedDict):
    """
    One training pair artifact row for a query-node supervision example.
    一个 query-node 监督样本对应的训练 pair artifact 行。

    Fields / 字段:
    - task_id: Task join key matching memory task, label, and graph artifacts.
      task_id：任务 join key，必须匹配 memory task、label 和 graph artifact。
    - node_id: Memory node id being supervised; must not be the question node `q`.
      node_id：被监督的 memory node id；不能是问题节点 `q`。
    - label: Binary evidence label, where 1 means gold evidence and 0 means sampled negative.
      label：二分类 evidence 标签，1 表示 gold evidence，0 表示采样负例。
    - sample_type: Sampling source used to create this row.
      sample_type：生成该样本行时使用的采样来源。
    """

    task_id: TaskId
    node_id: NodeId
    label: Literal[0, 1]
    sample_type: TrainPairSampleType
```

Validation rules:

- `task_id` must exist in input, label, and graph artifacts.
- `node_id` must be a memory node in the task, never `q`.
- `label=1` rows must exactly come from `gold_evidence_nodes`.
- `label=0` rows must not include any gold evidence node.
- `sample_type="positive"` requires `label=1`.
- All other sample types require `label=0`.
- Duplicate `(task_id, node_id, sample_type)` rows are invalid.
- Unknown top-level fields are invalid unless the contract is explicitly extended.

## Negative Sampling Config And Summary

The pair builder must use a typed config rather than free-form dict access.

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

```python
class TrainPairBuildSummary(TypedDict):
    """
    Summary record written beside train pair artifacts for reproducibility.
    写在 train pair artifact 旁边、用于复现性的汇总记录。

    Fields / 字段:
    - positive_count: Number of positive rows.
      positive_count：正例行数。
    - negative_count_by_type: Negative row counts grouped by sample type.
      negative_count_by_type：按 sample type 分组的负例行数。
    - avg_positive_per_task: Average positive rows per task.
      avg_positive_per_task：每个 task 的平均正例行数。
    - avg_negative_per_task: Average negative rows per task.
      avg_negative_per_task：每个 task 的平均负例行数。
    - tasks_with_no_positive: Task ids that had no gold evidence; must be empty.
      tasks_with_no_positive：没有 gold evidence 的 task id；必须为空。
    - sampling_config: Effective negative sampling config.
      sampling_config：实际生效的负采样配置。
    """

    positive_count: int
    negative_count_by_type: dict[str, int]
    avg_positive_per_task: float
    avg_negative_per_task: float
    tasks_with_no_positive: list[TaskId]
    sampling_config: dict[str, object]
```

## Seed Signal Contract

Seed signal is the frozen baseline retriever signal used before learned graph scoring. In the first trainable retriever, the default seed signal comes from dense retrieval.

The same seed signal provider must be used for:

- hard dense negative sampling.
- node numeric feature construction.
- trainable retrieval inference.

```python
@dataclass(frozen=True)
class SeedSignal:
    """
    Frozen seed retrieval signal for one memory node.
    一个 memory node 的冻结初始检索信号。

    Fields / 字段:
    - node_id: Memory node id receiving this seed signal.
      node_id：该 seed signal 对应的 memory node id。
    - score: Raw seed retriever score, dense cosine similarity for the default dense provider.
      score：seed retriever 原始分数；默认 dense provider 中为 dense cosine similarity。
    - rank: One-based rank after sorting by descending score and ascending node id tie-break.
      rank：从 1 开始的排名；按 score 降序、node id 升序打破平局。
    - rank_percentile: Rank percentile in [0, 1], where 0 means best and 1 means worst.
      rank_percentile：范围 [0, 1] 的排名百分位，0 表示最好，1 表示最差。
    """

    node_id: NodeId
    score: float
    rank: int
    rank_percentile: float
```

Rank percentile rule:

```text
rank_percentile = 0.0 if num_memory_nodes == 1
rank_percentile = (rank - 1) / (num_memory_nodes - 1) otherwise
```

```python
class SeedSignalProvider(Protocol):
    """
    Replaceable provider for frozen seed retrieval signals.
    可替换的冻结初始检索信号提供器。

    Methods / 方法:
    - score_task: Return one SeedSignal for every memory node in the task.
      score_task：为 task 中每个 memory node 返回一个 SeedSignal。
    """

    def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
        ...
```

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

## Graph Batch Contract

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

## Relation Vocab Contract

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
- A `directed=true` graph edge emits only a forward message edge in the first implementation.
- `relation_id` indexes the ordered relation vocab; it is not the same thing as graph artifact `edge_type`.
- Disabled edge types are filtered before message edge expansion.

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
    - ablation_name: Canonical experiment/ablation name.
      ablation_name：规范化的实验或 ablation 名称。
    """

    method_name: MethodName
    encoder_model: str
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
    ablation_name: str
```

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
    """

    optimizer_name: str
    learning_rate: float
    batch_size: int
    max_grad_norm: float
    random_seed: int
    pos_weight_enabled: bool
```

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

## Dev Evaluation Contract

The first implementation should not create a formal `dev_pairs.json` artifact.

Dev evaluation during training should:

1. Score every memory node for each dev task.
2. Build in-memory `RankedResult` records.
3. Use existing retrieval metrics against dev labels and graphs.
4. Select `best.pt` by the configured retrieval metric.

If dev BCE loss is needed, it should be computed from full-node labels derived in memory from `dev_memory_tasks.labels.json`, not from a separate dev pairs artifact.

## Validators

Phase 2 should add:

```text
validate_train_pairs(records, inputs_by_task_id, labels_by_task_id, graphs_by_task_id)
validate_negative_sampling_config(config)
validate_train_pair_build_summary(summary)
validate_trainable_model_config(config)
validate_trainable_training_config(config)
validate_trainable_checkpoint_metadata(checkpoint)
```

Validators must not repair, sort, drop, or infer data. Transformation belongs in explicitly named builder functions.
