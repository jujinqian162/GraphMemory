# Phase 2 R-GCN Trainable Retriever Plan

Date: 2026-05-27

Status: Discussion plan. This document records the current Phase 2 R-GCN design for review before implementation. It is not yet a committed implementation task list.

## 目标

实现第一版可训练图检索器：`dense-seeded R-GCN binary node scorer`。

它要回答的问题是：

```text
给定 query、已经构建好的 task graph、以及每个 memory node 的文本和初始检索信号，
模型应该给每个 memory node 打一个 evidence logit。
```

训练目标是二分类 evidence node scorer：

- gold evidence node 的 label 为 `1`。
- sampled negative node 的 label 为 `0`。
- loss 使用 `BCEWithLogitsLoss`。
- 推理时对所有 memory nodes 的 logit 排序，得到 top-k。

它不训练：

- 图构建规则。
- top-k 排序过程。
- Phase 1 手写 graph rerank 里的 `lambda_*` 权重。
- 默认冻结的 sentence-transformer text encoder。

## 已确认设计

- 第一版训练目标采用 binary node scorer，不做 pairwise/listwise ranking loss。
- text encoder 第一版默认 frozen。
- `train_pairs.json` 采用 query-node 样本格式。
- 新增 `graph_memory/learned/` 子包，避免把训练逻辑塞进 Phase 1 的 `retrieval.py` 或 `rerank.py`。
- R-GCN 不复用 `graph_memory/rerank.py` 的手写图分数组件。
- R-GCN 可以复用 seed retriever 的初始语义信号，尤其是 dense score/rank feature。
- ablation 通过结构性组件替换或 tensorization 过滤完成，不在模型 forward 里堆叠大量 `if` 分支。
- Phase 2 的稳定契约维护在项目级 contract 文档中：artifact schema 见 `docs/20-contracts/data-contracts.md`，retrieval method 和 seed signal 见 `docs/20-contracts/retrieval-contracts.md`，batch/config/checkpoint 见 `docs/20-contracts/model-contracts.md`。本 plan 不重复维护完整字段表。
- `scripts/*` 负责文件 IO；`graph_memory/learned/*` 只接收已读取、已验证或待验证的 Python 对象和 tensor，不直接读写训练输入 artifact。
- 第一版不新增正式 `dev_pairs.json` artifact；dev evaluation 对所有 memory nodes 做 full ranking。
- retrieval method 接入改为轻量静态 registry，避免继续扩散 `method in {...}` 判断。
- Phase 2 新增的具体类型必须使用完整双语三引号 docstring 解释用途和字段含义。

## 非目标

第一版不实现：

- GAT 或 R-GCN+GAT hybrid。
- 训练 graph construction。
- end-to-end fine-tune sentence-transformer。
- contrastive InfoNCE。
- pairwise margin ranking loss。
- listwise top-k differentiable ranking。
- persistent embedding cache 作为正式 artifact contract。
- distributed training。
- 动态插件发现或复杂 method plugin registry。

这些可以在第一版 R-GCN 跑通、可复现、可评估后再加。

## 与现有 Phase 1 的关系

当前 Phase 1 已经有三个稳定边界：

- `*_memory_tasks.input.json`：retrieval 和 graph construction 可见输入。
- `*_graphs.json`：确定性图构建产物。
- `ranked_results_{method}.json`：统一检索输出。

R-GCN 应该作为新的 `RetrievalMethod` 接入，而不是改造 `graph_memory/rerank.py`。

建议方法名：

```text
dense_rgcn_graph_retriever
```

Phase 2 对比时，方法关系应是并列的：

```text
dense                       # frozen dense flat baseline
dense_graph_rerank          # Phase 1 handwritten graph rerank
dense_rgcn_graph_retriever  # Phase 2 learned graph scorer
```

`dense_rgcn_graph_retriever` 的训练和推理可以使用 dense seed score 作为 node feature，但不能调用 Phase 1 graph rerank 的 `neighbor_propagation_scores`、`bridge_edge_scores` 或 `lambda_*` 组合公式。

### IO 边界

第一版不新增专门的 IO 包，也不新增 `graph_memory/learned/io.py`。现有 `graph_memory/io.py` 已经覆盖 JSON/CSV/JSONL 通用读写，Phase 2 只需要继续复用它。

边界如下：

- `scripts/*` 负责 CLI、路径、`read_json` / `write_json` / `write_jsonl`、run summary。
- `graph_memory/io.py` 负责通用文件 helper。
- `graph_memory/learned/data.py` 接收已读取对象，做 join、validation-facing transformation、batch example 构造。
- `graph_memory/learned/training.py` 不读写训练 artifact，只接收训练数据结构和 config。
- `graph_memory/learned/checkpoint.py` 可以封装 checkpoint save/load，因为 checkpoint 是模型运行状态，不是通用 artifact IO。

### method registry 调整

当前 Phase 1 的 `method in {...}` 判断在 Phase 2 会变脆，因为 learned retriever 需要 checkpoint，而 flat / graph rerank method 的输入要求不同。Phase 2 应新增轻量静态 `RetrievalMethodSpec` registry：

```text
name
requires_graphs
requires_graph_config
requires_checkpoint
seed_method
builder
```

实现要求：

- `SUPPORTED_METHODS` 从 registry keys 派生。
- CLI `choices` 从 registry keys 派生。
- `dense_rgcn_graph_retriever` 通过同一个 registry 注册。
- registry 是静态表，不做动态插件发现。
- 初版可以把 registry 放在 `graph_memory/retrieval.py`；如果 learned import 造成依赖变重或循环，再抽到 `graph_memory/retrieval_registry.py`。
- learned retriever 的 builder 可以 lazy import `graph_memory.learned.inference`，避免 Phase 1-only 使用路径提前加载训练依赖。

## 数据流

### 训练输入

R-GCN 训练需要：

```text
train_memory_tasks.input.json
train_memory_tasks.labels.json
train_graphs.json
train_pairs.json
```

dev 评估需要：

```text
dev_memory_tasks.input.json
dev_memory_tasks.labels.json
dev_graphs.json
```

其中 `train_graphs.json` 和 `dev_graphs.json` 复用当前 `scripts/build_graphs.py` 的输出，不新增训练专用图构建器。

第一版不产出正式 `dev_pairs.json`。如果训练时需要 dev BCE loss，从 dev labels 在内存中生成 full-node labels；best checkpoint 仍按 full-node retrieval metric 选择。

### train_pairs 生成

新增脚本：

```text
scripts/build_train_pairs.py
```

输入：

```text
--tasks  *_memory_tasks.input.json
--labels *_memory_tasks.labels.json
--graphs *_graphs.json
--output *_pairs.json
```

输出采用 query-node 样本格式：

```json
[
  {
    "task_id": "hotpot_x",
    "node_id": "m7",
    "label": 1,
    "sample_type": "positive"
  },
  {
    "task_id": "hotpot_x",
    "node_id": "m3",
    "label": 0,
    "sample_type": "hard_dense"
  }
]
```

字段含义：

| Field | Meaning |
|---|---|
| `task_id` | join key，必须存在于 tasks/labels/graphs。 |
| `node_id` | memory node id，不允许是 `q`。 |
| `label` | `1` 表示 gold evidence，`0` 表示负样本。 |
| `sample_type` | `positive`、`easy_random`、`hard_bm25`、`hard_dense`、`hard_graph_neighbor` 等。 |

完整 artifact 类型、validator、负采样 config 和 summary 字段以 `docs/20-contracts/data-contracts.md` 和 `docs/20-contracts/model-contracts.md` 为准。

初版负样本策略：

```text
positive:
  gold_evidence_nodes 中的全部节点。

easy_random:
  非 gold memory nodes 中随机采样。

hard_bm25:
  BM25 排名靠前但非 gold 的节点。

hard_dense:
  frozen dense 排名靠前但非 gold 的节点。

hard_graph_neighbor:
  与 gold node 有 graph edge 连接但本身非 gold 的节点。
```

推荐比例：

```text
positive : easy_random : hard_bm25 : hard_dense : hard_graph_neighbor
= 1 : 2 : 2 : 2 : 1
```

比例不进入模型 forward，只影响 pair builder。

## Artifact 设计

### 新增训练 artifact

```text
*_pairs.json
checkpoints/<run_name>/checkpoint_epoch_{n}.pt
checkpoints/<run_name>/best.pt
runs/<run_name>/train_metrics.jsonl
runs/<run_name>/train_run_summary.json
```

### checkpoint 内容

checkpoint 应包含：

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

`best.pt` 用于推理；epoch checkpoint 用于恢复训练和排查。`model_config` 至少必须能恢复模型维度和语义：encoder 名称、文本 prefix、hidden dim、layer 数、dropout、feature names/order、relation vocab/order、graph encoder 类型、message transform 类型、edge weight policy、canonical ablation name。详细 schema 维护在 `docs/20-contracts/model-contracts.md`。

### train_metrics.jsonl

每个 epoch 写一行：

```json
{
  "epoch": 3,
  "global_step": 1200,
  "train_loss": 0.214,
  "dev_loss": 0.271,
  "dev_recall_at_5": 0.684,
  "dev_full_support_at_10": 0.402,
  "dev_mrr": 0.719,
  "learning_rate": 0.0002,
  "grad_norm": 1.83,
  "positive_count": 9800,
  "negative_count_by_type": {
    "easy_random": 19600,
    "hard_bm25": 19600,
    "hard_dense": 19600,
    "hard_graph_neighbor": 9800
  }
}
```

## 新增模块结构

建议新增：

```text
graph_memory/learned/
  __init__.py
  data.py
  features.py
  tensorize.py
  model.py
  checkpoint.py
  training.py
  inference.py
```

职责：

| Module | Responsibility |
|---|---|
| `data.py` | 接收已读取的 tasks/labels/graphs/pairs，按 task join，构造 typed training examples 和 batches；不做文件 IO。 |
| `features.py` | 构造 seed signal 和 node numeric features，例如 seed score、seed rank percentile、is_question。 |
| `tensorize.py` | 把 artifact graph 转成 message passing tensor。 |
| `model.py` | R-GCN、GraphEncoder 抽象、scorer MLP。 |
| `checkpoint.py` | checkpoint metadata validation、save/load helper；不读取训练数据 artifact。 |
| `training.py` | train/eval loop、loss、optimizer、checkpoint。 |
| `inference.py` | checkpoint 加载和 `RetrievalMethod` 包装。 |

不建议把这些内容加入：

- `graph_memory/rerank.py`，因为它属于透明手写 graph score。
- `graph_memory/indexes/dense.py`，因为它只应该负责 frozen dense retrieval。
- `scripts/run_retrieval.py` 的主体逻辑，脚本应保持薄适配层。

## 核心抽象

具体类型字段见 `docs/20-contracts/data-contracts.md`、`docs/20-contracts/retrieval-contracts.md` 和 `docs/20-contracts/model-contracts.md`；双语 docstring 规则见 `docs/20-contracts/README.md`。

### GraphEncoder

图编码器接口：

```python
class GraphEncoder(Protocol):
    def forward(self, batch: GraphBatch, node_states: Tensor) -> Tensor:
        ...
```

实现：

```text
IdentityGraphEncoder
RGCNGraphEncoder
```

未来新增：

```text
RelationalGATGraphEncoder
```

训练 loop 和 scorer 不依赖具体图编码器类型。

### EvidenceScoringModel

顶层模型：

```python
class EvidenceScoringModel(nn.Module):
    def forward(self, batch: TrainingBatch) -> Tensor:
        h0 = self.input_encoder(batch.text_embeddings, batch.node_features)
        h = self.graph_encoder(batch.graph_batch, h0)
        v = h[batch.sample_node_index]
        q = h[batch.sample_query_index]
        return self.scorer(v, q, batch.sample_node_features)
```

这里的 `forward` 不应该关心当前是否 w/o graph、w/o edge type、w/o bridge。差异由构造阶段选择不同组件和 tensorizer 完成。

### NodeFeatureBuilder

负责构造数值特征：

```text
seed_score
seed_rank_percentile
is_question_node
```

第一版默认使用 dense seed signal：

```text
seed_score = frozen dense cosine similarity
seed_rank_percentile = (dense_rank - 1) / max(1, num_memory_nodes - 1)
```

这里的 seed signal 指 frozen seed retriever 在 learned graph scorer 之前给每个 memory node 的初始检索信号。它不是 label，也不是 Phase 1 handwritten graph score。

需要单独抽象为 `SeedSignalProvider`，因为同一组 seed score/rank 会被三个地方复用：

- hard dense negative sampling。
- node numeric feature construction。
- trainable retrieval inference。

这三个路径必须共用同一个排序、prefix、normalization 和 tie-break 规则，避免训练与推理特征不一致。第一版 provider 可以只实现 dense seed signal。

如果做 w/o seed score，构造阶段换成不输出 seed 特征的 builder，模型输入维度随之变化。

### EdgeTensorizer

负责把 JSON graph edge 转为 message edge tensor。

输入：

```text
MemoryGraph
enabled_edge_types
reverse_relation_policy
edge_weight_policy
relation_vocab
```

输出：

```text
edge_index: [2, num_message_edges]
relation_id: [num_message_edges]
edge_weight: [num_message_edges]
```

命名区分：

| Term | Meaning |
|---|---|
| `graph_edge` | `*_graphs.json` 中的原始边。 |
| `message_edge` | tensorizer 展开后的有向消息边。 |
| `edge_type` | 原始 artifact edge type，例如 `bridge`。 |
| `relation_id` | R-GCN 实际使用的关系 ID，例如 `bridge_forward`。 |

### MessageTransform

R-GCN 内部关系变换：

```text
TypedRelationTransform
SharedRelationTransform
```

`TypedRelationTransform`：

```text
每个 relation_id 一个 W_r。
```

`SharedRelationTransform`：

```text
所有 relation_id 共用一个 W_message。
```

w/o edge type 时，不改训练流程，只把 R-GCN layer 内部的 transform 换成 shared 版本。

## R-GCN 算法设计

### 输入表示

每个 task graph 有：

```text
N = 1 + num_memory_nodes
node 0 or specific index = q
memory nodes = m0 ... m{k}
```

文本 embedding：

```text
text_emb_q = frozen_encoder(query)
text_emb_m = frozen_encoder(source + ". " + text)
```

模型输入：

```text
h0_v = InputProjection([text_embedding_v, numeric_features_v])
```

张量形状：

```text
text_embeddings: [num_nodes, encoder_dim]
node_features:   [num_nodes, feature_dim]
h0:              [num_nodes, hidden_dim]
h:               [num_nodes, hidden_dim]
```

### 单层 R-GCN

推荐命名：

```text
RelationalGraphConvLayer
```

公式：

```text
m_v = sum_r sum_{u in N_r(v)}
        norm(u, v, r) * edge_weight(u, v) * W_r h_u

h_next_v = LayerNorm(ReLU(W_self h_v + Dropout(m_v) + b))
```

其中：

```text
norm(u, v, r) = 1 / max(1, in_degree_r(v))
```

初版建议：

- 使用 2 层 R-GCN。
- hidden dim 默认 256。
- dropout 默认 0.1。
- 显式加入 reverse relation。
- `q` 节点参与 message passing。
- loss 只计算 memory nodes，不计算 `q`。

### relation vocab

原始 edge types：

```text
sequential
query_overlap
entity_overlap
bridge
```

message relation 建议：

```text
query_overlap_forward
sequential_forward
sequential_reverse
entity_overlap_forward
entity_overlap_reverse
bridge_forward
bridge_reverse
```

如果原始边 `directed=false`，tensorizer 生成 forward 和 reverse 两条 message edges。

如果原始边 `directed=true`，只生成 forward，除非该实验显式启用 reverse directed edges。第一版不启用 directed reverse。

## Scorer 设计

推荐命名：

```text
EvidenceNodeScorer
```

对每个 sampled memory node：

```text
features_v = [
  h_v,
  h_q,
  h_v * h_q,
  sample_node_features
]

logit_v = MLP(features_v)
```

MLP 初版：

```text
Linear(input_dim, hidden_dim)
ReLU
Dropout
Linear(hidden_dim, 1)
```

`sample_node_features` 可以包含 seed score 和 rank percentile。这样 dense seed signal 既能进入 input projection，也能作为 scorer 的直接特征。若实验证明重复输入没有帮助，再移除 scorer 侧直接特征。

## 训练流程

### Batch 单位

初版建议按 task graph batching，而不是把所有 pairs 独立打散成完全独立样本。

一个 batch 包含若干 task graphs：

```text
tasks: B
total_nodes = sum nodes over B
total_message_edges = sum message_edges over B
samples = selected query-node pair rows from these tasks
```

这样可以：

- 每个 task 的 R-GCN message passing 只算一次。
- 同一 task 中多个 positive/negative node 共享图编码结果。
- 避免同一图重复 forward 造成浪费。

`GraphBatch` 和 `TrainingBatch` 必须是 dataclass，不允许用裸 `dict` 传入 model。所有 tensor index 默认为 batch-flattened global index。具体字段定义维护在 `docs/20-contracts/model-contracts.md`。

### 训练步骤

```text
for batch in train_loader:
    logits = model(batch)
    labels = batch.labels.float()
    loss = BCEWithLogitsLoss(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    scheduler.step()
```

优化器：

```text
AdamW
```

默认学习率：

```text
1e-4 for graph/scorer parameters
```

如果以后 unfreeze encoder：

```text
1e-5 for encoder parameters
1e-4 for graph/scorer parameters
```

### Loss

初版：

```text
loss = BCEWithLogitsLoss(logits, labels)
```

如果正负比例不稳定，可以使用 `pos_weight`：

```text
pos_weight = num_negative / num_positive
```

但第一版建议先记录比例并观察 dev loss/metrics，再决定是否启用 `pos_weight`。启用时必须写入 run summary。

### Dev evaluation

每个 epoch 后：

```text
1. 对 dev 每个 task 的所有 memory nodes 计算 logits。
2. 排序生成 in-memory RankedResult。
3. 复用现有 evaluation metric 函数计算 Recall/F1/Full Support/MRR/connectivity。
4. 以主指标选择 best.pt。
```

推荐 best checkpoint 指标：

```text
0.50 * Full Support@5
+ 0.30 * Recall@5
+ 0.20 * MRR
```

这里不直接用 training BCE loss 选 best，因为最终目标是 evidence retrieval ranking。

## 推理接入

新增：

```text
graph_memory/learned/inference.py
```

核心类：

```python
class TrainableGraphRetriever:
    name = "dense_rgcn_graph_retriever"

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...
```

它做：

```text
1. 找到对应 graph。
2. 构造 frozen text embeddings 和 seed features。
3. tensorize graph。
4. 对所有 memory nodes 输出 logits。
5. 按 logit 降序返回完整 ranked_nodes。
6. 复用 induced top-k graph edge extraction 生成 retrieved_subgraph。
```

`retrieved_subgraph` 可以复用现有 `graph_memory.rerank.induced_retrieved_subgraph`，因为它是纯 subgraph extraction，不包含手写 score 逻辑。

脚本接入有两个阶段：

第一阶段新增独立脚本：

```text
scripts/run_trainable_retrieval.py
```

第二阶段在稳定后扩展：

```text
scripts/run_retrieval.py --method dense_rgcn_graph_retriever --checkpoint ...
```

这样可以先降低 Phase 1 CLI 被训练参数污染的风险。

## Ablation 内部设计

ablation 不通过在 forward 里堆 `if` 完成，也不主要通过把某个 lambda 置 0 完成。R-GCN 的消融应尽量是结构性替换。

### w/o graph

构造：

```text
graph_encoder = IdentityGraphEncoder
```

行为：

```text
GraphEncoder.forward(...) 直接返回 h0。
```

训练 loop、scorer、loss 不变。

### w/o edge type

构造：

```text
message_transform = SharedRelationTransform
```

行为：

```text
所有 relation_id 共用同一个 W_message。
```

EdgeTensorizer 仍然可以输出 relation_id，但 R-GCN layer 不按 relation_id 选择不同矩阵。

### w/o bridge

构造：

```text
enabled_edge_types = {"sequential", "query_overlap", "entity_overlap"}
```

行为：

```text
EdgeTensorizer 过滤 artifact graph 中 edge_type == "bridge" 的 graph_edge。
模型根本看不到 bridge message_edge。
```

### w/o edge weight

构造：

```text
edge_weight_policy = UniformEdgeWeightPolicy
```

行为：

```text
所有 message_edge 的 edge_weight = 1.0。
```

R-GCN 公式不变。

### w/o seed score

构造：

```text
node_feature_builder = TextOnlyNodeFeatureBuilder
```

行为：

```text
不生成 seed_score 和 seed_rank_percentile。
InputProjection 和 EvidenceNodeScorer 的输入维度在构造阶段变小。
```

### num_layers = 0

构造：

```text
graph_encoder = IdentityGraphEncoder
```

这和 w/o graph 等价，应在 run summary 里记录 canonical ablation name，避免同一实验有两个名字。

## 可观察性

训练阶段必须可回答：

- loss 是否下降。
- dev retrieval metrics 是否提升。
- hard negative 是否仍然大量排在前面。
- graph encoder 是否真的改变了 node score。
- 哪类 edge ablation 影响最大。

推荐输出：

```text
runs/<run_name>/train_metrics.jsonl
runs/<run_name>/train_run_summary.json
runs/<run_name>/debug/negative_sampling_stats.json
runs/<run_name>/debug/dev_score_probe.jsonl
```

### negative_sampling_stats.json

记录：

```text
positive_count
negative_count_by_type
avg_positive_per_task
avg_negative_per_task
tasks_with_no_positive
```

`tasks_with_no_positive` 必须为 0，否则 pair builder 应失败。

### dev_score_probe.jsonl

用于训练后诊断，属于 label-aware debug，因此只能由 training/evaluation 阶段生成，不能由 retrieval-only 推理阶段生成。

每条记录可包含：

```json
{
  "debug_type": "trainable_score_probe",
  "task_id": "hotpot_x",
  "epoch": 3,
  "top_nodes": [
    {
      "node_id": "m7",
      "rank": 1,
      "logit": 4.12,
      "probability": 0.984,
      "label": 1,
      "sample_type": "positive"
    }
  ]
}
```

retrieval-only debug 不应包含 label。

## 测试策略

新增测试文件建议：

```text
tests/test_phase2_rgcn_pairs.py
tests/test_phase2_rgcn_tensorize.py
tests/test_phase2_rgcn_model.py
tests/test_phase2_rgcn_training.py
tests/test_phase2_rgcn_retrieval.py
```

### Pair builder tests

必须测试：

- positive samples 精确来自 `gold_evidence_nodes`。
- negative samples 不包含 gold evidence nodes。
- 每个 pair 的 task/node 都能在 input/graph 中找到。
- `q` 不能出现在 train pair 中。
- hard negative sample type 统计可复现。

### Tensorizer tests

必须测试：

- directed graph edge 只生成 forward message edge。
- undirected graph edge 生成 forward 和 reverse message edge。
- relation vocab 稳定。
- disabled edge type 被过滤。
- uniform edge weight policy 输出全 1。
- relation_id 不等于 artifact edge_type，命名区分清楚。

### Model tests

使用 tiny tensor，不加载真实 sentence-transformer。

必须测试：

- `IdentityGraphEncoder` 输出 shape 不变。
- `RGCNGraphEncoder` 输出 shape 正确。
- typed relation 和 shared relation 都能 forward。
- scorer 输出 `[num_samples]` logits。
- loss 能 backward，至少一个 trainable parameter 有非零梯度。

### Training tests

使用 fake encoder/fake embeddings。

必须测试：

- 一个 tiny batch 可以完成一次 optimizer step。
- checkpoint 能保存和加载。
- best checkpoint 按 dev metric 更新。
- train metrics JSONL 字段完整。

### Retrieval tests

必须测试：

- checkpoint retriever 输出完整 ranked_nodes。
- ranking score 有限且降序。
- retrieved_subgraph nodes 是 top-k。
- trainable retriever 不读取 label artifact。

## 验证策略

新增 validators：

```text
validate_train_pairs(records, inputs_by_task_id, labels_by_task_id, graphs_by_task_id)
validate_negative_sampling_config(config)
validate_train_pair_build_summary(summary)
validate_trainable_model_config(config)
validate_trainable_training_config(config)
validate_trainable_checkpoint_metadata(checkpoint)
```

关键规则：

- `train_pairs` 允许 label，因为它是训练 artifact，不是 retrieval input。
- `train_pairs` 不能被 test-time retrieval 读取。
- pair 中的 `node_id` 必须是 memory node。
- pair 的 `label=1` 必须出现在 gold evidence nodes。
- pair 的 `label=0` 必须不在 gold evidence nodes。
- 推理脚本只读取 tasks、graphs、checkpoint，不读取 labels 或 pairs。

## 代码质量原则

### forward 保持简单

`EvidenceScoringModel.forward` 只做数据流：

```text
features -> input projection -> graph encoder -> scorer -> logits
```

不要在 forward 中处理 artifact schema、采样、validation、checkpoint、metric。

### 构造阶段处理变化

实验差异在构造阶段解决：

```text
GraphEncoderFactory
NodeFeatureBuilder
EdgeTensorizer
MessageTransform
EdgeWeightPolicy
```

forward 不为每个 ablation 写单独分支。

### artifact 与 tensor 分离

JSON artifact 用 `TypedDict` 表达。

训练 batch 用 dataclass 表达，例如：

```text
GraphBatch
TrainingBatch
```

不要把原始 dict 直接传进 torch model。

### 类型注释必须双语完整

Phase 2 新增的具体类型必须使用三引号 docstring 写清楚中英文用途和字段含义。包括：

- artifact `TypedDict`。
- config dataclass。
- internal batch dataclass。
- concrete model/tensorizer/feature builder/sampler/retriever class。
- replaceable behavior `Protocol`。

字段解释不要只写在零散 inline comment 里；contract 文档和类型 docstring 是字段语义的主要维护位置。

### 训练 loop 不知道图算法细节

`training.py` 只知道：

```text
model(batch) -> logits
loss(logits, labels)
metrics from predictions
checkpoint save/load
```

它不应该知道 R-GCN relation 矩阵如何计算。

## Phase 3 扩展点

未来加 GAT 时，应新增：

```text
RelationalGATGraphEncoder
GraphAttentionLayer
AttentionDebugRecord
```

保持不变：

```text
train_pairs.json
TrainingBatch
EvidenceScoringModel
EvidenceNodeScorer
training loop
checkpoint format 的通用字段
RetrievalMethod 输出格式
```

可能新增：

```text
attention_weight_debug.jsonl
attention_entropy metric
head_count / attention_dropout config
```

因此第一版 R-GCN 需要保证 `GraphEncoder` 是可替换组件，而不是把 R-GCN layer 写死在训练 loop 中。

## 实施顺序建议

### Step 1: pair artifact

实现 `build_train_pairs.py` 和 pair validator。

验收：

```text
能从现有 train/dev inputs + labels + graphs 生成 pairs。
pairs 中没有非法 node。
正负样本统计写入 run summary。
```

### Step 2: tensorization

实现 `EdgeTensorizer`、relation vocab、edge weight policy。

验收：

```text
tiny graph 可以稳定转成 edge_index / relation_id / edge_weight。
所有 ablation 结构性过滤都在 tensorization 或组件构造中完成。
```

### Step 3: model forward/backward

实现 fake embedding 下的 R-GCN model。

验收：

```text
tiny batch forward shape 正确。
BCE loss 能 backward。
至少一个 relation weight 有梯度。
```

### Step 4: train loop

实现 `train_graph_retriever.py` 和 checkpoint。

验收：

```text
tiny synthetic train/dev 可以跑完整 epoch。
写出 metrics JSONL、run summary、best checkpoint。
```

### Step 5: retrieval wrapper

实现 `run_trainable_retrieval.py`。

验收：

```text
从 best.pt 输出标准 RankedResult。
现有 evaluate_retrieval.py 能直接评估输出。
```

### Step 6: first real smoke run

用小规模 HotpotQA split 运行：

```text
train 100
dev 50
test 50
```

验收：

```text
训练完成。
dev metrics 可见。
test prediction artifact 可被现有 evaluator 读取。
```

### Step 7: ablation smoke

至少跑：

```text
full rgcn
w/o graph
w/o edge type
w/o bridge
w/o seed score
```

验收：

```text
所有 ablation 使用同一 train/eval code path。
run summary 中记录组件选择。
metrics 可横向比较。
```

## 风险与默认处理

### 文本 embedding 计算慢

默认处理：

```text
先不设计正式 persistent embedding artifact。
训练脚本内部可以做进程内 cache。
如果真实训练成为瓶颈，再把 embedding cache 提升为正式 artifact contract。
```

### R-GCN 过拟合 hard negatives

默认处理：

```text
记录 per-sample-type loss 或 dev probe。
先调 negative ratio 和 dropout。
不立刻 unfreeze encoder。
```

### Graph 太密导致 message passing 慢

默认处理：

```text
沿用 Phase 1 graph build limits。
tensorizer 记录每个 batch 的 num_nodes / num_message_edges。
必要时加入 max_message_edges_per_task，但默认不裁剪。
```

### R-GCN 不如 dense baseline

默认处理：

```text
先比较 w/o graph 和 full rgcn。
如果 w/o graph 接近 dense，但 full rgcn 下降，优先查 edge tensorization 和 normalization。
如果 w/o graph 已明显差于 dense，优先查 seed feature 和 scorer。
```

## 需要后续讨论但已有默认值的事项

这些事项不阻塞第一版实现：

| Topic | Default |
|---|---|
| hidden dim | 256 |
| R-GCN layers | 2 |
| dropout | 0.1 |
| optimizer | AdamW |
| graph/scorer learning rate | 1e-4 |
| max grad norm | 1.0 |
| best checkpoint metric | `0.50 * Full Support@5 + 0.30 * Recall@5 + 0.20 * MRR` |
| encoder fine-tuning | disabled |
| formal embedding cache | deferred until runtime proves it necessary |

## 完成定义

第一版完成时应满足：

- 能生成 `train_pairs.json`。
- 能训练 frozen-encoder R-GCN binary node scorer。
- 能保存和加载 checkpoint。
- 能输出标准 `RankedResult`。
- 能复用现有 evaluator 计算 retrieval metrics。
- 至少有 full R-GCN、w/o graph、w/o edge type、w/o bridge、w/o seed score 五个可运行 ablation。
- 训练和推理阶段的 run summary 足够复现实验配置。
- retrieval-only 推理不读取 labels 或 train pairs。
