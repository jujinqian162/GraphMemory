# Learned Graph R-GCN Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `learned_graph_rgcn_retriever`，让 Ours Retriever 在保持现有 request-first 和 checkpoint-backed R-GCN 体系的前提下，学习候选图边选择、边权和 evidence node 排名。

**Architecture:** 新方法作为独立 public method 接入 registry/workflow，不替换现有 `dense_rgcn_graph_retriever` 或 `dense_ft_rgcn_graph_retriever`。实现上复用现有 dataset projector、`GraphBuildRequest`、R-GCN batching、checkpoint-backed retrieval、workflow stage config 和 evaluation 能力，只新增 method-specific proposal graph、learned edge gate、edge-aware loss 配置和必要的测试覆盖。

**Tech Stack:** Python dataclasses, PyTorch, existing graph_memory registry/config converter/workflow, existing R-GCN graph retriever internals, HotpotQA and 2Wiki dataset projectors, pytest, basedpyright, uv.

---

Date: 2026-07-03

Status: Draft implementation plan.

## 1. 决策摘要

新增 public method：

```text
learned_graph_rgcn_retriever
```

它和现有方法的关系是：

```text
dense_rgcn_graph_retriever
  graph: 当前共享 graph artifact
  train loss: BCE node ranking
  learned: input projection + typed R-GCN + evidence scorer

dense_ft_rgcn_graph_retriever
  graph: 当前共享 graph artifact
  train loss: BCE node ranking
  seed/text encoder: dense_ft checkpoint
  learned: input projection + typed R-GCN + evidence scorer

learned_graph_rgcn_retriever
  graph: method-specific high-recall proposal graph
  train loss: rank + edge + sparse
  learned: input projection + edge gate + typed R-GCN + evidence scorer
```

核心边界：

```text
dataset-specific record
  -> existing dataset projector
  -> TextRankingRequest / GraphBuildRequest / EvidenceEvaluationRequest
  -> proposal graph builder
  -> existing R-GCN batch tensorization + learned edge gate
  -> checkpoint-backed graph retrieval
  -> existing evaluator / result tables
```

不要把这个能力做成 `dense_rgcn_graph_retriever` 的配置开关。它必须是独立 result-table row，方便和当前 Ours、Dense-FT seeded Ours 做公平对比。

Workflow 层面必须按 method 维护独立依赖链。`learned_graph_rgcn_retriever` 的 graph artifact 是自己的 `proposal_graphs`，不依赖也不包裹其他 method 的 shared `graphs` stage。多个 methods 同时选中时，只是在 manifest/planner 中并列组合多条 method DAG。

## 2. 非目标

- 不重写 HotpotQA 或 2Wiki parser/converter/projector。
- 不把 `gold_dependency_edges` 写入 `GraphBuildRequest.input_visible_edges`、graph artifact 或 test-time tensor。
- 不替换现有 R-GCN method 的默认行为。
- 不复制 `TrainableGraphRetrievalMethod`、checkpoint loader、Dense-FT trainer 或 dense retriever。
- 不先做 full joint encoder finetuning。第一版仍使用 frozen text embedding provider。
- 不做大规模超参数搜索。三个 loss weight 可配置，但默认固定为 `1.0 / 0.2 / 0.05`。
- 不新增与现有 request-first 边界冲突的 generic wrapper，例如 `EvidenceRankingView`。

## 3. Loss 设计

训练目标：

```text
L = rank_loss_weight * L_rank
  + edge_loss_weight * L_edge
  + sparse_loss_weight * L_sparse
```

默认配置：

```json
{
  "loss": {
    "rank_loss_weight": 1.0,
    "edge_loss_weight": 0.2,
    "sparse_loss_weight": 0.05
  }
}
```

### 3.1 `L_rank`

`L_rank` 复用当前 R-GCN 的 node-level BCE：

```python
rank_loss = F.binary_cross_entropy_with_logits(
    node_logits,
    node_labels,
    pos_weight=pos_weight,
)
```

它负责 evidence ranking 主任务。权重默认 `1.0`，不要把辅助 loss 设计成能压过主任务。

### 3.2 `L_edge`

`L_edge` 是候选边辅助监督。它只在训练 label 中存在可用 `gold_dependency_edges` 时参与计算：

```python
edge_loss = F.binary_cross_entropy_with_logits(
    edge_gate_logits[labeled_edge_mask],
    edge_labels[labeled_edge_mask],
)
```

HotpotQA 当前没有 path label，`labeled_edge_mask` 为空时 `L_edge = 0.0`，metric record 中记录 `edge_loss_sample_count = 0`。

2Wiki 使用 `EvidenceLabel.gold_dependency_edges` 生成 edge labels，但这些 labels 只进入训练 loss，不进入 proposal graph。第一版 edge label 规则：

- proposal edge 的 endpoints 与某条 gold dependency edge 一致，则该 edge label 为 `1`。
- undirected proposal edge 可以覆盖任一方向的 gold dependency edge。
- 其他有 edge label 的 candidate edge 为 `0`。
- 若某个 task 没有 `gold_dependency_edges`，该 task 的 edges 不进入 edge loss 分母。

默认 `edge_loss_weight = 0.2`，含义是 edge supervision 是辅助信号，不是最终优化目标。

### 3.3 `L_sparse`

`L_sparse` 约束 learned edge gate 不要把 high-recall proposal graph 全部打开：

```python
sparse_loss = edge_gates.mean()
```

第一版先用简单 L1 gate penalty，不引入 target-density 预算，避免把实现复杂度提前扩大。若实验显示 gate 过度收缩，再作为后续改进切换到：

```python
sparse_loss = torch.relu(edge_gates.mean() - sparse_target_density).square()
```

默认 `sparse_loss_weight = 0.05`。这个权重只给开边行为一个成本，不应让模型为了稀疏牺牲明显的 evidence ranking。

## 4. Proposal Graph 设计

现有共享 `build_graph` 是 final graph 风格：规则决定最终可见边，R-GCN 只能在这些边上传播。新方法需要 high-recall proposal graph：规则和相似度只负责产生候选边，learned edge gate 负责学习哪些边真正有效。

不要改变现有共享 graph artifact 的默认语义。实现应新增 method-specific proposal graph artifact，避免影响 `dense_graph_rerank`、`dense_rgcn_graph_retriever` 和现有报告可复现性。

目标 call flow 是 method-local DAG：

```text
prepare
  |
proposal_graphs: learned_graph_rgcn_retriever
  |
pairs: learned_graph_rgcn_retriever
  |
train: learned_graph_rgcn_retriever
  |
retrieve: learned_graph_rgcn_retriever
  |
evaluate / aggregate
```

如果同一次实验还选择了现有 graph methods，它们拥有自己的依赖链：

```text
prepare
  |
graphs: dense_rgcn_graph_retriever
  |
pairs/train/retrieve: dense_rgcn_graph_retriever
```

这两条链只共享 `prepare` 产物和 dataset projector，不共享 graph artifact 语义。实现时不要把 `proposal_graphs` 写成 `graphs` 的下游 stage，也不要让新 method 等待其他 method 的 graph stage。

Proposal graph 第一版复用现有 `GraphBuilder` 规则，只提高召回：

- `sequential`：完全复用当前 rule。
- `query_overlap`：提高 `max_query_overlap`。
- `entity_overlap`：提高 `max_entity_neighbors`。
- `bridge`：提高 `max_bridge_edges`。

如果这四类边无法覆盖足够 candidate edges，再新增 `semantic_candidate` edge type 作为第二步。`semantic_candidate` 必须只使用 input-visible query/candidate text 的 frozen embeddings，不得使用 answer/supporting facts/evidences。

第一版建议先不新增 edge type，先验证“当前 edge vocabulary + high-recall caps + learned gate”是否带来收益。这样可以最大化复用当前 tensorizer、relation vocab、ablation 和 evaluation 能力。

## 5. 文件责任图

### Registry / Config

| File | Responsibility |
|---|---|
| `graph_memory/registry/retrieval.py` | 增加 `RetrievalMethodId.LEARNED_GRAPH_RGCN_RETRIEVER`，让 checkpoint graph retrieval settings 接受新方法。 |
| `graph_memory/registry/methods.py` | 注册新方法为 `RGCN_TRAINABLE`、graph-backed、checkpoint-backed、path-metric-capable，`seed_method=RetrievalMethodId.DENSE`。 |
| `graph_memory/registry/method_configs.py` | 增加 learned graph R-GCN 的 method config dataclass，复用 R-GCN encoder/pair/train settings，并加入 loss/proposal graph settings。 |
| `configs/methods/learned_graph_rgcn_retriever.json` | 新方法默认配置，包含 loss weights 默认 `1.0/0.2/0.05`。 |

### Graph / Proposal

| File | Responsibility |
|---|---|
| `graph_memory/graphs/config.py` | 增加 `ProposalGraphBuildConfig` 或 method-specific proposal config，不能改变 `GraphBuildConfig` 默认行为。 |
| `graph_memory/graphs/construction/builder.py` | 复用 `GraphBuilder` 和现有 edge rules；不要复制四套 rule。 |
| `scripts/build_proposal_graphs.py` | 新增低层 CLI，读取 dataset input records，按 dataset selector 构造 `GraphBuildRequest`，输出 method-specific proposal graphs。 |
| `scripts/workflow/manifest.py` | 在 `learned_graph_rgcn_retriever` 的 method dependency chain 中增加 proposal graph artifact path。 |
| `scripts/workflow/stage_configs.py` | 为新方法写 proposal graph stage config；不要把它挂到其他 method 的 graph stage 后面。 |
| `scripts/workflow/planner.py` | 新增 method-local proposal graph stage ordering：`prepare` 之后、该 method 的 `pairs/train` 之前。 |

### Model / Training

| File | Responsibility |
|---|---|
| `graph_memory/models/graph_retriever/config/records.py` | 增加 `RgcnLossConfig`，并保存到 checkpoint metadata。 |
| `graph_memory/models/graph_retriever/internals/tensorization.py` | 在不破坏现有 `EdgeTensorizer` 的前提下，为 learned method 增加 edge feature/gate tensor 输出。 |
| `graph_memory/models/graph_retriever/internals/neural.py` | 增加 `EdgeGateScorer` 和 gated R-GCN path，复用 `TypedRelationTransform`、`RelationalGraphConvLayer` 思路和 `EvidenceNodeScorer`。 |
| `graph_memory/models/graph_retriever/batching.py` | 为 learned method 构造 edge labels、edge gate features 和 sparse loss 所需 mask。现有 R-GCN batching 行为不变。 |
| `graph_memory/models/graph_retriever/training.py` | 在 `train_graph_retriever` 或新训练入口中计算 rank/edge/sparse loss，记录各项 loss 和 sample count。 |
| `graph_memory/models/graph_retriever/factory.py` | 根据 `graph_encoder_type` 或 method config 构造 gated model；现有 `rgcn`/`identity` 路径不变。 |
| `graph_memory/models/graph_retriever/checkpoint.py` | checkpoint 保存/加载 loss config 和新 method identity。 |

### Retrieval / Evaluation

| File | Responsibility |
|---|---|
| `graph_memory/retrieval/methods/trainable_graph.py` | 继续复用 checkpoint-backed adapter；只让 expected method 接受新 id。 |
| `graph_memory/models/graph_retriever/inference.py` | 加载 gated model，并用 proposal graph 产生 retrieved subgraph。 |
| `graph_memory/registry/retrieval_builders.py` | 继续复用 `_build_checkpoint_graph()`，不要新写一个 retrieval builder。 |
| `graph_memory/evaluation/path_metrics.py` | 复用现有 path metric 语义；新方法因为 graph-backed 自动支持 path metrics。 |

### Dataset / Experiment Configs

| File | Responsibility |
|---|---|
| `configs/experiments/hotpotqa_evidence_retrieval.json` | 作为唯一 HotpotQA canonical config，包含新 method、method config mapping、服务器和 dev-heldout profiles。 |
| `configs/experiments/2wiki_evidence_retrieval.json` | 添加新 method 和 method config mapping。 |
| `configs/experiments/2wiki_tiny.json` | 添加新 method 到 smoke/tiny 验证路径。 |
| `README.md` / `docs/40-operations/commands.md` | 更新公开 method 列表和运行命令。 |

## 6. 配置形状

`configs/methods/learned_graph_rgcn_retriever.json` 第一版建议：

```json
{
  "method": "learned_graph_rgcn_retriever",
  "default_profile": "quick",
  "encoder": {
    "batch_size": 64,
    "model_name": "models/intfloat-e5-base-v2",
    "passage_prefix": "passage: ",
    "query_prefix": "query: "
  },
  "proposal_graph": {
    "max_query_overlap": 80,
    "max_entity_neighbors": 30,
    "max_bridge_edges": 200,
    "use_spacy": false
  },
  "pairs": {
    "easy_random_per_positive": 2,
    "hard_bm25_per_positive": 2,
    "hard_dense_per_positive": 0,
    "hard_graph_neighbor_per_positive": 1,
    "hard_pool_size": 30,
    "random_seed": 13
  },
  "train": {
    "loss": {
      "rank_loss_weight": 1.0,
      "edge_loss_weight": 0.2,
      "sparse_loss_weight": 0.05
    },
    "model": {
      "ablation": "full_rgcn",
      "dropout": 0.1,
      "hidden_dim": 128,
      "num_layers": 2
    },
    "trainer": {
      "batch_size": 8,
      "device": "cuda",
      "epochs": 5,
      "learning_rate": 0.0001,
      "max_grad_norm": 1.0,
      "optimizer_name": "AdamW",
      "pos_weight_enabled": true,
      "random_seed": 13
    },
    "reporting": {
      "render_training_curves": true
    },
    "selection": {
      "best_metric": "dev_composite",
      "higher_is_better": true
    }
  }
}
```

配置落点：

- `rank_loss_weight`、`edge_loss_weight`、`sparse_loss_weight` 必须在 method config 中可配置。
- `RgcnLossConfig` 默认值必须是 `1.0`、`0.2`、`0.05`。
- `train_metrics.jsonl` 或 checkpoint metadata 中记录实际 loss weights，保证报告可追溯。
- 不要把 loss weights 放到 experiment-level CLI option；method config 是稳定入口。

## 7. HotpotQA 适配

HotpotQA 适配原则：复用现有 HotpotQA dataset package 和 request projectors。

```text
HotpotQARankingRecord
  -> HotpotQAToTextRankingRequest
  -> HotpotQAToGraphBuildRequest
  -> proposal graph
  -> learned_graph_rgcn_retriever
```

需要做的事：

1. 在 HotpotQA experiment configs 中添加 `learned_graph_rgcn_retriever`。
2. workflow 在该 method 的独立依赖链中生成 proposal graph artifact。
3. train stage 使用现有 `build_train_pairs.py` 输出的 node-level train pairs。
4. `L_edge` 因为没有 labeled `gold_dependency_edges` 自动为 `0.0`。
5. `L_sparse` 正常启用，约束 proposal graph 不被全部打开。
6. evaluation 继续报告 node-level metrics；HotpotQA 的 `Path Recall@10` / `Edge Recall@10` 保持当前语义，通常为 `N/A`。

HotpotQA 不允许做的事：

- 不从 `supporting_facts` 构造 test-time edge。
- 不把 gold evidence hints 放进 proposal graph。
- 不改变现有 `dense_rgcn_graph_retriever` 在 HotpotQA 上的 graph artifact。

验收：

- `scripts/experiment.py plan ... --config configs/experiments/hotpotqa_evidence_retrieval.json --methods learned_graph_rgcn_retriever` 能生成 proposal graphs、pairs、train、retrieve、evaluate、aggregate。
- `learned_graph_rgcn_retriever` 预测文件包含 `ranked_nodes` 和 graph-aware `retrieved_subgraph`。
- `edge_loss_sample_count = 0`，但训练不失败。
- 现有 HotpotQA 方法矩阵结果不因新 method 发生行为变化。

## 8. 2WikiMultiHopQA 适配

2Wiki 适配原则：复用已经实现的 2Wiki request-first dataset package。

```text
TwoWikiRankingRecord / TwoWiki label record
  -> TwoWikiToTextRankingRequest
  -> TwoWikiToGraphBuildRequest
  -> TwoWikiToEvidenceEvaluationRequest
  -> proposal graph
  -> learned_graph_rgcn_retriever
```

需要做的事：

1. 在 `configs/experiments/2wiki_evidence_retrieval.json` 和 `configs/experiments/2wiki_tiny.json` 中添加新方法和 method config mapping。
2. proposal graph 只使用 `question` 和 candidate sentence text/title/source metadata。
3. `gold_dependency_edges` 只从 `EvidenceLabel` 进入 training batch 的 edge labels。
4. `L_edge` 仅在 proposal edge endpoints 与 gold dependency edge 可对齐时计算。
5. path metrics 继续从 `prediction.retrieved_subgraph` 和 label-side `gold_dependency_edges` 计算。

2Wiki 不允许做的事：

- 不把 `evidences` / `evidences_id` / `supporting_facts` / `answer` 写入 `GraphBuildRequest`。
- 不用 gold dependency edge 直接补 proposal graph。
- 不为提高 path metric 分数改变 evaluator 语义。

验收：

- 2Wiki tiny workflow 能跑通 `learned_graph_rgcn_retriever`。
- `edge_loss_sample_count > 0` 至少在含 path labels 的 train/dev split 上成立。
- `Path Recall@10` / `Edge Recall@10` 对新方法输出 numeric value 或在没有 path-supported tasks 时输出 `N/A`。
- Leakage tests 证明 gold fields 不进入 proposal graph artifact。

## 9. 实施任务

### Task 1: 注册 public method 和 config 类型

**Files:**
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/registry/method_configs.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Test: `tests/test_method_registry.py`
- Test: `tests/test_registry_stage_configs.py`

- [ ] 增加 `RetrievalMethodId.LEARNED_GRAPH_RGCN_RETRIEVER = "learned_graph_rgcn_retriever"`。
- [ ] 新增或扩展 `LearnedGraphRgcnMethodConfig`，包含 `encoder`、`proposal_graph`、`pairs`、`train.loss`、`train.model`、`train.trainer`、`train.reporting`、`train.selection`。
- [ ] 增加 `RgcnLossSettings` / `RgcnLossConfig`，默认：

```python
@dataclass(frozen=True)
class RgcnLossSettings:
    rank_loss_weight: float = 1.0
    edge_loss_weight: float = 0.2
    sparse_loss_weight: float = 0.05
```

- [ ] 注册 method definition：

```python
MethodDefinition(
    identifier=RetrievalMethodId.LEARNED_GRAPH_RGCN_RETRIEVER,
    lifecycle=RetrievalLifecycle.RGCN_TRAINABLE,
    retrieval_settings_type=CheckpointGraphRetrievalSettings,
    dependencies=RetrievalDependencySpec(
        graphs=GraphInputSource.GRAPH_ARTIFACT,
        selected_config=SelectedConfigSource.NONE,
        model=ModelSource.CHECKPOINT_FILE,
        encoder=EncoderSource.CHECKPOINT_METADATA,
    ),
    method_config_type=LearnedGraphRgcnMethodConfig,
    train_artifact=TrainArtifactSpec("best.pt", ArtifactKind.FILE),
    seed_method=RetrievalMethodId.DENSE,
)
```

- [ ] Registry test 断言新 method 出现在 `Registry.methods.list_ids()`。
- [ ] Registry test 断言 `supports_path_metrics("learned_graph_rgcn_retriever") is True`。
- [ ] Config validation test 断言缺少 `train.loss.rank_loss_weight` 时 fail fast。

### Task 2: 新增 method-specific proposal graph stage

**Files:**
- Modify: `graph_memory/graphs/config.py`
- Modify: `graph_memory/graphs/construction/builder.py`
- Create: `scripts/build_proposal_graphs.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `scripts/workflow/planner.py`
- Test: `tests/test_proposal_graphs.py`
- Test: `tests/test_experiment_runner.py`

- [ ] 增加 proposal graph config dataclass，字段与 `GraphBuildConfig` 对齐但默认 caps 更高。
- [ ] `scripts/build_proposal_graphs.py` 复用 dataset selectors：

```text
--dataset hotpotqa|twowiki
--input <input records>
--output <proposal graphs>
--config <stage config>
```

- [ ] CLI 内部仍调用 dataset-owned `graph_build_requests_for_dataset()`，不要 import 具体 parser。
- [ ] CLI 内部用 `GraphBuilder(ProposalGraphBuildConfig(...))` 或现有 `GraphBuilder` + proposal config 构造 graphs。
- [ ] Manifest 为新方法写入 method-specific proposal graph artifact，例如：

```text
runs/<run>/graphs/train.learned_graph_rgcn_retriever.graphs.json
runs/<run>/graphs/dev.learned_graph_rgcn_retriever.graphs.json
runs/<run>/graphs/test.learned_graph_rgcn_retriever.graphs.json
```

- [ ] Planner 确保 proposal graph stage 在 `prepare` 之后、该 method 的 `pairs/train` 之前。
- [ ] Tests 断言只选择 `learned_graph_rgcn_retriever` 时不要求先生成其他 method 的 shared graph artifact。
- [ ] Tests 断言同时选择 `dense_rgcn_graph_retriever` 和 `learned_graph_rgcn_retriever` 时，两条 method graph chain 并列存在且各自读取自己的 graph artifact。
- [ ] Tests 断言新方法读取 proposal graph artifact。

### Task 3: 扩展 batch tensor，加入 edge features 和 edge labels

**Files:**
- Modify: `graph_memory/models/graph_retriever/internals/contracts.py`
- Modify: `graph_memory/models/graph_retriever/internals/tensorization.py`
- Modify: `graph_memory/models/graph_retriever/batching.py`
- Test: `tests/test_graph_retriever_batching.py`
- Test: `tests/test_learned_graph_edge_labels.py`

- [ ] 在不破坏现有 `GraphBatch` 消费者的前提下新增 learned graph edge tensors。优先新增独立 dataclass：

```python
@dataclass(frozen=True)
class LearnedEdgeBatch:
    edge_features: Tensor
    edge_label_mask: Tensor
    edge_labels: Tensor
```

- [ ] `TrainingBatch` 增加 `learned_edges: LearnedEdgeBatch | None = None`。
- [ ] 现有 R-GCN 方法构造 batch 时保持 `learned_edges=None`。
- [ ] learned method 构造 edge features，第一版包含：
  - artifact edge weight
  - relation id
  - source seed score / rank percentile
  - target seed score / rank percentile
  - source is question
  - target is question
- [ ] 从 `EvidenceLabel.gold_dependency_edges` 构造 edge labels；没有 labels 时 mask 全 false。
- [ ] Tests 覆盖 HotpotQA labels 空时 `edge_label_mask.sum() == 0`。
- [ ] Tests 覆盖 2Wiki gold edge 匹配 proposal edge endpoints 时 label 为 `1`。

### Task 4: 新增 learned edge gate 和 gated R-GCN

**Files:**
- Modify: `graph_memory/models/graph_retriever/internals/neural.py`
- Modify: `graph_memory/models/graph_retriever/factory.py`
- Modify: `graph_memory/models/graph_retriever/config/defaults.py`
- Test: `tests/test_learned_graph_model.py`

- [ ] 新增 `EdgeGateScorer`：

```python
class EdgeGateScorer(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int, dropout: float) -> None: ...
    def forward(self, edge_features: Tensor) -> Tensor: ...
```

- [ ] 新增 gated model output：

```python
@dataclass(frozen=True)
class EvidenceScoringOutput:
    node_logits: Tensor
    edge_gate_logits: Tensor | None = None
    edge_gates: Tensor | None = None
```

- [ ] 现有 `EvidenceScoringModel` 可以继续返回 `Tensor`，或统一迁移为 output object；若统一迁移，必须同步 dev/inference/tests。
- [ ] 新增 gated R-GCN path，把 `sigmoid(edge_gate_logits)` 乘进 message edge weights：

```python
gated_edge_weights = batch.graph_batch.edge_weights * edge_gates
```

- [ ] 复用 `TypedRelationTransform` 和 `EvidenceNodeScorer`；不要复制 evidence scorer MLP。
- [ ] Factory 根据 config 构造 `learned_rgcn` graph encoder/model。
- [ ] Tests 断言 gate 维度与 message edges 对齐。
- [ ] Tests 断言 `edge_gates=0` 时 message passing 不使用 proposal edges。

### Task 5: 实现 rank/edge/sparse loss

**Files:**
- Modify: `graph_memory/models/graph_retriever/config/records.py`
- Modify: `graph_memory/models/graph_retriever/training.py`
- Modify: `graph_memory/models/graph_retriever/dev_evaluation.py`
- Test: `tests/test_learned_graph_training_loss.py`
- Test: `tests/test_phase2_rgcn_training.py`

- [ ] `RgcnTrainingConfig` 或 learned method training config 保存 `loss_config`。
- [ ] 计算：

```python
total_loss = (
    loss_config.rank_loss_weight * rank_loss
    + loss_config.edge_loss_weight * edge_loss
    + loss_config.sparse_loss_weight * sparse_loss
)
```

- [ ] `rank_loss` 复用现有 BCE。
- [ ] `edge_loss` 在 `edge_label_mask.any()` 为 false 时返回同 device 的 `0.0` tensor。
- [ ] `sparse_loss` 在 `edge_gates is None` 时返回 `0.0`。
- [ ] Metric records 增加：
  - `train_rank_loss`
  - `train_edge_loss`
  - `train_sparse_loss`
  - `edge_loss_sample_count`
  - `mean_edge_gate`
  - `rank_loss_weight`
  - `edge_loss_weight`
  - `sparse_loss_weight`
- [ ] Tests 断言默认 weights 为 `1.0/0.2/0.05`。
- [ ] Tests 断言设置 `edge_loss_weight=0.0` 时 edge loss 不影响 total loss。

### Task 6: Checkpoint 和 inference 复用

**Files:**
- Modify: `graph_memory/models/graph_retriever/checkpoint.py`
- Modify: `graph_memory/models/graph_retriever/inference.py`
- Modify: `graph_memory/retrieval/methods/trainable_graph.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Test: `tests/test_phase2_rgcn_retrieval.py`
- Test: `tests/test_retrieval_provenance.py`

- [ ] Checkpoint 保存 `method_name = "learned_graph_rgcn_retriever"`。
- [ ] Checkpoint 保存 model/loss config，包含 loss weights。
- [ ] Inference 继续走 `CheckpointGraphRetrieverLoader`。
- [ ] `_build_checkpoint_graph()` 根据 `settings.method.value` 传入 `expected_method`。
- [ ] 新方法 provenance 中 `method` 为 `learned_graph_rgcn_retriever`。
- [ ] Prediction 的 `retrieved_subgraph` 基于 proposal graph 的 top-k induced subgraph。
- [ ] Tests 断言现有 `dense_rgcn_graph_retriever` checkpoint 不被新 method loader 接受，反之亦然。

### Task 7: Method config 和 HotpotQA workflow wiring

**Files:**
- Create: `configs/methods/learned_graph_rgcn_retriever.json`
- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Do not maintain a separate HotpotQA dev-full config; keep dev-heldout profiles in `configs/experiments/hotpotqa_evidence_retrieval.json`.
- Modify: `tests/test_cli_contracts.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `README.md`
- Modify: `docs/40-operations/commands.md`

- [ ] 创建 method config，loss weights 默认 `1.0/0.2/0.05`。
- [ ] HotpotQA active configs 增加 method 和 method config mapping。
- [ ] Plan/render tests 断言 HotpotQA 新方法 stage 顺序包含 proposal graphs。
- [ ] Train payload tests 断言 HotpotQA `edge_loss_sample_count = 0` 不导致训练失败。
- [ ] Docs 更新 method 列表，不把新方法描述为替代旧 R-GCN。

### Task 8: 2Wiki workflow wiring 和 leakage tests

**Files:**
- Modify: `configs/experiments/2wiki_evidence_retrieval.json`
- Modify: `configs/experiments/2wiki_tiny.json`
- Modify: `tests/test_twowiki_workflow.py`
- Modify: `tests/test_twowiki_leakage_boundaries.py`
- Create or modify: `tests/test_learned_graph_twowiki_edge_loss.py`

- [ ] 2Wiki active configs 增加 method 和 method config mapping。
- [ ] Workflow plan tests 断言 tiny config 可选择 `learned_graph_rgcn_retriever`。
- [ ] Leakage tests 断言 proposal graph artifact 不包含 `gold_dependency_edges`、`supporting_facts`、`evidences`、`evidences_id`、`answer`。
- [ ] Edge loss tests 用 tiny fixture 断言至少一条 proposal edge 可被 gold dependency edge 标为 positive。
- [ ] Evaluation tests 断言 path metrics 使用 `retrieved_subgraph`，不是 edge gate labels。

### Task 9: 验证

**Files:**
- No new production files unless failures identify missing coverage.

- [ ] Run targeted tests outside the Windows sandbox:

```powershell
uv run pytest `
  tests/test_method_registry.py `
  tests/test_registry_stage_configs.py `
  tests/test_proposal_graphs.py `
  tests/test_graph_retriever_batching.py `
  tests/test_learned_graph_model.py `
  tests/test_learned_graph_training_loss.py `
  tests/test_phase2_rgcn_retrieval.py `
  tests/test_twowiki_leakage_boundaries.py `
  tests/test_twowiki_workflow.py `
  -q
```

- [ ] Run type check outside the Windows sandbox:

```powershell
uv run basedpyright --level error
```

- [ ] Run full tests before publishing:

```powershell
uv run pytest -q
```

- [ ] Run HotpotQA plan smoke:

```powershell
uv run python scripts/experiment.py plan learned_graph_hotpotqa_smoke `
  --config configs/experiments/hotpotqa_evidence_retrieval.json `
  --profile smoke `
  --methods learned_graph_rgcn_retriever `
  --force
```

- [ ] Run 2Wiki plan smoke:

```powershell
uv run python scripts/experiment.py plan learned_graph_2wiki_smoke `
  --config configs/experiments/2wiki_tiny.json `
  --profile smoke `
  --methods learned_graph_rgcn_retriever `
  --force
```

- [ ] On CUDA-capable environment, run one tiny 2Wiki workflow:

```powershell
uv run python scripts/experiment.py run learned_graph_2wiki_smoke `
  --config configs/experiments/2wiki_tiny.json `
  --profile smoke `
  --methods learned_graph_rgcn_retriever `
  --force
```

## 10. 接受标准

- `scripts/experiment.py methods list` 显示 `learned_graph_rgcn_retriever`。
- 新方法有独立 method config，loss weights 默认 `1.0`、`0.2`、`0.05` 且可配置。
- 选择新方法时 workflow 生成 method-specific proposal graph artifact。
- 现有 `dense_rgcn_graph_retriever`、`dense_ft_rgcn_graph_retriever` 行为不变。
- HotpotQA 能训练新方法；没有 edge labels 时 edge loss 自动为 0。
- 2Wiki 能训练新方法；有 `gold_dependency_edges` 的样本可产生 edge loss。
- `gold_dependency_edges` 只用于 train/eval label，不进入 test-time graph/proposal graph。
- 新方法复用 checkpoint-backed graph retriever path，不引入第二套 retrieval adapter。
- 新方法 path metrics 由 registry capability 判定，并从 `retrieved_subgraph` 计算。
- 训练 metrics 能拆分 rank/edge/sparse 三项 loss，便于报告和 ablation。

## 11. 风险和控制

### Workflow graph artifact 可能被错误建模为全局依赖

风险：实现时如果把 `proposal_graphs` 写成全局 `graphs` 的下游 stage，会错误地让新 method 依赖其他 method 的 graph chain，也会让只跑新 method 的 workflow 多出无关 stage。

控制：第一版把 proposal graph 建模为 `learned_graph_rgcn_retriever` 自己的 method-local dependency。旧方法继续读取自己的 graph artifact；新方法不关心旧方法是否被选中。

### Edge gate 改动可能冲击现有 R-GCN

风险：统一修改 `EvidenceScoringModel.forward()` 返回值会牵动 training/dev/inference。

控制：优先新增 learned model path，保留旧 model 返回 `Tensor` 的行为；如果统一返回 output object，必须同一 patch 覆盖所有现有 R-GCN tests。

### `L_edge` 在 HotpotQA 上没有监督

风险：HotpotQA 训练时 edge loss 恒为 0，容易误解为功能没工作。

控制：metric record 明确记录 `edge_loss_sample_count = 0`；HotpotQA 主要验证 rank + sparse + proposal graph，不验证 edge supervision。

### Proposal graph 过密或过稀

风险：caps 太大导致训练慢和噪声多；caps 太小又回到漏边问题。

控制：不要先做大网格。先用固定 high-recall caps，并报告 avg edges、mean edge gate、retrieved subgraph edge count。后续只做 A/B/C ablation：

```text
A: rank only
B: rank + sparse
C: rank + edge + sparse
```

### 2Wiki edge label 对齐可能有歧义

风险：gold dependency edge 是 label-side dependency，不一定总能直接匹配 proposal edge endpoints。

控制：第一版只对可直接 endpoint match 的 proposal edges 加 edge labels；无法匹配的 path labels 仍用于 evaluator，不强行补 graph。

## 12. 后续 ablation 建议

完成第一版后再考虑：

```text
learned_graph_rgcn_retriever_rank_only
learned_graph_rgcn_retriever_rank_sparse
learned_graph_rgcn_retriever_full
```

这三个 ablation 用于回答结构问题：

- BCE rank loss 是否已经足够？
- sparse gate penalty 是否提升泛化和解释性？
- 2Wiki edge supervision 是否带来 path metric 提升？

不要在第一版就搜索 `rank_loss_weight`、`edge_loss_weight`、`sparse_loss_weight`。默认值是固定设计选择，不是调参结果。
