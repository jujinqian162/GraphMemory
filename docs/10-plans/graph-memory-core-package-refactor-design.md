# Graph Memory 核心包分层重构设计

日期：2026-06-02

状态：设计审阅中。本文当前覆盖已确认的总体方案、逐模块职责迁移设计，以及第三轮审阅所需的实施批次、行为等价标准和验证矩阵。本文不是实施记录，不代表代码已经迁移。

## 1. 背景

项目最初有意采用 `library-core + thin CLI` 架构，并在 Phase 1 保持 `graph_memory/` 基本扁平，以免过早引入复杂目录层次。这个决策在早期是合理的：当时核心目标是先跑通 HotpotQA、BM25、Dense、Graph Rerank 和统一评估链路。

当前情况已经变化：

- `graph_memory/types.py` 同时承载磁盘 artifact、评估表格、rerank 配置、dense 配置、训练模型配置和 tensor batch。
- `graph_memory/retrieval.py` 同时承载方法构造、方法分派、缓存路径、批量运行、结果组装和若干低层构造细节。
- `graph_memory/validation.py` 已经成为覆盖所有 artifact 与训练 tensor 的大型验证文件。
- `graph_memory/learned/` 已经形成独立子系统，但其中仍然存在训练、推理、模型构造、batching 和特征职责交错。
- 后续需求不再只是增加一个算法，而是会增加 Dense-FT、Memory Stream、GraphRAG、MemGPT-style、GAT、2WikiMultiHopQA 和工具轨迹等不同语义族。

因此，本轮重构不是推翻早期设计，而是执行早期架构文档中已经预留的提取规则：当模块出现多个独立实现或已经难以导航时，将其拆分为有明确依赖方向的子包。

## 2. 重构目标

本轮重构只改变内部组织，不改变实验行为。

### 2.1 必须保持不变的外部行为

- `scripts/experiment.py` 的 CLI 用法保持不变。
- `scripts/workflow/` 本轮不修改。
- workflow 调用的底层 CLI 保持原有参数名、默认值和行为：
  - `scripts/prepare_hotpotqa.py`
  - `scripts/build_graphs.py`
  - `scripts/run_retrieval.py`
  - `scripts/tune_graph_rerank.py`
  - `scripts/build_train_pairs.py`
  - `scripts/train_graph_retriever.py`
  - `scripts/run_trainable_retrieval.py`
  - `scripts/evaluate_retrieval.py`
  - `scripts/aggregate_tables.py`
- 磁盘 artifact 的 JSON、JSONL、CSV 和 checkpoint schema 保持不变。
- 公开 retrieval method 名称保持不变：
  - `bm25`
  - `dense`
  - `bm25_graph_rerank`
  - `dense_graph_rerank`
  - `dense_rgcn_graph_retriever`
- 排序规则、打分公式、top-k 语义、retrieved subgraph 语义、配置覆盖顺序、随机种子语义、模型结构和训练目标保持不变。
- 在相同输入、相同配置、相同依赖环境和相同随机种子下，训练结果与推理结果保持一致。

### 2.2 允许改变的内部细节

- `graph_memory.*` 内部 Python 导入路径。
- 文件位置和子包结构。
- 内部类名、函数名和组合方式。
- 内部 dataclass、Protocol 和请求对象。
- scripts 内部对 `graph_memory.*` 的 import。
- tests 内部对 `graph_memory.*` 的 import。

### 2.3 非目标

本轮不实现：

- 新 baseline 或新 dataset。
- 新图构建规则。
- 新 artifact schema。
- 动态插件发现。
- 通用依赖注入容器。
- 通用 pipeline engine。
- workflow 内部重构。
- 性能优化或跨 task batching 行为变化。
- dense embedding cache 正式 artifact。
- 为历史内部 import 路径保留全量兼容层。

## 3. 核心设计原则

### 3.1 Single Level of Abstraction Principle

同一个高层函数或方法中，只允许出现同一级别的概念。

例如，高层 retrieval 用例可以表达：

```text
解析运行请求
构造 retrieval method
逐 task 执行 ranking
组装结果 artifact
验证结果
```

它不应直接出现：

```text
query_prefix
passage_prefix
SentenceTransformer.encode
neighbor_type_weights
edge_index
relation_ids
torch.device
```

这些概念属于更低层的运行时配置、具体算法或 tensor 实现。

### 3.2 用对象隔离状态与策略，用函数表达纯变换

适合使用对象的情况：

- 对象拥有明确生命周期，例如 encoder、模型、checkpoint-backed retriever。
- 多个步骤共享同一组运行时状态，例如 `DenseRuntime`。
- 行为可替换，例如 graph edge rule、score component、graph encoder。
- 需要通过构造阶段完成策略组合，例如 R-GCN ablation。

适合保持函数式的情况：

- JSON、CSV、JSONL IO。
- artifact validator。
- 文本 tokenization。
- 数值归一化。
- metric primitive。
- 小型、确定性的转换。

本轮不追求“所有逻辑都变成 class”。目标是消除无边界的参数传递和职责混杂，而不是把纯函数机械包装成对象。

### 3.3 目录结构表达职责，依赖方向表达层次

目录深度不能单独表达整个系统的抽象层次，因为真实依赖是有向图，不是树。

采用以下规则：

- 根目录下放并列的领域包和公共叶子包。
- `application/` 是明确的最高层用例编排。
- 每个领域包内部，越深越接近实现细节。
- 同目录文件应处于相近抽象层次。
- 最终以依赖方向约束为准，不能只追求目录视觉整齐。

### 3.4 只抽象真实共同点

底层算法即使“看起来相似”，只要语义不同，就不强行合并。

例如：

- `SequentialEdgeRule`
- `QueryOverlapEdgeRule`
- `EntityOverlapEdgeRule`
- `BridgeEdgeRule`

它们可以有一致命名、一致目录位置和一致调用接口，但不应为了减少几行代码而合并成充满条件分支的通用 edge rule。

同理：

- flat dense retrieval
- handwritten graph rerank
- trainable graph retrieval
- 未来 Memory Stream
- 未来 GraphRAG

只共享顶层 `RetrievalMethod` 输出契约。它们的内部语义不应被强迫进入同一个父类层次或同一个万能 Context。

### 3.5 Context 只能表示内聚的单次操作输入

Context 本身不是反模式。问题在于 Context 是否表示一个边界清晰、字段内聚的操作。

允许：

```text
PreparedGraphInput
  当前 task
  已计算的 IDF
  已提取的 entity map
```

因为这些字段共同服务于一次 graph construction，且没有大量互斥字段。

禁止：

```text
RetrievalBuildContext
  graphs?
  graph_config?
  dense_encoder?
  checkpoint_path?
  text_embedding_provider?
  seed_signal_provider?
  device?
```

因为这些字段分别属于互斥的方法族。新增方法时继续追加字段会使 Context 变成整个系统的参数仓库。

不要通过 Context 继承解决问题。应通过组合对象和按方法族区分的请求对象解决。

## 4. 方案比较

### 4.1 方案 A：仅移动文件

做法：

- 将大型文件拆成多个文件。
- 保留当前函数式构造链。
- 保留大型 Context 和长参数传递。

优点：

- 改动风险最低。
- 初始迁移速度快。

缺点：

- 只能缓解视觉拥挤。
- `query_prefix` 等低层参数仍然穿透高层函数。
- 新增方法时仍然会扩大共同 Context。
- 不能真正建立稳定的职责边界。

结论：不采用。

### 4.2 方案 B：领域包 + application service + 组合对象

做法：

- 按领域拆分子包。
- 用 `application/` 表达高层用例。
- 用组合对象隔离运行时状态。
- 用明确的方法族构造请求替代万能 Context。
- 在有状态、可替换的边界使用对象。
- 纯算法、validator 和 IO 保持函数式。

优点：

- 解决当前阅读困难和参数隧道。
- 不需要动态框架。
- 可以逐批迁移，每批独立验证。
- 适合后续新增语义差异较大的 baseline。

缺点：

- 需要同步更新 scripts 和 tests 的内部 import。
- 迁移必须严格分批，否则难以定位行为偏差。

结论：采用。

### 4.3 方案 C：全面框架化

做法：

- 动态插件发现。
- 依赖注入容器。
- 大型抽象基类层次。
- 通用 pipeline engine。

优点：

- 理论扩展性高。

缺点：

- 当前方法数量、团队规模和实验性质不支持这种复杂度。
- 运行路径更隐式，反而降低可审计性。
- 容易把未知的未来变化错误固化为框架。

结论：不采用。

## 5. 目标目录结构

```text
graph_memory/
  application/
    build_graphs.py
    run_retrieval.py
    tune_graph_rerank.py
    build_train_pairs.py
    train_graph_retriever.py
    evaluate_retrieval.py

  contracts/
    common.py
    tasks.py
    graphs.py
    ranking.py
    training_pairs.py
    metrics.py
    observability.py

  validation/
    common.py
    tasks.py
    graphs.py
    ranking.py
    training_pairs.py
    metrics.py
    model.py

  datasets/
    splits.py
    hotpotqa/
      records.py
      parser.py
      converter.py
      compatibility.py

  text/
    tokens.py
    lexical.py
    entities.py

  graphs/
    config.py
    index.py
    statistics.py
    views.py
    construction/
      builder.py
      context.py
      edge_accumulator.py
      rules/
        contracts.py
        sequential.py
        query_overlap.py
        entity_overlap.py
        bridge.py

  retrieval/
    contracts.py
    catalog.py
    requests.py
    resolver.py
    factory.py
    execution/
      service.py
      results.py
    signals.py
    methods/
      flat/
        method.py
        bm25.py
        dense.py
      graph_rerank/
        method.py
        config.py
        engine.py
        components.py
        candidates.py
        normalization.py
        debug.py
      trainable_graph.py
    tuning/
      grid.py
      initial_scores.py
      service.py

  training_pairs/
    config.py
    builder.py
    samplers.py

  models/
    graph_retriever/
      contracts.py
      text_embeddings.py
      config/
        records.py
        defaults.py
        loading.py
      factory.py
      inference.py
      training.py
      dev_evaluation.py
      checkpoint.py
      internals/
        contracts.py
        batching.py
        features.py
        tensorization.py
        neural/
          model.py
          encoders.py
          transforms.py
          layers.py
          scorer.py

  evaluation/
    metrics.py
    connectivity.py
    service.py
    tables.py
    failure_cases.py

  infrastructure/
    io.py
    runtime_environment.py
    run_summary.py

  experiment.py
  io.py
  observability.py
  retrieval_registry.py
  training_config.py
```

最后五个根目录模块不是长期理想结构，而是本轮刻意保留的 workflow integration ports。原因见第 7 节。

## 6. 依赖方向

高层依赖低层，低层不反向感知高层。

```text
scripts/*.py
  -> graph_memory.application
  -> graph_memory.infrastructure

graph_memory.application
  -> datasets
  -> graphs
  -> retrieval
  -> training_pairs
  -> models.graph_retriever
  -> evaluation
  -> validation

datasets / graphs / retrieval / training_pairs / models / evaluation
  -> contracts

graphs
  -> text

retrieval.methods.graph_rerank
  -> graphs.views

models.graph_retriever
  -> graphs.views

all packages
  -> infrastructure only when the dependency is genuinely technical
```

### 6.1 禁止依赖

- `contracts/` 不依赖算法实现。
- `validation/` 不依赖 `application/`。
- `graphs/` 不依赖 retrieval、training_pairs、models 或 evaluation。
- `retrieval/` 不依赖 `application/`。
- `models/graph_retriever/` 不依赖 CLI。
- `infrastructure/` 不知道研究算法。
- 核心算法不直接读取或写入 JSON、CSV、JSONL artifact。
- 核心算法不直接解析 CLI 参数。

### 6.2 关于 `application/`

`application/` 不是另一个脚本目录。它表达“已经脱离 CLI，但仍然负责组织一个完整用例”的高层服务。

例如：

```text
scripts/run_retrieval.py
  解析 argparse
  读取文件
  构造 RunRetrievalRequest
  调用 application.run_retrieval
  写文件与 run summary
```

```text
application/run_retrieval.py
  验证请求
  解析具体 method build request
  构造 method
  调用 retrieval execution service
  返回 ranked result records
```

这可以阻止低层配置细节重新泄漏到 CLI 和顶层编排中。

## 7. Workflow 兼容边界

本轮不修改 `scripts/workflow/`。当前 workflow 直接 import 以下根目录模块：

- `graph_memory.io`
- `graph_memory.observability`
- `graph_memory.retrieval_registry`
- `graph_memory.training_config`
- `graph_memory.experiment`

因此，本轮允许保留五个窄兼容入口：

```text
graph_memory/io.py
graph_memory/observability.py
graph_memory/retrieval_registry.py
graph_memory/training_config.py
graph_memory/experiment.py
```

规则：

- 这些文件只能做明确 re-export 或极薄适配。
- 只能暴露 workflow 当前实际消费的名称。
- 不允许把新的核心逻辑继续添加到这些文件。
- 不为 tests 或历史内部 import 额外扩大兼容面。
- 后续单独重构 workflow 时，再决定是否删除这些 integration ports。

这与“不要保留旧结构 facade”不冲突。它们不是为历史内部路径兜底，而是为了满足本轮明确冻结的 workflow 边界。

## 8. RetrievalBuildContext 的替代设计

当前 `RetrievalBuildContext` 同时容纳：

```text
method
task_inputs
graphs
encoder_model
query_prefix
passage_prefix
graph_config
dense_encoder
checkpoint_path
text_embedding_provider
seed_signal_provider
device
```

问题：

- 大量字段只对部分方法有效。
- 字段之间存在隐含约束。
- 新增方法时倾向于继续追加字段。
- 高层函数被迫了解低层参数。
- `query_prefix` 等 dense 内部细节穿透多个抽象层级。

### 8.1 组合对象

```python
@dataclass(frozen=True)
class DenseConfig:
    model_name: str
    query_prefix: str
    passage_prefix: str
    batch_size: int


@dataclass(frozen=True)
class DenseRuntime:
    config: DenseConfig
    encoder: SentenceEncoder | None = None


@dataclass(frozen=True)
class GraphIndex:
    graph_by_task_id: dict[TaskId, MemoryGraph]


@dataclass(frozen=True)
class TrainableGraphRuntime:
    checkpoint_path: Path
    device: str
    text_embedding_provider: TextEmbeddingProvider | None = None
    seed_signal_provider: SeedSignalProvider | None = None
    dense_runtime: DenseRuntime | None = None
```

`GraphIndex` 放在 `graphs/index.py`。它只表示已验证 graph artifact 的按 task 查询视图，不负责文件 IO，也不负责构图。

`TextEmbeddingProvider` 与 `SentenceEncoder` 放在 `models/graph_retriever/contracts.py`。`TrainableGraphRuntime` 可以依赖这个公开模型边界，但不能 import `models/graph_retriever/internals/*`。

### 8.2 按方法族区分构造请求

```python
@dataclass(frozen=True)
class FlatMethodBuildRequest:
    method: MethodName
    seed_retriever: SeedRetrieverBuildRequest


@dataclass(frozen=True)
class GraphRerankMethodBuildRequest:
    method: MethodName
    seed_retriever: SeedRetrieverBuildRequest
    graphs: GraphIndex
    config: GraphRerankConfig


@dataclass(frozen=True)
class TrainableGraphMethodBuildRequest:
    method: MethodName
    graphs: GraphIndex
    runtime: TrainableGraphRuntime
```

顶层 request 可以携带 CLI 层允许的可选输入，但可选性只能停留在解析边界：

```text
CLI args
  -> RunRetrievalRequest
  -> RetrievalRequestResolver
  -> 精确的方法族 BuildRequest
  -> RetrievalMethodFactory
```

一旦进入具体 method factory，不再允许出现“整个系统所有可能字段”的 Context。

### 8.3 不采用 Context 继承

不设计：

```text
BaseContext
  -> DenseContext
  -> GraphContext
  -> TrainableContext
```

原因：

- 继承会把“可复用字段”误表达为“is-a”关系。
- 方法族需要的是不同能力组合，不是单一继承轴。
- 多重继承或层层 subclass 会使构造路径更难理解。

采用组合：

```text
GraphRerankMethodBuildRequest
  has SeedRetrieverBuildRequest
  has GraphIndex
  has GraphRerankConfig
```

## 9. 第二轮审阅：逐模块迁移设计

本节明确每个现有模块的职责归宿，以及哪些逻辑需要对象化、哪些逻辑保持函数式。

### 9.1 `graph_memory/types.py`

现状问题：

- 该文件同时包含 artifact schema、算法结果、配置和 tensor batch。
- 导入方众多，任何新增类型都会继续扩大中心文件。
- 阅读某个算法时，需要在大型文件中跨层级寻找相关结构。

拆分规则：

| 当前内容 | 目标位置 | 处理方式 |
|---|---|---|
| `TaskId`, `NodeId`, `MethodName`, `Score`, `Json*` | `contracts/common.py` | 保持简单 alias |
| `MemoryItem`, `MemoryTaskInput`, `MemoryTaskLabels`, `CombinedMemoryTask` | `contracts/tasks.py` | 保持 JSON-shaped `TypedDict` |
| `QuestionNode`, `GraphMemoryNode`, `GraphEdge`, `MemoryGraph` | `contracts/graphs.py` | 保持 JSON-shaped `TypedDict` |
| `RankedNodeRecord`, `RetrievedSubgraph`, `RankedResult` | `contracts/ranking.py` | artifact 类型保持 `TypedDict` |
| `TrainPairRecord`, `TrainPairBuildSummary` | `contracts/training_pairs.py` | artifact 类型保持 `TypedDict` |
| `MetricRow`, `TaskMetricRow`, `MetricTableRow`, `FailureCase` | `contracts/metrics.py` | 保持 CSV/debug shaped contract |
| `GraphStatistics`, `RunSummary`, debug record | `contracts/observability.py` | 保持 artifact contract |
| `RankedNode` | `retrieval/contracts.py` | 内部算法结果 dataclass |
| `DenseConfig` | `retrieval/methods/flat/dense.py` | 跟随 dense 算法 |
| `GraphBuildConfig` | `graphs/config.py` | 跟随图构建领域 |
| `GraphRerankConfig`, `RerankResult`, score breakdown | `retrieval/methods/graph_rerank/config.py` 与 `engine.py` | 跟随 rerank 领域 |
| `NegativeSamplingConfig` | `training_pairs/config.py` | 跟随 pair generation |
| `NodeFeatureConfig`, `TrainableModelConfig`, `TrainableTrainingConfig` | `models/graph_retriever/config/records.py` | 跟随 trainable model |
| `SeedSignal` | `retrieval/signals.py` | seed retrieval 公共语义 |
| `GraphBatch`, `TrainingBatch` | `models/graph_retriever/internals/contracts.py` | 限定在 tensor 子系统 |
| `Retriever` Protocol | `retrieval/contracts.py` | 顶层 retrieval 领域契约 |

结论：

- `graph_memory/types.py` 最终删除，但不能在 Batch 1 末尾删除。
- 迁移期间 `types.py` 只能作为逐步缩小的临时迁移文件，服务尚未迁移到目标领域包的内部类型。
- 新代码不得新增 `from graph_memory.types`。
- 不保留聚合式 re-export。

原因：

- `NodeFeatureConfig`、`TrainableModelConfig`、`GraphBatch` 和 `TrainingBatch` 的最终位置在 `models/graph_retriever/`。
- `NegativeSamplingConfig` 的最终位置在 `training_pairs/`。
- `GraphBuildConfig` 的最终位置在 `graphs/`。
- 如果 Batch 1 强行删除 `types.py`，就会被迫提前创建多个后续领域包，或引入宽泛临时 re-export。这两种做法都会破坏分批迁移的边界。
- 调用方直接 import 自己实际依赖的领域类型。

### 9.2 `graph_memory/validation.py`

现状问题：

- 文件较大，但问题主要是物理组织，不是函数式风格。
- artifact validator、config validator 和 tensor validator 同处一个文件。

目标结构：

```text
validation/
  common.py
  tasks.py
  graphs.py
  ranking.py
  training_pairs.py
  metrics.py
  model.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| `ContractValidationError`, record narrowing、公共 field helper、task alignment | `validation/common.py` |
| input、label、label leakage validation | `validation/tasks.py` |
| graph node、edge、graph artifact validation | `validation/graphs.py` |
| ranked result、retrieved subgraph validation | `validation/ranking.py` |
| pair、pair summary、negative sampling config validation | `validation/training_pairs.py` |
| metric row validation | `validation/metrics.py` |
| model config、training config、checkpoint metadata、GraphBatch、TrainingBatch validation | `validation/model.py` |

风格规则：

- validator 继续保持纯函数。
- validator 不修复、不排序、不推断、不补默认值。
- `common.py` 可以保留少量文件级 helper，因为它是明确的底层叶子工具模块。
- 领域 validator 只调用 `common.py` 和本领域 contract，不反向 import 算法实现。

### 9.3 `graph_memory/io.py` 与 `graph_memory/observability.py`

现状问题：

- IO 本身没有设计问题。
- observability 混合了运行时环境、run summary、graph statistics 和 rerank debug。

目标结构：

```text
infrastructure/
  io.py
  runtime_environment.py
  run_summary.py

graphs/
  statistics.py

retrieval/methods/graph_rerank/
  debug.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| JSON、CSV、JSONL、config read/write/merge | `infrastructure/io.py` |
| `collect_environment()` | `infrastructure/runtime_environment.py` |
| `now_iso()`, `build_run_summary()`, `write_run_summary()` | `infrastructure/run_summary.py` |
| `graph_statistics()` | `graphs/statistics.py` |
| rerank score debug record、rerank config digest | `retrieval/methods/graph_rerank/debug.py` |

由于 workflow 本轮不修改：

- 根目录 `io.py` 暂时 re-export workflow 所需 IO。
- 根目录 `observability.py` 暂时 re-export workflow 所需时间与 summary 能力。

### 9.4 `graph_memory/hotpotqa.py` 与 `graph_memory/splits.py`

现状问题：

- `hotpotqa.py` 同时有 raw record dataclass、解析、转换和 compatibility artifact 组装。
- 这些职责属于同一 dataset，但抽象层次不同。

目标结构：

```text
datasets/
  splits.py
  hotpotqa/
    records.py
    parser.py
    converter.py
    compatibility.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| `sample_split()` | `datasets/splits.py` |
| `HotpotQADocument`, `HotpotQASupportingFact`, `HotpotQAExample` | `datasets/hotpotqa/records.py` |
| raw JSON shape narrowing 与 parse helper | `datasets/hotpotqa/parser.py` |
| `ConvertedHotpotQAExample`, `HotpotQAConversionResult`, conversion functions | `datasets/hotpotqa/converter.py` |
| `combined_memory_tasks()` | `datasets/hotpotqa/compatibility.py` |

风格规则：

- parser helper 可以保持纯函数。
- converter 保持确定性函数，不引入 dataset plugin registry。
- 后续增加 2Wiki 或 tool trajectory 时，新增并列 dataset 子包，不扩张 HotpotQA 文件。

未来示意：

```text
datasets/
  hotpotqa/
  twowiki/
  tool_trajectory/
```

### 9.5 `graph_memory/text.py` 与 `graph_memory/entities.py`

目标结构：

```text
text/
  tokens.py
  lexical.py
  entities.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| `content_tokens()` 与短 token 处理 | `text/tokens.py` |
| `compute_idf()`, `lexical_score()` | `text/lexical.py` |
| title alias、heuristic entities、spaCy entity extraction | `text/entities.py` |

风格规则：

- 全部保持纯函数。
- 不引入 tokenizer class 或 entity extractor class，除非未来真的出现需要组合的多种 extractor。
- `entities.py` 可以依赖 `tokens.py`。
- `lexical.py` 可以依赖 `tokens.py`。
- `tokens.py` 不反向依赖其他 text 模块。

### 9.6 `graph_memory/graphs.py`

现状问题：

- 顶层 `build_graph()` 同时准备共享状态、调用规则、维护 edge 去重和组装 artifact。
- 四种 edge 构造逻辑是并列算法，却被塞在同一文件的私有函数中。
- 新增 tool dependency、parameter flow 或 GraphRAG-style 构图时，继续扩张同一文件会更难阅读。

目标结构：

```text
graphs/
  config.py
  statistics.py
  views.py
  construction/
    builder.py
    context.py
    edge_accumulator.py
    rules/
      contracts.py
      sequential.py
      query_overlap.py
      entity_overlap.py
      bridge.py
```

核心对象：

```python
class GraphEdgeRule(Protocol):
    def add_edges(
        self,
        graph_input: PreparedGraphInput,
        accumulator: EdgeAccumulator,
    ) -> None:
        ...


@dataclass(frozen=True)
class GraphBuilder:
    config: GraphBuildConfig
    rules: tuple[GraphEdgeRule, ...]

    def build(self, task_input: MemoryTaskInput) -> MemoryGraph:
        ...

    def build_many(self, task_inputs: list[MemoryTaskInput]) -> list[MemoryGraph]:
        ...
```

```python
@dataclass
class EdgeAccumulator:
    edges: list[GraphEdge]
    seen_edge_keys: set[tuple[str, str, str]]

    def add(...): ...
```

```python
@dataclass(frozen=True)
class PreparedGraphInput:
    task_input: MemoryTaskInput
    documents: list[str]
    idf: dict[str, float]
    entities_by_node_id: dict[NodeId, set[str]]
```

`PreparedGraphInput` 是允许存在的 Context，因为：

- 只服务单次 graph construction。
- 字段全部内聚。
- 没有互斥字段。
- edge rule 只看到自己所需的同层预计算状态。

规则类并列存在：

```text
SequentialEdgeRule
QueryOverlapEdgeRule
EntityOverlapEdgeRule
BridgeEdgeRule
```

规则：

- 保持现有执行顺序，避免 edge 输出顺序改变。
- 保持现有排序和截断逻辑。
- 不强行抽象规则内部算法。
- `EdgeAccumulator` 统一承担 edge key、去重和 append。
- `views.py` 放置纯 graph view 操作，例如 induced subgraph 和 adjacency 构造。

未来新增：

```text
ToolDependencyEdgeRule
ParameterFlowEdgeRule
```

它们应作为并列 rule 出现，而不是往已有 rule 中添加 dataset 分支。

### 9.7 `graph_memory/retrieval_registry.py`

现状判断：

- 静态 registry 的存在是合理的。
- registry 解决公开 method 名称和 capability 元数据的单一来源问题。
- 问题不在 registry，而在 runtime builder 目前集中于 `retrieval.py`。

目标结构：

```text
retrieval/
  catalog.py
  resolver.py
  factory.py
```

职责：

| 模块 | 职责 |
|---|---|
| `catalog.py` | `RetrievalMethodSpec` 和静态 method catalog；只描述公开方法与 capability |
| `resolver.py` | 根据公开 method 和 application request 解析出精确的 method family build request |
| `factory.py` | 根据精确 build request 构造具体 `RetrievalMethod` |

规则：

- `catalog.py` 不 import 具体 method class。
- catalog 仍然是静态表，不做动态插件发现。
- catalog 不保存大型运行时对象。
- capability query 继续从 catalog 元数据派生。
- workflow 继续通过根目录 `retrieval_registry.py` integration port 访问已有 query。

未来新增方法时：

```text
新增 method 实现
新增 build request 或复用现有 request
在 factory 增加构造分支
在 catalog 增加一条元数据
补 requirement validation tests
```

不要把 callable builder 直接塞入 catalog。这样可以保持元数据层不依赖重型实现，避免 import torch 或 sentence-transformers。

### 9.8 `graph_memory/retrieval.py`

现状问题：

- 该文件承担多个抽象层次。
- `RetrievalBuildContext` 是可选字段宇宙。
- `_build_seed_retriever()` 暴露 dense 低层细节。
- graph rerank cache path 与普通 retrieval path 纠缠。
- 结果组装与 token 估计混在 method 构造文件。

目标拆分：

```text
retrieval/
  contracts.py
  requests.py
  resolver.py
  factory.py
  execution/
    service.py
    results.py
  signals.py
  methods/
    flat/
      method.py
      bm25.py
      dense.py
    graph_rerank/
      method.py
    trainable_graph.py
  tuning/
    initial_scores.py
```

迁移规则：

| 当前内容 | 目标位置 | 处理方式 |
|---|---|---|
| `DenseEncoder` Protocol | `retrieval/methods/flat/dense.py` | 跟随 dense 实现 |
| `Retriever` Protocol | `retrieval/contracts.py` | 顶层 flat retrieval contract |
| `RetrievalMethod` Protocol | `retrieval/contracts.py` | 顶层 public method contract |
| `RetrievalBuildContext` | 删除 | 替换为 typed request + resolver |
| `InitialScoreCache` | `retrieval/tuning/initial_scores.py` | 跟随 tuning cache |
| `ScorePipelineMethod` | `retrieval/methods/flat/method.py` | 只包装 flat retriever |
| `GraphRerankMethod` | `retrieval/methods/graph_rerank/method.py` | 只组织 seed ranking 与 rerank engine |
| `PrecomputedInitialRetriever` | 删除或收进 `initial_scores.py` | tuning 私有实现，不进入普通 runtime |
| method builder functions | `retrieval/factory.py` | 改成按 request 类型分派 |
| `build_retrieval_method()` | `retrieval/factory.py` | 接受精确 build request |
| `precompute_initial_score_cache()` | `retrieval/tuning/initial_scores.py` | 改为 `InitialScorePrecomputer` |
| cached rerank execution | `retrieval/tuning/service.py` | tuning 专属路径 |
| `run_retrieval()` | `application/run_retrieval.py` + `retrieval/execution/service.py` | 分离用例编排与逐 task 执行 |
| `assemble_ranked_result()` | `retrieval/execution/results.py` | 保持纯函数 |
| token 估计 | `retrieval/execution/results.py` | 命名为公开叶子 helper |
| `_build_seed_retriever()` | `retrieval/factory.py` 或 flat 子包 factory | 不再接收散装 dense 参数 |

目标高层代码应接近：

```python
def run_retrieval(request: RunRetrievalRequest) -> list[RankedResult]:
    build_request = request_resolver.resolve(request)
    method = method_factory.build(build_request)
    return retrieval_service.run(method=method, tasks=request.task_inputs, top_k=request.top_k)
```

这里不会出现 `query_prefix`、checkpoint load、graph rerank component 或 tensor 细节。

### 9.9 `graph_memory/indexes/`

现状问题：

- BM25 与 Dense 本质上是 flat retrieval 实现，不是独立于 retrieval 的通用 index 子系统。
- `indexes/` 命名容易误导后续开发者把任何 retrieval method 都塞进这里。

目标结构：

```text
retrieval/methods/flat/
  method.py
  bm25.py
  dense.py
```

规则：

- `BM25TaskRetriever` 和 `DenseTaskRetriever` 保持并列。
- BM25 继续保持简单 class，因为它实现 `Retriever.rank()`。
- Dense 使用 `DenseRuntime` 聚合 encoder 配置和 encoder 实例。
- 不在本轮改变 encoder 调用次数、batch 行为或分数计算。

### 9.10 `graph_memory/rerank.py` 与 `graph_memory/rerank_config.py`

现状问题：

- score component、候选扩展、归一化、组合、subgraph extraction 和兼容 helper 混在同一文件。
- 文件级私有函数承载核心算法步骤，导致入口看似简单，但真实流程需要跨文件内部跳转阅读。

目标结构：

```text
retrieval/methods/graph_rerank/
  method.py
  config.py
  engine.py
  components.py
  candidates.py
  normalization.py
  debug.py

graphs/
  views.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| `GraphRerankConfig` 与 config parse/ensure | `graph_rerank/config.py` |
| `GraphRerankMethod` | `graph_rerank/method.py` |
| `ScoreContext`, component Protocol 和四个 score component | `graph_rerank/components.py` |
| `rank_graph_from_initial_scores()` | `GraphRerankEngine.rank()` in `engine.py` |
| component score combination | `ScoreCombiner.combine()` in `engine.py` |
| candidate expansion | `candidates.py` 纯函数 |
| normalization | `normalization.py` 纯函数 |
| induced subgraph | `graphs/views.py` 纯函数 |
| traversal adjacency | `graphs/views.py` 纯函数 |
| compatibility helper | 仅在确有 tests 或 scripts 使用时放入明确的 `compatibility.py`；否则删除 |

核心对象：

```python
@dataclass(frozen=True)
class GraphRerankEngine:
    config: GraphRerankConfig
    components: tuple[NodeScoreComponent, ...]
    combiner: ScoreCombiner

    def rank(
        self,
        initial_scores: dict[NodeId, float],
        graph: MemoryGraph,
        *,
        top_k: int,
        include_score_breakdown: bool = False,
    ) -> RerankResult:
        ...
```

规则：

- score component 保持并列类。
- 不把不同 score 公式合并为通用条件函数。
- component 顺序保持现状。
- 归一化和组合公式保持现状。
- `lambda_path` 保持既有语义，不在本轮实现新 path score。

### 9.11 `graph_memory/tuning.py`

现状问题：

- grid parse、objective、best selection、initial score cache 和 dev evaluation 编排处于同一模块。
- tuning 高层函数暴露 dense 细节参数。

目标结构：

```text
retrieval/tuning/
  grid.py
  initial_scores.py
  service.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| grid 默认值、grid record parse | `grid.py` |
| initial score cache 与预计算 | `initial_scores.py` |
| objective 与 best config selection | `service.py` |
| dev tuning loop | `GraphRerankTuner.tune()` in `service.py` |

核心对象：

```python
@dataclass(frozen=True)
class GraphRerankTuner:
    evaluator: Evaluator
    score_precomputer: InitialScorePrecomputer

    def tune(self, request: GraphRerankTuningRequest) -> GraphRerankConfig:
        ...
```

规则：

- `query_prefix` 只存在于 `DenseRuntime.config`。
- tuner 只接收 seed runtime 或 score precomputer，不接收散装 encoder 参数。
- initial score cache 继续是本次 tuning invocation 内的显式缓存，不提升为磁盘 artifact。

### 9.12 `graph_memory/evaluation.py`

现状问题：

- metric primitive、图连通辅助、聚合、failure case 和 table split 混在同一文件。
- 图连通计算中的 adjacency 与 reachability 是低层图视图逻辑。

目标结构：

```text
evaluation/
  metrics.py
  connectivity.py
  service.py
  tables.py
  failure_cases.py
```

迁移规则：

| 当前逻辑 | 目标位置 |
|---|---|
| Recall、F1、Full Support、MRR | `metrics.py` |
| connected evidence、query evidence connectivity | `connectivity.py` |
| prediction/label/graph join 与 aggregate evaluation | `service.py` |
| table column split 与列选择 | `tables.py` |
| failure case build | `failure_cases.py` |

使用对象：

```python
@dataclass(frozen=True)
class GraphConnectivity:
    undirected_adjacency: dict[NodeId, set[NodeId]]
    directed_adjacency: dict[NodeId, set[NodeId]]

    @classmethod
    def from_graph(cls, graph: MemoryGraph, allowed_nodes: set[NodeId]) -> "GraphConnectivity":
        ...

    def reachable_from(self, start: NodeId) -> set[NodeId]:
        ...
```

说明：

- metric primitive 继续保持纯函数。
- adjacency 是共享的派生状态，使用小对象封装比在多个 metric helper 中重复构造更清晰。
- `evaluation/service.py` 保留纯函数入口 `evaluate_results()`。当前 evaluator 没有需要持有的生命周期状态，不额外引入 `Evaluator` class。

### 9.13 `graph_memory/learned/data.py`

现状问题：

- train pair generation 不属于 learned model 内部。
- pair artifact 未来也可能服务 Dense-FT。
- 文件包含多个并列 sampling 算法，适合按策略拆分，但不适合强行统一内部语义。

目标结构：

```text
training_pairs/
  config.py
  builder.py
  samplers.py
```

核心对象：

```python
class NegativeSampler(Protocol):
    sample_type: TrainPairSampleType

    def sample(self, context: PairSamplingContext, desired_count: int) -> list[NodeId]:
        ...


@dataclass(frozen=True)
class TrainPairBuilder:
    config: NegativeSamplingConfig
    samplers: tuple[NegativeSampler, ...]

    def build(
        self,
        task_inputs: list[MemoryTaskInput],
        labels: list[MemoryTaskLabels],
        graphs: list[MemoryGraph],
    ) -> TrainPairBuildResult:
        ...
```

并列 sampler：

```text
EasyRandomNegativeSampler
BM25HardNegativeSampler
DenseHardNegativeSampler
GraphNeighborNegativeSampler
```

规则：

- sampler 命名、目录和接口一致。
- sampler 内部算法保持各自独立。
- 保持现有 sampler 执行顺序、随机数生成方式、去重 key 和截断规则。
- 不在本轮修改 negative ratio。
- `PairSamplingContext` 只描述单个 task 的内聚采样状态，不包含互斥方法族字段。

### 9.14 `graph_memory/learned/features.py`

该文件当前混合两类职责：

- retrieval-level seed signal。
- trainable model tensor feature。

拆分：

```text
retrieval/
  signals.py

models/graph_retriever/
  contracts.py
  text_embeddings.py

models/graph_retriever/internals/
  features.py
```

迁移规则：

| 当前内容 | 目标位置 |
|---|---|
| `SeedSignal`, `SeedSignalProvider`, `RetrieverSeedSignalProvider` | `retrieval/signals.py` |
| `TextEmbeddingProvider`, `SentenceEncoder` | `models/graph_retriever/contracts.py` |
| `DenseTextEmbeddingProvider` | `models/graph_retriever/text_embeddings.py` |
| `NodeFeatureTensors`, `NodeFeatureBuilder` | `models/graph_retriever/internals/features.py` |

理由：

- `TextEmbeddingProvider` 是 trainable graph retriever 的公开运行时注入边界，会被 `TrainableGraphRuntime`、loader、training 和 tests 共同引用。
- 它不能放在 `internals/features.py`，否则 retrieval request 或 factory 会反向依赖模型内部实现。
- `internals/features.py` 只保留模型内部数值特征构造，例如 `NodeFeatureBuilder`。

关于 dense encoding：

- 当前 flat dense retrieval 和 trainable graph retrieval 都使用 sentence-transformer，但调用方式并不完全相同。
- 长期适合提取共享的 `DenseEncodingService`，作为 frozen embedding consumer 的低层公共能力。
- 本轮首先移动职责，不立即统一调用路径。
- 后续如提取共享 encoder service，必须单独做行为保持验证，避免 batch 方式变化引入浮点差异或性能行为变化。

### 9.15 `graph_memory/learned/tensorize.py`

目标位置：

```text
models/graph_retriever/internals/tensorization.py
```

保留并列策略对象：

```text
ArtifactEdgeWeightPolicy
UniformEdgeWeightPolicy
EdgeTensorizer
```

规则：

- tensorization 属于模型内部低层实现。
- 保持 relation vocab、forward/reverse edge 生成顺序、过滤规则和 dtype。
- `model_visible_graph()` 迁入 `graphs/views.py`。它表达按 enabled edge type 过滤 graph artifact 的通用视图，不属于 tensorization 细节。

### 9.16 `graph_memory/learned/batching.py`

目标位置：

```text
models/graph_retriever/internals/batching.py
```

建议对象：

```python
@dataclass(frozen=True)
class GraphRetrieverBatchBuilder:
    model_config: TrainableModelConfig
    text_embedding_provider: TextEmbeddingProvider
    seed_signal_provider: SeedSignalProvider
    edge_tensorizer: EdgeTensorizer
    feature_builder: NodeFeatureBuilder

    def build_training_batches(...): ...
    def build_full_ranking_batches(...): ...
```

原因：

- 当前多个函数层层传递相同的 model config、embedding provider、seed provider。
- 这些是同一个 batching 生命周期中的稳定依赖。
- 构造一次 builder 后，具体 batch 方法只需要业务输入。

保留函数：

- `move_training_batch()` 可以继续是纯函数。

规则：

- batch 拼接顺序、node offset、sample 顺序、dtype 和 device move 行为保持不变。

### 9.17 `graph_memory/learned/model.py`

现状问题：

- graph encoder、message transform、layer、scorer 和顶层 model 都在同一文件。
- 它们虽然都属于神经网络，但层级不同。

目标结构：

```text
models/graph_retriever/internals/neural/
  model.py
  encoders.py
  transforms.py
  layers.py
  scorer.py
```

迁移规则：

| 当前内容 | 目标位置 |
|---|---|
| `GraphEncoder`, `IdentityGraphEncoder`, `RGCNGraphEncoder` | `encoders.py` |
| `MessageTransform`, `TypedRelationTransform`, `SharedRelationTransform` | `transforms.py` |
| `RelationalGraphConvLayer` | `layers.py` |
| `EvidenceNodeScorer` | `scorer.py` |
| `EvidenceScoringModel` | `model.py` |
| relation degree normalization | `layers.py` 叶子 helper |

规则：

- `EvidenceScoringModel.forward()` 只表达同级数据流。
- forward 不知道 checkpoint、artifact、validation、ablation、采样或 metrics。
- ablation 继续通过 construction-time component replacement 表达。
- 未来 GAT 作为与 R-GCN 并列的 encoder 实现加入 `encoders.py` 或独立子包，不修改 training loop。

### 9.18 `graph_memory/learned/training.py`

现状问题：

- 默认 config、ablation 解释、model factory、training loop、dev prediction、metric selection 和 state dict copy 混在同一文件。
- `inference.py` 反向 import `training.py` 中的 `build_model_from_config()`，说明 model construction 放错层。

目标结构：

```text
models/graph_retriever/
  config/
    records.py
    defaults.py
    loading.py
  factory.py
  training.py
  dev_evaluation.py
```

迁移规则：

| 当前内容 | 目标位置 |
|---|---|
| `default_model_config()` 与 ablation mapping | `config/defaults.py` |
| `build_model_from_config()` | `factory.py` |
| `TrainableTrainingResult`, training loop | `training.py` |
| `_predict_dev()` | `dev_evaluation.py` |
| best metric selection | `dev_evaluation.py` 或 `training.py` 的窄 helper |
| state dict CPU copy | `checkpoint.py` 或 `training.py` 的叶子 helper |

建议对象：

```python
@dataclass(frozen=True)
class GraphScoringModelFactory:
    def build(self, config: TrainableModelConfig) -> EvidenceScoringModel:
        ...


@dataclass(frozen=True)
class GraphRetrieverTrainer:
    model_factory: GraphScoringModelFactory
    batch_builder: GraphRetrieverBatchBuilder
    dev_evaluator: GraphRetrieverDevEvaluator

    def train(self, request: TrainGraphRetrieverRequest) -> TrainableTrainingResult:
        ...
```

说明：

- trainer 高层只表达训练生命周期。
- optimizer、scheduler、loss、clip grad、checkpoint selection 保持现有语义。
- factory 负责 ablation 对应的组件组装。
- inference 与 training 都依赖 factory；inference 不再依赖 training。

### 9.19 `graph_memory/learned/checkpoint.py`

目标位置：

```text
models/graph_retriever/checkpoint.py
```

规则：

- checkpoint 属于模型运行时状态，可以保留 PyTorch IO。
- checkpoint schema 完全不变。
- metadata validation 进入 `validation/model.py`。
- checkpoint loader 不知道 CLI。

### 9.20 `graph_memory/learned/inference.py`

目标结构：

```text
models/graph_retriever/
  inference.py

retrieval/methods/
  trainable_graph.py
```

拆分职责：

| 模块 | 职责 |
|---|---|
| `models/graph_retriever/inference.py` | checkpoint 加载、模型恢复、full-ranking model inference |
| `retrieval/methods/trainable_graph.py` | 将模型推理适配为统一 `RetrievalMethod.rank_task()` |

建议对象：

```python
@dataclass(frozen=True)
class CheckpointGraphRetrieverLoader:
    model_factory: GraphScoringModelFactory

    def load(self, runtime: TrainableGraphRuntime) -> GraphRetrieverInference:
        ...


@dataclass(frozen=True)
class TrainableGraphRetrievalMethod:
    name: MethodName
    graph_index: GraphIndex
    inference: GraphRetrieverInference

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...
```

这样 retrieval 层只知道统一 method contract，模型层只知道 tensor inference，不再互相穿透职责。

### 9.21 `graph_memory/training_config.py`

目标位置：

```text
models/graph_retriever/config/
  records.py
  defaults.py
  loading.py
```

迁移规则：

| 当前内容 | 目标位置 |
|---|---|
| `EncoderConfig`, `ModelConfigValues` | `config/records.py` |
| train config read、profile resolve、section parse | `config/loading.py` |
| 默认值和 ablation 组件映射 | `config/defaults.py` |

由于 workflow 本轮不修改：

- 根目录 `training_config.py` 暂时 re-export workflow 使用的 loader 和 device helper。

### 9.22 `graph_memory/experiment.py`

保持现状：

- 它仍然是 `scripts.workflow` 的兼容 facade。
- 本轮不扩大也不缩减它的 API。
- workflow 专项重构时再处理。

## 10. 文件级私有函数规则

目标不是机械消灭所有 `_helper()`，而是消灭“文件级私有函数承载跨层级核心流程”的情况。

允许保留文件级 helper：

- JSON shape narrowing。
- validator 底层字段检查。
- 简短数值 helper。
- 与单个算法文件高度内聚、不会形成跨模块调用的叶子操作。

应改为命名对象或公开叶子函数：

- 构造策略。
- 多步骤算法阶段。
- 被多个调用路径复用的逻辑。
- 拥有共享状态的逻辑。
- 需要单独测试和替换的逻辑。

判断标准：

```text
如果 helper 需要携带多个参数穿过多层调用，
或者它代表一个可以独立命名的算法阶段，
就不应继续藏在大型模块的文件级私有函数中。
```

## 11. 未来需求的结构落点

来自项目原始计划的未来需求，不在本轮实现，但目标结构需要给出自然落点。

| 未来需求 | 自然落点 | 不应放入 |
|---|---|---|
| Dense-FT | `retrieval/methods/flat/dense_ft.py` 与独立训练包 | 现有 frozen dense class 的条件分支 |
| Memory Stream | `retrieval/methods/memory_stream.py` | graph rerank engine |
| GraphRAG | `retrieval/methods/graph_rag/`，必要时新增并列 graph builder | provenance graph builder 的条件分支 |
| MemGPT-style | `retrieval/methods/memgpt_style.py` | 通用 retrieval Context |
| GAT | `models/graph_retriever/internals/neural/` 下并列 encoder | R-GCN layer 的大量 if |
| 2WikiMultiHopQA | `datasets/twowiki/` | HotpotQA converter |
| tool trajectory | `datasets/tool_trajectory/` 与新增 graph rules | HotpotQA edge rules 的 dataset if |
| `tool_dependency` | 新 `ToolDependencyEdgeRule` | `BridgeEdgeRule` |
| `parameter_flow` | 新 `ParameterFlowEdgeRule` | `EntityOverlapEdgeRule` |

原则：

- 新需求优先新增并列实现。
- 只有真实共享语义稳定后，才向下提取公共能力。
- 不因“未来可能会需要”提前创建动态插件框架。

## 12. 第二轮审阅结论

本节提出的核心决策：

1. 删除 `types.py` 中心文件，按领域拆分 contract 与内部类型。
2. `validation.py` 拆包，但保持函数式。
3. `graphs.py` 改为 `GraphBuilder + EdgeAccumulator + 并列 GraphEdgeRule`。
4. 保留静态 retrieval catalog，但将 catalog、resolver 和 factory 分离。
5. 删除 `RetrievalBuildContext`，改为组合 runtime 与按方法族区分的 typed build request。
6. 将 flat retriever 移入 retrieval 领域，不再使用含义模糊的顶层 `indexes/`。
7. 将 graph rerank 拆为 method、engine、component、candidate、normalization 和 graph view。
8. 将 tuning 从普通 retrieval runtime 中隔离。
9. 将 train pair generation 从 learned model 子系统移出，形成可复用的训练数据领域。
10. 将 learned 子包改组为 `models/graph_retriever/`，分离 config、factory、training、dev evaluation、inference 和内部 tensor/neural 实现。
11. inference 与 training 共同依赖 model factory，消除 inference 依赖 training 的反向关系。
12. 本轮只为 workflow 保留窄 integration ports，不为旧内部 import 保留全量 facade。

## 13. 当前审阅边界

本文当前已经覆盖：

- 总体结构。
- 依赖方向。
- Context 替代方案。
- workflow 兼容边界。
- 全部现有核心模块的职责迁移。
- 必要的对象化边界。
- 应保持函数式的边界。
- 未来需求的自然落点。

本文尚未进入实施阶段。实施批次、每批回归基线、测试矩阵、文档同步范围和删除顺序将在本轮设计审阅通过后追加。

## 14. 第三轮审阅：实施层设计

总体结构和逐模块职责已经确认。真正实施前，还需要冻结迁移策略、行为等价标准、测试矩阵、删除顺序和 OpenSpec 拆分方式。

本轮重构的风险不在单个 move 操作，而在多个 move 同时发生后难以判断行为偏差来自哪里。因此，实施必须遵循以下原则：

```text
先建立基线
  -> 先新增目标位置
  -> 再迁移一个边界
  -> 立即更新调用方
  -> 运行该边界的回归验证
  -> 搜索残余 import
  -> 最后删除旧实现
```

禁止：

- 一次性搬完整个 `graph_memory/`。
- 在移动代码的同时修改算法。
- 在移动代码的同时优化性能。
- 在移动代码的同时改变 CLI。
- 先删除旧文件，再批量修复 import。
- 为了让迁移“暂时能跑”而新增宽泛 re-export。

### 14.1 重构提交的判定标准

每个实施批次都必须满足：

1. CLI 参数名、默认值和输出路径语义不变。
2. workflow 不修改。
3. 对应领域的 focused tests 通过。
4. 对应领域的行为等价 fixture 通过。
5. `basedpyright` 不新增 error。
6. 不存在指向已迁移旧模块的残余 import。
7. 不引入新的万能 Context、长参数链或反向依赖。

如果某个批次无法满足这些条件，应缩小批次，而不是把未验证状态继续传递到下一批。

## 15. 行为等价标准

“无行为变化”不能只用“pytest 通过”表达。不同输出需要不同粒度的等价检查。

### 15.1 必须完全一致的内容

以下内容要求完全一致：

- CLI 参数名称。
- CLI 默认值。
- CLI required/optional 关系。
- CLI parser 的 `choices`。
- workflow 生成的 stage 顺序。
- workflow 生成的底层命令参数。
- 静态 retrieval catalog 的公开 method 名称和 capability。
- HotpotQA conversion 后的 input、label 和 compatibility artifact。
- graph artifact 的 node 顺序、edge 顺序、edge 类型、权重和 directed 字段。
- train pair artifact 的 row 顺序、sample type、label 和 summary 统计。
- graph rerank config parse、默认值和 grid。
- graph rerank 候选扩展、component 顺序、score 组合和 tie-break。
- ranked result 中的 task 顺序、method、ranked node 顺序和 retrieved subgraph。
- metric 表格的字段和数值。
- checkpoint schema、config 字段和 relation vocab 顺序。
- trainable batching 的 node 顺序、sample 顺序、relation id 顺序和 dtype。

### 15.2 需要归一化后比较的内容

以下内容本来就具有运行时变化，不要求原始文本逐字节一致：

| 内容 | 比较方式 |
|---|---|
| `started_at`, `finished_at`, `created_at` | 排除时间值，验证字段存在且格式合法 |
| `latency_ms`, timings | 验证字段存在、有限且非负；不要求数值一致 |
| environment record | 验证必要键存在；不要求机器相关值一致 |
| checkpoint 文件二进制 | 解析后比较 schema、config 和 tensor 结构，不比较序列化字节 |
| GPU 训练数值 | 使用既有容差和 ranking invariant；不承诺跨设备逐位一致 |

### 15.3 CPU tiny fixture 的更强要求

对于不依赖外部模型下载的 CPU tiny fixture，应尽可能要求固定随机种子下完全一致：

- graph construction artifact 完全一致。
- BM25 ranking 和 score 完全一致。
- graph rerank ranking 和 score 完全一致。
- train pair artifact 完全一致。
- tensorization tensor 完全一致。
- model forward logits 完全一致。
- 单步 optimizer 后 state dict 完全一致。

如果纯移动代码后这些 fixture 出现差异，应视为真实回归，而不是接受新的容差。

### 15.4 Dense 路径比较

真实 sentence-transformer dense 路径可能受依赖版本、设备和底层数值库影响。重构验收分两层：

1. 使用 fake encoder 的 fixture 要求 embedding 调用参数、ranking、score 和 tie-break 完全一致。
2. 使用现有本地 encoder 的 smoke run 验证完整 CLI 可执行、输出 schema 合法、ranking invariant 成立。

本轮不调整 dense encoder batching、缓存或调用次数。共享 `DenseEncodingService` 仍然是后续独立变更。

## 16. 基线冻结

在迁移第一行应用代码之前，新增一个基线冻结任务。

### 16.1 现有测试基线

记录：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-refactor-baseline -p no:cacheprovider
uv run basedpyright --outputjson --level error
openspec validate --all --strict
```

如果 `uv run` 受到本机缓存权限影响，改用 repo `.venv` 中的对应命令。

### 16.2 CLI 契约快照

为以下脚本增加 parser contract tests：

```text
scripts/experiment.py
scripts/build_graphs.py
scripts/run_retrieval.py
scripts/tune_graph_rerank.py
scripts/build_train_pairs.py
scripts/train_graph_retriever.py
scripts/run_trainable_retrieval.py
scripts/evaluate_retrieval.py
scripts/aggregate_tables.py
```

测试内容：

- 参数名称。
- 默认值。
- required/optional。
- choices。
- compatibility alias，例如 `evaluate_retrieval.py --gold`。

不建议只比较整段 `--help` 文本。帮助文本中的换行格式可能随 Python argparse 版本变化。应直接检查 parser action。

### 16.3 Workflow 契约快照

使用 tiny 临时 run root，记录并断言：

- `scripts/experiment.py init` 产生的 manifest 核心字段。
- `scripts/experiment.py plan` 产生的 stage 顺序。
- 每个 stage 的 command 及参数。
- method narrowing。
- profile mapping。
- ablation selection。
- `--ablations-only` 的 fail-fast 行为。

对时间字段和临时绝对路径做归一化。

### 16.4 领域级 golden fixtures

新增固定 tiny fixture，覆盖：

```text
HotpotQA raw example
  -> input + labels + compatibility record
  -> graph artifact
  -> bm25 ranked result
  -> fake-dense ranked result
  -> graph-rerank ranked result
  -> train pairs + summary
  -> graph tensorization
  -> model forward
  -> one-step CPU training
```

fixture 应尽量使用小型内存对象，不依赖真实数据目录和外部下载。

## 17. 分批迁移顺序

### 17.1 Batch 0：OpenSpec 与基线冻结

范围：

- 创建 OpenSpec proposal、design、spec 和 tasks。
- 记录完整测试基线。
- 增加 CLI parser contract tests。
- 增加 workflow plan contract tests。
- 增加领域级 golden fixtures。

禁止：

- 不移动生产代码。
- 不改算法。

验收：

- 新增回归测试在当前代码上通过。
- 后续每个 batch 都能复用同一组 fixture。

### 17.2 Batch 1：拆分公共 contract、validation 与 infrastructure

范围：

```text
contracts/
validation/
infrastructure/
```

操作：

- 按第 9.1、9.2、9.3 节拆分 `types.py`、`validation.py`、`io.py` 和 `observability.py`。
- 更新已经迁入 `contracts/`、`validation/` 和 `infrastructure/` 的调用方 import。
- 为 workflow 保留窄 `io.py` 与 `observability.py` integration port。
- 暂时保留算法模块位置。

注意：

- Batch 1 不删除 `types.py`。
- Batch 1 不为了删除 `types.py` 提前创建 retrieval、models、training_pairs 等后续领域包。
- `types.py` 中的 artifact contract 可以迁入 `contracts/`；尚未迁移领域的内部类型继续暂留 `types.py`。
- 每个后续领域批次迁移自己拥有的类型，并同步缩小 `types.py`。
- 不改变任何 dataclass 字段、默认值、TypedDict 字段或 validator 行为。

验收：

- artifact validation tests 全部通过。
- IO 与 run summary tests 全部通过。
- `contracts/`、`infrastructure/` 和已迁移的 `validation/` 模块不再从 `graph_memory.types` 获取 artifact contract。
- 新增代码不引入新的 `from graph_memory.types`。
- 根目录 `io.py` 与 `observability.py` 只包含允许的窄 re-export。

### 17.3 Batch 2：拆分 dataset 与 text 领域

范围：

```text
datasets/
text/
```

操作：

- 按第 9.4、9.5 节移动 HotpotQA parser、converter、compatibility artifact 和 split helper。
- 拆分 token、lexical 和 entity helper。
- 更新 scripts 与 tests import。

注意：

- 保持 raw parsing error 文本。
- 保持 conversion 顺序。
- 保持 tokenization、IDF 和 lexical score 公式。

验收：

- HotpotQA golden artifact 完全一致。
- text、entity 和 conversion focused tests 全部通过。

### 17.4 Batch 3：建立 graph construction 领域

范围：

```text
graphs/
```

操作：

- 按第 9.6 节提取 `GraphBuilder`、`PreparedGraphInput`、`EdgeAccumulator` 和并列 `GraphEdgeRule`。
- 提取 `GraphIndex`、graph statistics 和 graph views。
- 保持 edge rule 执行顺序：

```text
SequentialEdgeRule
QueryOverlapEdgeRule
EntityOverlapEdgeRule
BridgeEdgeRule
```

注意：

- 本批允许对象化，但不允许修改 rule 内部算法。
- `PreparedGraphInput` 只服务一次构图。
- `GraphIndex` 只服务按 task 查询。

验收：

- graph golden artifact 完全一致。
- edge 顺序和权重完全一致。
- graph validation 与 evaluation connectivity tests 通过。

### 17.5 Batch 4：建立 evaluation 领域

范围：

```text
evaluation/
```

操作：

- 按第 9.12 节拆分 metric primitive、connectivity、service、tables 和 failure cases。
- 引入 `GraphConnectivity` 小对象封装 adjacency 派生状态。
- 保留 `evaluate_results()` 纯函数入口。

注意：

- 不修改指标定义。
- 不修改 CSV 列。
- 不改变 task join 和 fail-fast 行为。

验收：

- metric golden row 完全一致。
- failure case 输出完全一致。
- evaluation focused tests 通过。

### 17.6 Batch 5：建立 retrieval 核心领域并删除万能 Context

范围：

```text
retrieval/contracts.py
retrieval/catalog.py
retrieval/requests.py
retrieval/resolver.py
retrieval/factory.py
retrieval/execution/
retrieval/methods/flat/
retrieval/signals.py
```

操作：

- 按第 8、9.7、9.8、9.9、9.14 节移动 flat retriever 与 seed signal。
- 引入 typed runtime 与 method-family build request。
- 删除 `RetrievalBuildContext`。
- 将 `query_prefix`、`passage_prefix` 收入 `DenseConfig`。
- 将结果 assembly 与 token approximation 移入 execution 子包。
- 保留根目录 `retrieval_registry.py` integration port，供 workflow 使用。

注意：

- 本批先处理 flat BM25 和 dense。
- graph rerank method 仍可临时通过新 factory 调用旧 rerank engine。
- trainable graph method 仍可临时通过新 factory 调用旧 learned inference。
- 临时适配只能放在 factory 的明确分支内，不新增宽 facade。

验收：

- BM25 和 fake-dense ranked result 完全一致。
- `scripts/run_retrieval.py` CLI contract 不变。
- retrieval catalog capability query 不变。
- 全仓不存在 `RetrievalBuildContext`。
- 高层 `run_retrieval` 路径不出现散装 `query_prefix` 透传。

### 17.7 Batch 6：拆分 graph rerank 与 tuning

范围：

```text
retrieval/methods/graph_rerank/
retrieval/tuning/
```

操作：

- 按第 9.10、9.11 节提取 method、engine、components、candidate expansion、normalization、debug 和 tuning。
- 将 induced subgraph 和 adjacency graph view 放入 `graphs/views.py`。
- 使用 `GraphRerankEngine` 和 `GraphRerankTuner` 表达有状态组合。

注意：

- component 顺序、归一化和 score 组合公式完全不变。
- tuning cache 仍然只存在于单次 invocation 内存中。
- 不实现 `lambda_path` 新逻辑。

验收：

- graph-rerank golden result 完全一致。
- tuning selected config 完全一致。
- `scripts/tune_graph_rerank.py` CLI contract 不变。
- 删除旧 `rerank.py`、`rerank_config.py` 和 `tuning.py` 前完成残余 import 搜索。

### 17.8 Batch 7：提取 training pair 领域

范围：

```text
training_pairs/
```

操作：

- 按第 9.13 节将 pair generation 从 `learned/data.py` 移出。
- 引入 `TrainPairBuilder` 与并列 `NegativeSampler`。
- 保持 sampler 顺序、random state、去重和截断语义。

注意：

- 本批不修改 negative ratio。
- 本批不实现 Dense-FT。
- sampler 可以依赖 retrieval flat method 或 seed signal，但不能依赖 trainable model。

验收：

- pair artifact 和 summary 完全一致。
- `scripts/build_train_pairs.py` CLI contract 不变。
- 全仓不存在对 `graph_memory.learned.data` 的依赖。

### 17.9 Batch 8：重组 trainable graph model

范围：

```text
models/graph_retriever/
```

操作：

- 按第 9.15 至 9.21 节移动 config、checkpoint、tensorization、batching、neural components、factory、training、dev evaluation 和 inference。
- 引入 `GraphRetrieverBatchBuilder`。
- 引入 `GraphScoringModelFactory`。
- 将 retrieval adapter 放入 `retrieval/methods/trainable_graph.py`。
- 消除 inference 对 training 的依赖。
- 保留根目录 `training_config.py` integration port，供 workflow 使用。

注意：

- 模型 forward 数学完全不变。
- relation vocab 顺序完全不变。
- batching 拼接顺序完全不变。
- checkpoint schema 完全不变。
- ablation 映射完全不变。

验收：

- tensorization golden tensor 完全一致。
- model forward golden logits 完全一致。
- CPU one-step training golden state 完全一致。
- checkpoint round-trip tests 通过。
- trainable retrieval ranking tests 通过。
- `scripts/train_graph_retriever.py` 和 `scripts/run_trainable_retrieval.py` CLI contract 不变。

### 17.10 Batch 9：删除旧模块与全量验证

范围：

- 删除已经无调用方的旧模块。
- 更新 docs 中的实现路径。
- 增加架构依赖测试。

预期删除：

```text
graph_memory/types.py
graph_memory/validation.py
graph_memory/hotpotqa.py
graph_memory/splits.py
graph_memory/text.py
graph_memory/entities.py
graph_memory/graphs.py
graph_memory/indexes/
graph_memory/retrieval.py
graph_memory/rerank.py
graph_memory/rerank_config.py
graph_memory/tuning.py
graph_memory/evaluation.py
graph_memory/learned/
```

预期保留的根目录 integration ports：

```text
graph_memory/io.py
graph_memory/observability.py
graph_memory/retrieval_registry.py
graph_memory/training_config.py
graph_memory/experiment.py
```

验收：

- 全仓残余 import 搜索为空。
- 全量 tests 通过。
- type check 通过。
- OpenSpec strict validation 通过。
- workflow init、plan、status 和轻量 smoke run 通过。

## 18. 测试矩阵

### 18.1 按领域验证

| 领域 | 必须验证 |
|---|---|
| contracts | 字段、默认值、TypedDict 结构不变 |
| validation | 合法输入继续通过；非法输入继续 fail-fast；异常类型不变 |
| datasets | HotpotQA parse、cleaning、conversion 和 compatibility artifact 不变 |
| text | token、entity、IDF 和 lexical score 不变 |
| graphs | node、edge、顺序、权重、去重和 statistics 不变 |
| retrieval flat | BM25、fake dense 的调用参数、score、ranking 和 tie-break 不变 |
| graph rerank | candidate、component、normalization、score、subgraph 和 tuning 不变 |
| evaluation | metric、table、failure case 和 connectivity 不变 |
| training pairs | sampling 顺序、随机性、去重、summary 和 validation 不变 |
| model tensorization | relation vocab、edge expansion、filter、dtype 和 tensor 顺序不变 |
| model neural | forward、backward、ablation construction 和 state dict 不变 |
| checkpoint | schema、load、save、method mismatch failure 不变 |
| CLI | 参数名称、默认值、choices、required、alias 不变 |
| workflow | stage、command、profile、method、variant、manifest 和 fail-fast 不变 |

### 18.2 全量验证命令

最终至少执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-core-refactor -p no:cacheprovider
uv run basedpyright --outputjson --level error
openspec validate --all --strict
```

补充：

- 如果 `uv run` 因本机缓存权限失败，使用 repo `.venv` 的同类命令。
- 如果 type checker 输出大量 warning，验收以 `--level error` 的 error 数量为准。
- `.pytest_tmp/` 在本机存在权限问题时，不应继续使用该目录作为 basetemp。

### 18.3 架构依赖测试

新增：

```text
tests/test_architecture_dependencies.py
```

测试使用标准库 `ast` 扫描 import，不引入额外依赖。

至少断言：

- `contracts/` 不 import 算法包。
- `graphs/` 不 import retrieval、training_pairs、models、evaluation 或 application。
- `retrieval/` 不 import application。
- `models/graph_retriever/` 不 import application 或 scripts。
- `infrastructure/` 不 import datasets、graphs、retrieval、training_pairs、models 或 evaluation。
- 删除旧模块后，不再出现旧 import 路径。
- 根目录 integration port 只 import 批准的新目标模块。
- `graph_memory/experiment.py` 仍然只承担 workflow facade。

这个测试非常重要。目录分层如果没有自动化约束，几次快速开发后会重新退化。

## 19. 删除顺序

每个旧模块都按以下顺序删除：

```text
1. 新建目标模块
2. 移动逻辑
3. 更新 production import
4. 更新 scripts import
5. 更新 tests import
6. 运行 focused tests
7. rg 搜索旧路径
8. 删除旧模块
9. 再次运行 focused tests
```

删除前必须执行：

```powershell
rg -n "graph_memory\.<old_module>|from graph_memory\.<old_module>|import graph_memory\.<old_module>" graph_memory scripts tests docs
```

仅允许五个根目录 integration port 留存。`types.py` 是临时迁移例外，只能逐批缩小，并必须在 Batch 9 删除。

最终禁止因为迁移方便而保留：

- `types.py` 大型 re-export。
- `retrieval.py` 大型 re-export。
- `rerank.py` 大型 re-export。
- `learned/` 整包兼容 re-export。
- `indexes/` 整包兼容 re-export。

## 20. 命名规则补充

目录和对象命名需要让读者直接判断层级。

| 名称 | 语义 |
|---|---|
| `contracts` | 稳定数据语言，不包含算法 |
| `application` | 一个完整用例的高层编排 |
| `service` | 一个领域操作的入口，不解析 CLI，不做文件 IO |
| `catalog` | 静态元数据，不构造重型对象 |
| `resolver` | 将宽边界输入解析为精确内部请求 |
| `factory` | 根据精确 config 或 request 构造对象图 |
| `builder` | 逐步构造领域对象或 batch |
| `runtime` | 可复用的运行时状态组合 |
| `context` | 仅限单次操作内聚状态 |
| `views` | 从已有领域对象派生只读视图 |
| `internals` | 不应被其他领域直接依赖的低层实现 |

避免：

- `utils.py`
- `helpers.py`
- `common.py` 中塞入领域逻辑
- `manager.py`
- `processor.py`
- `context.py` 作为任意参数容器

`common.py` 只允许存在于 contract 或 validation 等底层叶子位置，并且必须保持小而明确。

## 21. 文档同步规则

本设计文档位于 `docs/10-plans/`，因为它记录一次时间有界的重构方案。

实施完成后，将稳定结论提升到长期文档：

| 文档 | 同步内容 |
|---|---|
| `docs/30-design/architecture.md` | 新目录结构、依赖方向、application boundary、workflow integration ports |
| `docs/30-design/abstractions.md` | runtime composition、typed build request、Context 使用规则、对象与纯函数边界 |
| `docs/20-contracts/retrieval-contracts.md` | 新内部 retrieval build boundary；公开 method 和 artifact contract 不变 |
| `docs/20-contracts/model-contracts.md` | 更新实现模块路径；模型字段和 checkpoint schema 不变 |
| `docs/30-design/testing-strategy.md` | golden fixture、CLI parser contract、architecture dependency test |
| `docs/40-operations/implementation-handoff.md` | 新模块导航表 |
| `docs/40-operations/commands.md` | 确认 CLI 命令无需变化 |
| `docs/README.md` | 将本计划保留为历史入口，并指向更新后的稳定设计 |

规则：

- 不复制同一规则到多个 durable 文档。
- contract 语义留在 `20-contracts/`。
- 架构规则留在 `30-design/`。
- 操作命令留在 `40-operations/`。
- 本文最终作为迁移决策与执行历史保留。

## 22. OpenSpec 拆分建议

不建议把全部迁移放入一个超大 OpenSpec change。目标架构可以统一设计，但实施应拆成四个连续 change。

### 22.1 Change A：`refactor-core-foundations-and-graph-domain`

包含：

- Batch 0：基线冻结。
- Batch 1：contracts、validation、infrastructure。
- Batch 2：datasets、text。
- Batch 3：graphs。
- Batch 4：evaluation。

原因：

- 这些领域是 retrieval 和 model 的低层依赖。
- 行为大多是确定性 artifact 和 pure-function 逻辑，适合先稳定。

### 22.2 Change B：`refactor-retrieval-domain-boundaries`

包含：

- Batch 5：retrieval 核心、flat methods、catalog、resolver、factory、runtime composition。
- Batch 6：graph rerank、tuning。

核心验收：

- 删除 `RetrievalBuildContext`。
- 不再出现散装 dense 参数跨层传递。
- public method、CLI、ranking 和 tuning 结果不变。

### 22.3 Change C：`refactor-trainable-graph-model-domain`

包含：

- Batch 7：training pairs。
- Batch 8：models/graph_retriever。

核心验收：

- train pair artifact 不变。
- tensor、model、checkpoint、训练和推理行为不变。
- inference 不再依赖 training。

### 22.4 Change D：`finalize-core-package-refactor`

包含：

- Batch 9：删除旧模块。
- 架构依赖测试。
- 全量验证。
- durable docs 同步。

原因：

- 最终删除和文档提升应发生在所有迁移边界稳定之后。
- 该 change 可以清晰审计哪些兼容入口被刻意保留。

## 23. 实施前最终检查表

开始实现前，需要确认：

- [x] 外部兼容边界：CLI 与 artifact schema 冻结。
- [x] workflow 本轮不修改。
- [x] 允许改变内部 import 与数据表示。
- [x] 目标目录结构已确认。
- [x] 逐模块职责迁移已确认。
- [x] `RetrievalBuildContext` 本轮删除。
- [x] Context 使用规则已明确。
- [x] 对象化与纯函数边界已明确。
- [x] 未来需求落点已明确。
- [x] 分批迁移顺序已明确。
- [x] 行为等价标准已明确。
- [x] 测试矩阵已明确。
- [x] 删除顺序已明确。
- [x] durable docs 同步规则已明确。
- [x] OpenSpec change 拆分已明确。

如果本节审阅通过，下一步不是直接修改代码，而是创建 Change A 的 OpenSpec proposal，并先执行 Batch 0 基线冻结。
