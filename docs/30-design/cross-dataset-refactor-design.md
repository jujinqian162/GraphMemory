# 跨数据集 Task View 重构设计

日期：2026-06-17

状态：设计基线。本文描述跨数据集架构抽象；截至 2026-06-19，HotpotQA 已采用 dataset-owned record + projector + consumer request 的第一阶段实现。

## 1. 问题背景和产生原因

当前项目已经按目录拆出了 `datasets`、`retrieval`、`graphs`、`evaluation`、`registry` 和 `workflow` 等模块，但这些模块之间仍然共享一个过强的 HotpotQA 任务语义。也就是说，目录层面的模块化已经存在，语义层面的模块化还不够彻底。

当前 HotpotQA 主链路可以概括为：

```text
HotpotQA raw
  -> HotpotQARankingRecord / HotpotQALabelRecord
  -> HotpotQA projector
  -> TextRankingRequest / GraphBuildRequest / GraphRankingRequest / TemporalMemoryRankingRequest / EvidenceEvaluationRequest
  -> RetrievalMethod ranks candidate item ids
  -> evaluation compares gold_evidence_sentence_ids through EvidenceEvaluationRequest
```

这个设计在 Phase 1 是合理的。早期目标是先跑通 HotpotQA evidence retrieval，并比较 BM25、Dense、Graph Rerank、R-GCN、Dense-FT 等方法。HotpotQA 的自然表达就是 question、context sentences、supporting facts，因此将 sentence 转成 memory node，再对 node id 排序，是一个清晰、低成本、可验证的实验切片。

问题出现在新数据集和新 baseline 变多之后。LongMemEval-v2、2WikiMultiHopQA、MuSiQue、工具轨迹和 GraphRAG-style baseline 的自然任务表达不再都是“问题 + 文档句子 + gold evidence sentence”。例如 LongMemEval-v2 更接近 trajectory/session/turn/haystack/assets 的长期记忆任务；2WikiMultiHopQA 更接近多跳实体、段落和关系路径任务。它们可以被投影成 evidence ranking，但这只是其中一种视图，不应该成为所有方法和指标的唯一内部语言。

旧中心 task 形态的局促感包括：

- 单个中心 task input 同时承担了 dataset conversion artifact、retriever input、graph builder input 和 evaluation join key 的职责。
- 单个中心 candidate item 的字段形状偏向 HotpotQA sentence：`source`、`sentence_id`、`position` 和 `document_sentence`。
- BM25、Dense、Memory Stream、Graph Builder、training pair builder 和 evaluator 都直接读取同一套 task shape。
- Evaluation 默认绑定 HotpotQA supporting sentence IDs、graph connectivity 和 ranked node ids，导致 turn full support、session full support、answer quality 等指标没有自然位置。
- Registry 当前主要描述 method lifecycle、依赖和训练产物，还没有描述 dataset 能提供哪些 task view、method 消费哪些 request view、metric suite 消费哪些 prediction/eval view。
- Workflow 当前按 method lifecycle 拼 stage，默认所有方法共享 prepare/build_graph/retrieve/evaluate 语义；这适合 HotpotQA evidence retrieval，但不适合跨任务族扩展。

这不是早期设计错误，而是早期设计达到了它当时的边界。第一阶段的目标是把一个 evidence retrieval benchmark 做成可靠实验系统；下一阶段的目标是让系统成为可扩展的 memory retrieval / graph retrieval 实验平台。

## 2. 设计目标

本设计的核心目标是：让新增 dataset、新增 retriever、新增 metric suite 互相解耦，并通过显式连接层完成语义转换。

目标：

- Dataset 模块负责 raw data 的解析、清洗、校验、split 和资产管理，不直接适配具体 retriever。
- Retriever 模块只声明自己需要的输入 request 和输出 prediction，不依赖任何 dataset raw schema。
- Graph 构建不再假设所有数据集都是 HotpotQA sentence graph，而是从统一 Graph Build View 和 dataset-specific rule set 构造图。
- Evaluation 不依赖 retriever 内部语义，而是从 prediction view、eval labels 和 projection context 计算 metric suite。
- Workflow 和 registry 从“按 method lifecycle 固定 stage”升级为“按 capability / view dependency 规划 stage”。
- 连接层成为一等公民：dataset-to-task、task-to-request、prediction-to-metric-unit 都通过 projection/adapter 表达。

非目标：

- 本文不定义具体 JSON schema、Python 类字段或 CLI 参数。
- 本文不写实施步骤、任务拆分或迁移顺序。
- 本文不要求当前 HotpotQA artifact 立即改变。
- 本文不引入动态插件系统、通用依赖注入容器或通用 pipeline engine。
- 本文不把所有 graph builder、retriever 或 evaluator 强行合并成一个万能父类。

## 3. 总体架构语言

本设计采用几个经典架构概念：

- Ports and Adapters：dataset、retriever、graph、evaluation 都通过稳定端口交互，具体数据集和方法作为 adapter 挂接。
- Anti-Corruption Layer：projection 层隔离不同 bounded context 的语义，避免 LongMemEval、HotpotQA、2Wiki 的原始概念污染 retriever 实现。
- Interface Segregation：只保留少数 task view 和少数 retriever request，而不是每个 dataset/method 组合都定义一套接口。
- Capability Negotiation：registry 描述“谁能提供什么 view、谁需要什么 view、谁产出什么 prediction、谁能评估什么 prediction”。
- Stable Core, Replaceable Edges：核心 task/request/prediction/eval view 稳定，dataset adapter、graph rule set 和 metric suite 可替换。

## 4. 分层模型

目标数据流：

```text
Dataset Raw
  -> Dataset Adapter
  -> Task View
  -> Retriever Request Projection
  -> Retrieval Method
  -> Prediction View
  -> Evaluation Projection
  -> Metric Suite
```

带图方法的数据流：

```text
Task View
  -> Graph Build View
  -> Graph Rule Set
  -> Graph Artifact
  -> Graph Ranking Request / Context Gathering Request
  -> GraphRerank / GraphRAG / trainable graph method
```

带训练方法的数据流：

```text
Task View + Label View + optional Graph View
  -> Training View
  -> Method-specific training adapter
  -> Model Artifact
  -> Retrieval Method
```

各层职责：

| Layer | 责任 | 不应承担 |
|---|---|---|
| Dataset Adapter | 解析 raw schema，清洗数据，校验数据集完整性，管理 assets 和 split | 调用 BM25/Dense/GraphRerank，不计算最终指标 |
| Task View | 表达 benchmark 被转成的任务语义 | 暴露 raw dataset 私有字段，绑定某个 retriever |
| Projection | 在 Task View、Retriever Request、Eval Unit 之间做语义映射 | 实现 retrieval algorithm 或 metric formula |
| Retriever Request | 表达一类方法运行所需的最小输入 | 携带 dataset raw schema 或 label-only 信息 |
| Retrieval Method | 根据 request 产出 prediction | 读取 labels，计算 dataset-specific metric |
| Prediction View | 表达方法输出，如 ranked items、context items、answer、latency | 知道 turn full support 或 session full support 如何计算 |
| Eval View | 表达 label、id mapping、metadata 和 metric context | 改变 retriever 输出或补做 retrieval |
| Metric Suite | 计算某个 dataset/task 的指标集合 | 反向约束 retriever 读取 dataset-specific fields |
| Workflow/Registry | 根据 capability 生成 stage DAG 和 artifact manifest | 用字符串散落判断具体方法和数据集 |

### 4.1 Split role 和 label 可见性

跨数据集之后，split 不应只是 Dataset Adapter 的内部细节，也不应散落在 workflow 命令参数里。Dataset Adapter 可以暴露 raw/official split、可用资产和每条记录的稳定 id；一次实验实际使用哪个 source、count、seed、offset、coverage cap 和 asset subset，应由 Benchmark Recipe / Split Policy 写入 manifest，并在生成下游 stage config 前固定下来。

label 可见性必须同时受端口和 split role 约束：

- train labels 只能进入训练、训练样本构造和训练期诊断。
- dev labels 可以进入 tuning、model selection 和 dev evaluation，但不能反向改变 test projection、test graph build 或 test retrieval request。
- test labels 只能在 prediction artifact 已经固定之后进入 evaluation 和 failure-case analysis。
- Graph Build View、Retriever Request 和 retrieval-visible Graph Artifact 对任何 split 都不得读取 gold answer、gold evidence、gold dependency path 或 judge label。
- 如果某个方法需要 coverage 调整，例如 Memory Stream 的 cleaned importance artifact 只覆盖部分 test 任务，调整必须在 manifest / split policy 层显式记录，并在 RetrieveIO 和 EvaluateIO 配置生成前同时作用于 input 与 label artifact，避免 retrieval 与 evaluation 的 task 集不一致。

## 5. 少数 Task View

Task View 是“数据集被投影成什么研究任务”的稳定表达。它属于实验语义层，不属于任何具体 retriever。

建议保留少数高层 Task View：

| Task View | 适用问题 | 典型数据集 |
|---|---|---|
| Evidence Ranking View | 从候选 evidence items 中找出支持证据 | HotpotQA、2WikiMultiHopQA、MuSiQue |
| Context Gathering View | 从长期记忆、trajectory、session 或 haystack 中收集回答所需上下文 | LongMemEval-v2、agent trajectory benchmarks |
| Graph Build View | 从任务对象中构造图节点、边和 edge semantics | HotpotQA、LongMemEval-v2、2WikiMultiHopQA、GraphRAG |
| Training View | 为 trainable retriever 提供正负样本、pair、label 和可选图上下文 | Dense-FT、R-GCN、未来 trainable GraphRAG |
| Answer Evaluation View | 对 answer generation 或 judge-based answer quality 评估提供 label/context | LongMemEval-v2、multi-hop QA generation |

当前 `HotpotQARankingRecord` 是 HotpotQA-owned prepared artifact；它通过 projector 进入 Evidence Ranking 相关 request，而不应作为所有未来任务的通用核心语言。

## 6. 少数 Retriever Request

Retriever Request 是“某类方法实际运行需要的最小输入”。它属于方法运行层，不属于 dataset。

建议保留少数高层 Request：

| Retriever Request | 消费方法 | 输入语义 |
|---|---|---|
| Text Ranking Request | BM25、Dense、Dense-FT、text-only seed retrievers | query text + candidate item texts |
| Graph Ranking Request | Graph Rerank、R-GCN-style graph retrieval | query node + candidate nodes + graph edges + optional seed scores |
| Temporal Memory Ranking Request | Memory Stream-style retrievers | query + memory items + recency/position/importance signals |
| Context Gathering Request | GraphRAG、LongMemEval-style retrieval | question + text store + graph/session/trajectory context |
| Answer Request | answer generation or reader models | question + retrieved context + optional assets |

BM25 和 Dense 不应该知道候选文本来自 HotpotQA sentence、LongMemEval turn、2Wiki paragraph 还是 tool trace chunk。它们只消费 Text Ranking Request，并产出 Ranking Prediction。

GraphRerank 不应该知道图来自 HotpotQA document order、LongMemEval session order 还是 2Wiki entity relation。它只消费 Graph Ranking Request。

GraphRAG 不应该知道数据集 raw schema。它只消费 Context Gathering Request，并产出 Context Prediction 或 Ranking Prediction。

## 7. Graph 构建设计

Graph 构建的扩展点不应放在 GraphRerank 或 GraphRAG 内部，而应放在 Graph Build View 和 Graph Rule Set。

设计边界：

```text
Dataset-specific Task View
  -> Graph Build View
  -> Dataset/task-specific Graph Rule Set
  -> Graph Artifact
  -> Graph Ranking Request / Context Gathering Request
```

Graph Build View 表达通用图构造输入：

- task identity
- query node 或 query context
- candidate nodes
- known edges from dataset, if any
- node grouping metadata, such as document/session/turn/entity
- asset references, if the benchmark includes screenshots or multimodal context
- edge rule hints, if the dataset provides trustworthy structural signals

Graph Rule Set 表达如何建边：

- HotpotQA 可以使用 sentence order、query overlap、entity overlap、bridge relation。
- LongMemEval-v2 可以使用 turn order、same session、action-observation relation、trajectory provenance、haystack relation、asset link。
- 2WikiMultiHopQA 可以使用 paragraph-entity mention、entity co-occurrence、input-visible wiki relation、hyperlink、question entity link。

`supporting paragraph`、gold relation path、gold dependency edge 这类监督信息不属于 retrieval-visible Graph Rule Set。它们可以进入 Eval View、Training View 或 dev tuning 选择逻辑，但不能参与 test graph construction 或 retrieval request 构造。

Graph Artifact 应只表达图本身和必要 metadata，不应把 metric-specific gold labels 混入 retrieval-visible 图中。

GraphRerank 的稳定依赖是：

```text
Graph Ranking Request
  -> ranked item ids + optional retrieved subgraph
```

GraphRAG 的稳定依赖是：

```text
Context Gathering Request
  -> retrieved context items / ranked evidence ids / optional reasoning path
```

因此，适配 LongMemEval-v2 的 GraphRerank 不需要修改 GraphRerank 方法代码；新增的是 LongMemEval Graph Build View projection 和 LongMemEval Graph Rule Set。适配 2WikiMultiHopQA 的 GraphRAG 也不需要修改 GraphRAG 方法代码；新增的是 2Wiki Graph Build View projection、entity/paragraph graph rule set 和 2Wiki Metric Suite。

## 8. Evaluation 设计

Evaluation 也需要从 retriever 中解耦。Retriever 只产出 Prediction View；Metric Suite 决定如何解释这些 prediction。

稳定数据流：

```text
Prediction View
  + Eval View
  + Evaluation Projection Context
  -> Metric Suite
  -> Metric Rows
```

Prediction View 示例：

| Prediction View | 内容 | 不包含 |
|---|---|---|
| Ranking Prediction | task id、ranked item ids、scores、latency、optional trace / retrieved subgraph | gold labels、metric names |
| Context Prediction | task id、retrieved context ids/text refs、scores、latency、optional reasoning path refs | answer correctness、gold path labels |
| Answer Prediction | task id、answer text、optional citations、latency | metric formula |

当前 `ranked_results_{method}.json` 中的 `retrieved_subgraph` 可以被理解为 Ranking Prediction 的 optional trace。它是方法输出的可解释性记录，不是 metric label，也不应包含 gold evidence marker。未来 GraphRAG 或 path-oriented 方法可以输出 reasoning path refs，但这些 refs 只能指向 retrieval 过程中实际使用的节点、边或 context item，不能直接写入 gold path。

Eval View 示例：

| Eval View | 内容 |
|---|---|
| Evidence Eval View | gold evidence item ids、optional dependency/path labels |
| LongMemEval Eval View | gold answer、gold support ids、item-to-turn map、item-to-session map、domain/type metadata |
| MultiHop Eval View | gold paragraphs、gold entities、gold relation/path labels、answer labels |

LongMemEval-v2 的 turn full support 和 session full support 不应出现在 Dense、BM25、GraphRerank 或 GraphRAG 代码中。它们属于 Metric Suite 对 ranked item ids 的不同投影：

```text
ranked item ids
  -> project to evidence item ids
  -> project to turn ids
  -> project to session ids
```

然后 Metric Suite 计算：

```text
Evidence Recall@k
Turn Full Support@k
Session Full Support@k
Answer EM/F1/Judge Score, if answer prediction exists
Latency / Query
```

这保证了 Dense 这类方法只负责排序，不知道 turn/session/domain/difficulty 等 dataset-specific 概念。

## 9. Registry 和 Workflow 设计

下一轮 registry 不应只注册 method，还应注册 capability。

需要区分五类 registry / planner record：

| Registry | 负责什么 |
|---|---|
| Dataset Registry | 某个 dataset 能解析哪些 raw inputs，能提供哪些 Task View 和 Eval View |
| Benchmark Recipe / Split Policy Registry | 某次实验使用哪些 split source、count、seed、offset、asset subset 和 coverage rule |
| Projection Registry | 哪些 view 可以投影成哪些 Retriever Request、Graph Build View 或 Eval Unit |
| Method Registry | 某个 method 消费什么 request，产出什么 prediction，需要什么 model/graph/seed/tuning dependency |
| Metric Suite Registry | 某个 metric suite 消费什么 prediction 和 eval view，输出什么 metric family |

Workflow planner 的职责是做 capability matching：

```text
requested dataset + requested method + requested metric suite
  -> resolve benchmark recipe and split policy
  -> resolve dataset views
  -> validate asset coverage and label visibility
  -> insert required projections
  -> insert graph build stage if method needs graph request
  -> insert train/tune stages if method declares those capabilities
  -> insert matching evaluator
  -> emit artifact manifest
```

这样，当用户请求 `graph_rerank + longmemeval_v2` 时，workflow 不应通过字符串判断“longmemeval 特殊处理”。它应该看到：

```text
graph_rerank consumes Graph Ranking Request
longmemeval_v2 provides Context Gathering View and Graph Build View
projection registry can build Graph Ranking Request from those views
longmemeval metric suite can evaluate Ranking Prediction
```

当用户请求 `graphrag + 2wiki_multihopqa` 时，workflow 应看到：

```text
graphrag consumes Context Gathering Request
2wiki_multihopqa provides Evidence Ranking View and Graph Build View
projection registry can build graph/text store request
2wiki metric suite can evaluate ranked/context evidence and optional answer
```

## 10. Exported API 与模块内部 API

这里的 API 指架构端口，不代表必须以同名 Python 对象实现。最终代码可以选择 TypedDict、dataclass、Protocol、registry record 或 artifact schema，但可见性边界应保持一致。

| 模块/上下文 | Exported API | Internal API |
|---|---|---|
| Dataset | Dataset Definition、Dataset Adapter Port、Dataset Record Set、Asset Manifest、Official Split Metadata | raw field parser、dataset-specific cleaning helper、download/checksum helper |
| Task Views | Evidence Ranking View、Context Gathering View、Graph Build View、Training View、Answer Evaluation View | dataset-specific intermediate records、temporary normalization records |
| Projection | Projection Definition、Projection Registry、View-to-Request Adapter、Prediction-to-EvalUnit Adapter | concrete dataset/method mapping helper、id remapping tables |
| Retrieval | Method Definition、Retriever Request Port、Prediction View、Method Capability Record | BM25 scorer、Dense encoder wrapper、GraphRerank engine internals、GraphRAG traversal internals |
| Graphs | Graph Build View Port、Graph Rule Set、Graph Artifact、Graph Index/View | individual edge rule implementations、token/entity overlap helpers |
| Evaluation | Metric Suite Definition、Eval View、Metric Row、Metric Selection Key | metric primitive helpers、dataset-specific aggregation helper、failure-case formatter |
| Training | Training View、Train Artifact Contract、Trainable Method Capability | pair sampler internals、loss/model/trainer internals |
| Workflow/Registry | Capability Planner、Benchmark Recipe、Split Policy、Artifact Manifest、Stage Config Boundary | argv construction helper、path naming helper、resume/status implementation details |

重要边界规则：

- Exported API 是跨模块依赖点；Internal API 只能被本 bounded context 内部使用。
- Dataset Adapter 不导入 retrieval method implementation。
- Retrieval Method 不导入 dataset raw parser 或 metric suite。
- Metric Suite 不调用 retriever，只解释 prediction。
- Graph Rule Set 可以 dataset-specific，但 GraphRerank/GraphRAG 只能依赖 Graph Request。
- Label-only 信息只能进入 training、tuning 和 evaluation 端口，不能进入 retrieval-visible request。
- Label-only 信息还必须受 split role 限制：test labels 只能在 prediction 固化后的 evaluation / failure-case analysis 中出现。
- Split Policy 必须在 stage config 生成前固定 input、label、graph、prediction、metric 的 task 集；不得在 retrieval 或 evaluation 中临时补救 task coverage。
- 不通过一个万能 Context 串联所有字段；每个 request/view 都只表达一个内聚操作。

## 11. 典型适配场景

### 11.1 Dense 适配 LongMemEval-v2

```text
LongMemEval raw
  -> LongMemEval Dataset Adapter
  -> Context Gathering View
  -> Text Ranking Request projection
  -> Dense
  -> Ranking Prediction
  -> LongMemEval Eval View
  -> LongMemEval Metric Suite
```

Dense 不知道 turn/session/haystack/screenshot。它只看 query text 和 candidate texts。

### 11.2 GraphRerank 适配 LongMemEval-v2

```text
LongMemEval Context Gathering View
  -> Graph Build View
  -> LongMemEval Graph Rule Set
  -> Graph Artifact
  -> Graph Ranking Request
  -> GraphRerank
  -> Ranking Prediction
  -> LongMemEval Metric Suite
```

GraphRerank 不知道边来自 HotpotQA sentence order 还是 LongMemEval session/turn provenance。它只看 graph topology、edge type、seed scores 和 candidate ids。

### 11.3 GraphRAG 适配 2WikiMultiHopQA

```text
2Wiki raw
  -> 2Wiki Dataset Adapter
  -> Evidence Ranking View + Graph Build View
  -> 2Wiki entity/paragraph Graph Rule Set
  -> Context Gathering Request
  -> GraphRAG
  -> Context Prediction / Ranking Prediction
  -> 2Wiki Metric Suite
```

GraphRAG 不知道 2Wiki raw schema。它只看 graph、text store、query 和 retrieval config。

### 11.4 HotpotQA 继续作为 Evidence Ranking View

```text
HotpotQA raw
  -> HotpotQA Dataset Adapter
  -> Evidence Ranking View
  -> Text Ranking Request / Graph Build View / Training View
  -> existing methods
  -> Evidence Metric Suite
```

这使当前系统可以平滑演进：现有 `HotpotQARankingRecord` 保持 HotpotQA-owned artifact 角色，`TextRankingRequest`、`GraphBuildRequest`、`GraphRankingRequest`、`TemporalMemoryRankingRequest` 和 `EvidenceEvaluationRequest` 作为消费者侧稳定边界。

## 12. 设计不变量

以下规则应作为下一轮 refactor 的架构不变量：

- 新增 dataset 时，优先新增 Dataset Adapter、Task View projection、Graph Rule Set 和 Metric Suite；不应修改 BM25/Dense/GraphRerank 的核心实现。
- 新增 retriever 时，优先声明它消费的 Retriever Request 和产出的 Prediction View；不应读取 dataset raw schema。
- 新增 metric 时，优先新增 Metric Suite 或 Evaluation Projection；不应让 retriever 产出 metric-specific 字段。
- 图构建差异属于 Graph Build View 和 Graph Rule Set；不属于 GraphRerank 或 GraphRAG 方法本体。
- 评估粒度差异属于 Eval View 和 Metric Suite；不属于 Ranking Prediction。
- Registry 必须能回答 capability 问题，而不是只回答 method name 是否存在。
- Workflow stage 应由 capability dependency 推导，而不是靠 dataset/method 字符串散落判断。
- Benchmark Recipe / Split Policy 必须记录 split source、count、seed、offset、asset coverage 和任何 method-specific cap，并保证 input、label、graph、prediction、metric 的 task 集一致。
- 每个跨模块 API 必须有明确 owner；其他文档只引用，不复制 schema truth。

## 13. 当前代码的设计定位

当前代码可以按如下方式理解，为后续迁移提供稳定语义：

- `HotpotQARankingRecord` 是当前 HotpotQA prepared artifact，不应继续被视为所有 dataset 的永久通用 task type。
- `HotpotQALabelRecord` 是当前 HotpotQA label artifact，不应覆盖 LongMemEval 的 turn/session/answer labels。
- `MemoryGraph` 是当前 evidence graph artifact，不应假设所有 graph 都由 document sentence 顺序和 query/entity overlap 构造。
- `RetrievalMethod` 的“不给 labels、不算 metrics、不写文件”原则是正确的，应保留。
- 当前 method registry 的 lifecycle/dependency 思路是正确起点，但需要扩展为 view/request/prediction/metric capability registry。
- 当前 evaluate stage 是 HotpotQA evidence retrieval evaluator 的实现，不应作为所有未来 dataset 的全局 evaluator。

本文的核心判断是：下一轮重构不应只继续移动文件，而应把 `Task View -> Retriever Request -> Prediction View -> Eval View` 这条连接层建立为正式架构。这样，新增 LongMemEval-v2、2WikiMultiHopQA 或 GraphRAG-style baseline 时，主要工作落在 projection、graph rules 和 metric suite 上，而不是重写所有 baseline。
