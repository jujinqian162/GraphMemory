# Evidence Retrieval 与 Execution Provenance Retrieval 领域重构计划

> 状态：已锁定，OpenSpec 实施中  
> 分支：`feature/execution-provenance-retrieval-domain`  
> 当前阶段：领域重构已落地；具体 benchmark 由独立 OpenSpec change 适配
> 论文参考：`docs/raw/paper.tex` 中的 typed execution-provenance graph、semantic seed selection、provenance-constrained expansion 与 path scoring

## 1. 本文档要锁定的结论

这次重构不再把所有“带图的方法”视为同一种 graph retrieval。最终系统必须明确区分：

1. 传统 evidence retrieval 数据集提供或派生的 `EvidenceGraph`；
2. execution provenance 数据集原生提供的 `ExecutionProvenanceGraph`；
3. GraphRAG 从候选文本内部构建的 entity knowledge graph。

其中前两者是两种不同的 dataset graph 语义。第三种只是 GraphRAG 的方法内部索引，不是数据集图工件，也不能通过一个通用 `graph` 字段与前两者混用。

最终公开实验方法矩阵固定为：

| Benchmark family | 公开方法 |
|---|---|
| Traditional evidence retrieval：HotpotQA、2Wiki、MuSiQue | BM25、Dense、Dense-FT、GraphRAG、Dense R-GCN、Dense-FT R-GCN，共 6 个 |
| Execution provenance：`twowiki_provenance` | BM25、Dense、GraphRAG、Execution-Provenance Retriever、Provenance R-GCN，共 5 个 |

硬性限制：

- 不提供 `execution provenance dataset -> EvidenceGraph R-GCN` 的 projection。
- 不提供 `evidence retrieval dataset -> Execution-Provenance Retriever` 的 projection。
- 不再保留 `bm25_graph_rerank` 和 `dense_graph_rerank` 两个公开方法；它们不能仅改名冒充 GraphRAG。
- GraphRAG 只有一个公开方法 ID：`graphrag`。
- R-GCN 恢复为独立节点打分模型，移除 beam decoder、stop action、frontier state、dynamic oracle 和 beam loss。
- 第一版 Execution-Provenance Retriever 为不可训练方法；可训练版本不进入本次领域重构的必做范围。
- benchmark 适配不得把不具备原生图语义的数据源伪装成 execution provenance graph。
- 彻底删除 Memory Stream。当前分支的代码、配置、Registry、请求、调参、工作流、CLI、测试和活动文档中不得保留 Memory Stream 实现或兼容分支。

Memory Stream 的删除采用 current-only migration：旧配置中的 `memory_stream` 必须直接报 unsupported method；不得保留 alias、deprecated 字段、静默忽略、自动迁移、旧 artifact loader 或 tombstone registry entry。Git 历史和纯归档材料可以保留过去记录，但不得被当前代码、配置或活动 OpenSpec 工件引用为可用能力。

## 2. 为什么当前抽象必须拆开

当前代码中的 `MemoryGraph`、`GraphRankingRequest` 和 `GraphInputSource.GRAPH_ARTIFACT` 都把“graph”当成单一概念，但现有 graph 实际上是传统 evidence graph：

- 节点只有 `question` 与 `graph_item`；
- 边只有 `sequential`、`query_overlap`、`entity_overlap`、`bridge`；
- `GraphRankingRequest` 同时被 heuristic graph rerank 与两个 R-GCN 使用；
- Registry 只能表达“需要图工件”或“不需要图工件”，不能表达图的语义；
- `run_retrieve_stage(..., graphs=...)` 暴露了一个无类型的通用 graph 入口。

这一抽象无法安全容纳 execution provenance。即使二者都可以写成 nodes/edges，它们的节点含义、边方向、生成责任、训练标签和评价方式也完全不同。

重构后禁止出现以下接口：

```python
GenericGraphRankingRequest(graph: object, graph_kind: str)
```

也禁止 retriever 在运行时根据 `graph_kind` 分支解释同一个 graph 对象。

## 3. Request 是 dataset 与 retriever 的解耦边界

### 3.1 保留独立的具体 Request

领域层使用四种互不替代的具体 request：

```python
@dataclass(frozen=True)
class TextRankingRequest:
    task_id: str
    query_text: str
    candidates: Sequence[TextCandidate]


@dataclass(frozen=True)
class EvidenceGraphRankingRequest:
    task_id: str
    query_text: str
    candidates: Sequence[TextCandidate]
    graph: EvidenceGraph
    initial_scores: Mapping[str, float]


@dataclass(frozen=True)
class GraphRAGRequest:
    task_id: str
    query_text: str
    candidates: Sequence[TextCandidate]
    knowledge_graph: EntityKnowledgeGraph


@dataclass(frozen=True)
class ExecutionProvenanceRankingRequest:
    task_id: str
    query_text: str
    candidates: Sequence[TextCandidate]
    graph: ExecutionProvenanceGraph
```

字段语义：

- `TextRankingRequest.candidates`：可被公开检索和排序的文本单元。BM25、Dense、Dense-FT 只看到这些内容。
- `EvidenceGraphRankingRequest.graph`：传统 evidence dataset 的 evidence item 关系图，只供两个 R-GCN 使用。
- `EvidenceGraphRankingRequest.initial_scores`：R-GCN 的 Dense 或 Dense-FT seed feature，不代表 graph edge。
- `GraphRAGRequest.knowledge_graph`：GraphRAG 从当前 candidates 构建的 entity/relation 索引，不是 dataset gold graph。
- `ExecutionProvenanceRankingRequest.graph`：数据集轨迹中真实存在的 typed execution/dataflow graph。
- `ExecutionProvenanceRankingRequest.candidates`：该 provenance graph 中允许被检索的节点子集；每个 candidate ID 必须对应 graph node。Task/Agent 等上下文节点可以在图中但不必成为检索候选。

`RankingMethodRequest` 可以继续作为以上 concrete request 的 `TypeAlias` union，仅用于静态类型标注和统一执行循环。它不是一个装所有 graph 的实体类，不拥有通用 `graph` 字段，也不允许调用方绕过 concrete request 校验。

### 3.2 projection 责任

未来 projector 的合法关系固定为：

| Dataset family | 可产生的 request / artifact | 明确禁止 |
|---|---|---|
| Evidence retrieval | `TextRankingRequest`、`EvidenceGraphBuildRequest`、evidence labels | `ExecutionProvenanceRankingRequest` |
| Execution provenance | `TextRankingRequest`、`ExecutionProvenanceRankingRequest`、provenance labels | `EvidenceGraphBuildRequest`、`EvidenceGraphRankingRequest` |

GraphRAG 不要求每个 dataset 单独实现 `ToGraphRAGRequest`。所有支持文本检索的数据集先产生 `TextRankingRequest`，GraphRAG 自己的 request assembler 再执行：

```text
TextRankingRequest
  -> entity extraction / normalization
  -> EntityKnowledgeGraph
  -> GraphRAGRequest
```

这能保证 entity extraction 属于 GraphRAG 方法，而不是被复制到每个 dataset adapter。

R-GCN 的 request assembler 执行：

```text
TextRankingRequest + EvidenceGraph artifact + Dense(FT) seed scores
  -> EvidenceGraphRankingRequest
```

Execution-Provenance Retriever 不从普通文本或 evidence graph 猜图。兼容 adapter 必须直接提供 `ExecutionProvenanceRankingRequest`；显式标注为 synthetic 的离线转换器必须先生成独立、可审计的数据集。

## 4. 三种 graph 的所有权与语义

### 4.1 EvidenceGraph：传统 evidence dataset 专用

当前 `MemoryGraph` 更名为 `EvidenceGraph`，当前 `GraphBuildRequest` 更名为 `EvidenceGraphBuildRequest`。这是语义澄清，不改变 HotpotQA、2Wiki、MuSiQue 当前图构建规则。

建议节点类型：

- `question`
- `evidence_item`，替代含义过宽的 `graph_item`

建议边类型继续为：

- `sequential`
- `query_overlap`
- `entity_overlap`
- `bridge`

该图由 heuristic evidence graph builder 构建，只用于：

- Dense R-GCN；
- Dense-FT R-GCN；
- evidence 数据集的训练 pair 构建与现有结构评价。

GraphRAG 不读取它，Execution-Provenance Retriever 也不读取它。

### 4.2 EntityKnowledgeGraph：GraphRAG 方法内部索引

GraphRAG 内部图包含：

- entity：规范化名称、aliases、类型、描述、关联 candidate IDs；
- relation：source/target entity、描述、关联 candidate IDs、权重。

它由 candidates 的文本构建，可在 retrieval artifact/cache 中持久化，但其所有权始终属于 GraphRAG。它不是 `EvidenceGraph`，也不是 `ExecutionProvenanceGraph`。

### 4.3 ExecutionProvenanceGraph：执行与数据流专用

领域 contract 支持论文完整方向，但不会要求每个数据集伪造缺失事件。

第一版核心节点类型：

- `Task`
- `Agent`
- `ToolCall`
- `ToolOutput`
- `Answer`

来源真实提供时允许的扩展节点类型：

- `Observation`
- `Evidence`
- `Claim`
- `Verification`
- `Decision`

关键规则：数据源若没有 Claim 或 Verification，就不创建这些节点；领域层不得调用 LLM、NLI 或规则生成器伪造它们。

第一版核心边类型及方向：

| Edge | 方向 | 语义 |
|---|---|---|
| `contains` | Task -> execution node | 节点属于该任务；主要用于作用域，不作为高权重推理边 |
| `invokes` | Agent -> ToolCall | agent 发起工具调用 |
| `returns` | ToolCall -> ToolOutput | 工具调用产生实际输出 |
| `feeds` | ToolOutput -> downstream ToolCall | 上游输出被绑定到下游工具参数 |
| `grounds` | ToolOutput/Evidence -> Answer/Claim | 来源明确标注该输出支撑下游结论 |
| `precedes` | execution node -> later node | 只有时间顺序，没有足够证据声明数据依赖 |

revision 数据真实存在时允许：

- `contradicts`
- `invalidates`
- `affects`
- `supersedes`

Claim/Verification 数据真实存在时允许：

- `supports`
- `verifies`
- `depends_on`

`feeds` edge 必须允许携带 field binding metadata，例如：

```text
output_field
input_parameter
binding_value_hash
binding_kind
```

只有明确的 `param_for_next_tool`、参数引用或等价 source signal 才能生成 `feeds`。单纯因为两个 tool step 相邻，只能生成低语义强度的 `precedes`，不能冒充数据依赖。

## 5. GraphRAG 的最终行为

### 5.1 一个方法，不是两个 seed 变体

删除公开方法：

- `bm25_graph_rerank`
- `dense_graph_rerank`

新增公开方法：

- `graphrag`

`graphrag` 不是当前 graph rerank 的改名。当前 graph rerank 在 `EvidenceGraph` 上做 heuristic neighbor propagation；目标 GraphRAG 在 entity/relation graph 上执行 entity search。

### 5.2 参考旧 FastGraphRAG，但不直接合并旧分支

实现参考 `feature/fast-graph-rag` 中已经验证过的设计：

1. 对 candidates 做确定性的 entity extraction、normalization 与 alias catalog；
2. 根据 entity co-occurrence 构建 relation，并保留 entity/relation 到 candidate IDs 的映射；
3. 使用 query entity linking、lexical match 和 dense entity similarity 形成 seed；
4. 在 entity graph 上执行 Personalized PageRank；
5. 将 entity/relation 分数投影回 candidate 分数；
6. dense candidate score 只作为缺少 entity signal 时的 fallback 或受控混合项。

旧分支基于较早架构，不能 cherry-pick 或 merge 后再修补；应按当前 request-first、registry 和 workflow 结构重新移植核心算法。

第一版仍采用非 LLM entity extraction，保证实验可复现和成本可控。论文和报告中应准确称为“参考 FastGraphRAG 的非 LLM entity-search GraphRAG baseline”，不能声称复现完整的 LLM community-summary GraphRAG pipeline。

### 5.3 GraphRAG 输出

GraphRAG 必须返回统一的 ranked candidate IDs。它可以额外返回实际参与搜索的 entity/relation trace，但不能把 entity edge 伪装成 evidence dependency edge 或 execution provenance edge。

## 6. R-GCN 去 beam：恢复哪一种行为

“恢复原本的 R-GCN”以 commit `036d12d` 的直接父版本行为为准，但不能简单执行整提交回滚，因为后续 request-first、batching、device propagation、registry 和 checkpoint 修复仍需保留。

最终模型行为固定为：

1. frozen text embedding 与 numeric node features 进入 input projection；
2. 经过多层 typed R-GCN message passing；
3. node scorer 为每个 evidence item 独立输出一个 logit；
4. 训练使用正负 node pair 的 binary cross-entropy；
5. 推理时一次前向计算所有 candidate logits；
6. 按 logit 降序排序并直接取 top-k；
7. retrieved subgraph 仅由 top-k evidence nodes 在 `EvidenceGraph` 上诱导得到。

必须删除的 beam 行为：

- beam decoder 与 beam hypothesis；
- autoregressive select/stop action；
- frontier-conditioned scoring；
- dynamic oracle；
- beam loss、stop loss、coverage/frontier auxiliary loss；
- training/inference beam size、max steps、length penalty、deduplication 等配置；
- beam 专用 metric、trace 字段、测试和文档。

保留：

- R-GCN typed message passing；
- Dense 与 Dense-FT seed features；
- hard-negative pair sampling；
- 原有 R-GCN ablations，例如 relation typing、edge type、graph encoder 等；
- 两个独立公开方法 ID。

去 beam 后 checkpoint model/config schema 必须更新。当前 beam checkpoint 不提供兼容加载路径，两个 R-GCN 都重新训练；不要为历史 checkpoint 增加双模型分支。

两个方法的差别只在 seed/encoder 来源：

- `dense_rgcn_graph_retriever`：使用基础 Dense；
- `dense_ft_rgcn_graph_retriever`：使用对应 Dense-FT 模型，并保留 Dense-FT training dependency。

## 7. Ours：Execution-Provenance Retriever

### 7.1 第一版定位

第一版 `execution_provenance_retriever` 是不可训练的 typed path retriever。它实现论文的核心结论：semantic relevance 负责找 seed，显式 execution/dataflow provenance 负责恢复完整路径。

它不做以下事情：

- 不从 flat passages 自动抽取 provenance graph；
- 不把 chronological adjacency 全部当作 dependency；
- 不为缺失的 Claim/Verification 运行 LLM 或 NLI；
- 不复用 evidence R-GCN；
- 不在第一版增加新的 trainable GNN。

### 7.2 检索流程

第一步，semantic seed selection：

- 对 `ExecutionProvenanceRankingRequest.candidates` 计算 query-node dense similarity；
- 选择 top-s 节点作为 seeds；
- ToolOutput 和有文本结果的 ToolCall 是主要 seeds，Task/Agent 默认不参与 seed competition。

第二步，typed provenance expansion：

- 从 seeds 沿 `returns`、`feeds`、`grounds`、`supports`、`depends_on` 等允许的 dependency edge 双向恢复上下游；
- `contains` 和 `precedes` 默认不作为强 dependency，只能提供低权重上下文或 tie-break；
- 扩展遵守 node/edge type 合法转移，例如 `ToolCall -> ToolOutput -> ToolCall`；
- `feeds` 的 field binding match 比纯 adjacency 获得更高权重；
- 遇到 `invalidates`、`supersedes` 或 invalidated lifecycle state 时施加惩罚或停止把旧节点当作有效支持。

第三步，path scoring。以论文公式为基础，第一版使用：

```text
score(path, query)
  = alpha * semantic_relevance
  + beta  * binding_consistency
  + gamma * provenance_completeness
  + eta   * explicit_grounding
  - delta * path_length
  - zeta  * invalidation_penalty
```

各项语义：

- `semantic_relevance`：path 中 seed/节点与 query 的相关度；
- `binding_consistency`：`feeds` 是否具有明确 output-field -> input-parameter 绑定；
- `provenance_completeness`：是否保留完整 ToolCall -> ToolOutput 与跨工具依赖段；
- `explicit_grounding`：仅当来源提供 grounds/support edge 时生效；
- `path_length`：抑制无关扩展；
- `invalidation_penalty`：revision 信息存在时抑制失效路径。

第四步，选择 top paths，将 path 分数投影到 candidate node 分数。最终返回：

- 完整 ranked candidate IDs；
- 实际选中的 provenance paths；
- 实际遍历并用于打分的 typed edges。

不能像普通 induced subgraph 一样，把 top-k 节点之间所有碰巧存在的 edge 都声称为检索路径。

### 7.3 与 GraphRAG 的核心差别

| GraphRAG | Execution-Provenance Retriever |
|---|---|
| 从文本抽取 entity/relation | 读取数据集原生 execution/dataflow events |
| entity co-occurrence / semantic relation | Agent、ToolCall、ToolOutput 与参数数据流 |
| PPR 扩散 entity relevance | 沿 typed dependency path 受约束扩展 |
| 投影回相关 candidate | 恢复完整、可审计的执行证据路径 |

二者都可用于两类 benchmark 的对比，但它们使用的 graph 语义不可互换。

### 7.4 可训练版本的处理

本计划不实现 trainable provenance retriever。若不可训练版本证明 typed provenance 有增益，再单独提出方法，例如 `trainable_execution_provenance_retriever`：

- frozen dense node/query embeddings；
- edge-type embedding；
- 轻量 next-edge 或 path scorer；
- trajectory/template/tool-chain 隔离的数据划分；
- 不复用现有 evidence R-GCN checkpoint 或 relation vocabulary。

小规模 synthetic benchmark 的模板重复风险使得直接训练新的重型 GNN 很容易学到固定路径模板，而不是泛化的 provenance dependency，因此必须通过隔离划分、结构平衡和独立测试约束可训练版本。

## 8. Registry 的目标抽象

### 8.1 MethodDefinition 不再靠 lifecycle 猜输入与指标

当前 `GraphInputSource.NONE/GRAPH_ARTIFACT` 删除，替换为显式 method metadata：

```python
@dataclass(frozen=True)
class MethodInputSpec:
    request_type: type[object]
    required_artifact: RequiredArtifact
    supported_families: frozenset[RetrievalTaskFamily]


class RequiredArtifact(StrEnum):
    NONE = "none"
    EVIDENCE_GRAPH = "evidence_graph"


class RetrievalTaskFamily(StrEnum):
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    EXECUTION_PROVENANCE = "execution_provenance"
```

说明：

- `ExecutionProvenanceGraph` 已经是 `ExecutionProvenanceRankingRequest` 的一部分，不经过通用 workflow graph artifact 参数。
- `EntityKnowledgeGraph` 由 GraphRAG assembler 构建，也不声明为 dataset graph artifact。
- 只有 evidence R-GCN 需要预构建的 `EvidenceGraph` artifact。

MethodDefinition 还应显式声明 capability，而不是通过“是否需要 graph artifact + lifecycle”推断：

```python
@dataclass(frozen=True)
class RetrievalCapabilities:
    produces_ranked_nodes: bool
    produces_native_edge_trace: bool
    trainable: bool
```

删除或重写当前 `supports_path_metrics()`。某个指标能否计算，应由 dataset evaluation labels 决定；method capability 只说明它是否原生输出自己实际遍历的 edge trace。

### 8.2 Registry 方法表

| Method ID | Request | Families | 预构建 graph artifact | Trainable |
|---|---|---|---|---|
| `bm25` | TextRankingRequest | evidence、provenance | none | no |
| `dense` | TextRankingRequest | evidence、provenance | none | no |
| `dense_ft` | TextRankingRequest | evidence | none | yes：encoder |
| `graphrag` | GraphRAGRequest | evidence、provenance | none：内部 entity graph | no |
| `dense_rgcn_graph_retriever` | EvidenceGraphRankingRequest | evidence | EvidenceGraph | yes |
| `dense_ft_rgcn_graph_retriever` | EvidenceGraphRankingRequest | evidence | EvidenceGraph | yes |
| `execution_provenance_retriever` | ExecutionProvenanceRankingRequest | provenance | none：request 原生携带 | no |

Registry 最终只能包含上表 7 个唯一公开方法 ID；不得为 Memory Stream 保留 tombstone、隐藏 entry 或兼容 alias。

### 8.3 Build payload 也必须有具体语义

删除含义模糊的：

- `GraphRerankBuildPayload`
- `CheckpointGraphBuildPayload`
- `run_retrieve_stage(..., graphs: list[MemoryGraph] | None)`

改为具体 payload：

- `FlatRetrievalBuildPayload(text_requests, ...)`
- `GraphRAGBuildPayload(text_requests, ...)`
- `EvidenceRgcnBuildPayload(text_requests, evidence_graphs, ...)`
- `ExecutionProvenanceBuildPayload(provenance_requests, ...)`

Registry builder 必须校验 concrete payload 与 method definition 的 `request_type`/family/artifact 一致。禁止把错误 graph 类型传进 builder 后再靠 retriever 报错。

### 8.4 Memory Stream 删除边界

本次不是 deprecate，而是删除。至少覆盖：

- 删除 `graph_memory/retrieval/methods/memory_stream/` 与所有 tuning 实现；
- 删除 `TemporalMemoryRankingRequest`、recency/importance 专用 request settings、payload 与校验器；
- 删除 Registry enum、method definition、builder、selector 和 capability 分支；
- 删除 dataset projector 中只服务 temporal-memory request 的投影路径；
- 删除 experiment planning、stage model、retrieve stage、status/reporting 中的专用分派；
- 删除 `scripts/tune_memory_stream.py` 及相关 CLI 命令；
- 删除 `configs/method_configs/memory_stream.yaml`、Memory Stream dataset/experiment config 与默认配置引用；
- 删除专用行为测试、活动计划、配置说明和未完成的 `openspec/changes/add-memory-stream-retrieval`；
- 只有仍被非 Memory Stream 调用者真实使用的通用代码可以保留，并必须改成不带 temporal/memory-stream 语义的名字。

禁止用以下方式假装删除：保留未注册类、兼容 import、deprecated config key、空 builder、旧方法到 Dense 的 alias、对旧配置静默降级、读取旧 tuning artifact。旧配置必须在 method/config validation 边界明确失败。

## 9. Workflow 与 graph build 的最终行为

`build_graphs` 不能全局删除，因为两个 evidence R-GCN 仍依赖它；但它必须更名并降级为 evidence-only stage：

```text
build_graphs.py              -> build_evidence_graphs.py
GraphBuildRequest            -> EvidenceGraphBuildRequest
MemoryGraph                  -> EvidenceGraph
graph artifact path/name     -> evidence graph artifact path/name
```

Planner 只在以下情况调度 evidence graph construction：

- 选择 Dense R-GCN 或 Dense-FT R-GCN；
- 训练 pair/R-GCN 训练需要；
- 某个 evidence-only evaluation 明确需要该 artifact。

Planner 不因选择 GraphRAG 而调度 evidence graph stage。GraphRAG 的 entity index 在其 builder/method 内构建或加载缓存。

未来 execution provenance workflow 直接加载 dataset adapter 产生的 `ExecutionProvenanceRankingRequest`，不经过 `build_evidence_graphs.py`。

训练阶段：

- Dense-FT 只服务 evidence profile 中的 Dense-FT 与 Dense-FT R-GCN；
- pair building 与 R-GCN training 只接受 EvidenceGraph；
- Execution-Provenance Retriever 第一版无 train stage；
- GraphRAG 第一版无 train stage，也不再复用 graph-rerank tuning stage。

当前 `graph_rerank` tuning、search space 和 lifecycle 在 GraphRAG 移植完成后删除。

## 10. 输出与评价边界

所有检索方法都必须返回完整的 ranked candidate list，保证现有 top-k evidence retrieval 指标仍可统一计算。

`RetrievalTrace` 只记录方法真实使用的信息：

- BM25/Dense/Dense-FT：无原生 edge trace；
- GraphRAG：可记录 entity/relation search trace，但类型必须与 evidence/provenance edge 分开；
- R-GCN：可记录 top-k nodes 在 EvidenceGraph 上的 induced edges；
- Execution-Provenance Retriever：记录实际选中 path 和实际 traversed provenance edges。

评价层可以使用 gold labels/graph 判断某个 ranked node set 覆盖了多少 gold path，但 gold graph 绝不能进入 BM25、Dense 或 GraphRAG 的检索输入。这样 path metric 的计算不会变成测试时答案泄漏。

Execution provenance 指标与论文保持结论一致：优先验证显式 provenance 是否提升 path recovery，而不是强迫数据源提供它没有的 Claim/Verification 标签。Claim support、contradiction、invalidation、impact tracing 只有在来源或后续 revision augmentation 提供相应 gold labels 时才启用。

## 11. 分阶段实施顺序

### Phase 0：审查并锁定本文档

- 确认方法矩阵、request 类型、graph 所有权和 ours 第一版算法。
- 本阶段不修改代码。

### Phase 1：彻底删除 Memory Stream

- 删除方法、请求、Registry、配置、调参、工作流、CLI、dataset temporal projector、测试和活动 OpenSpec change。
- 删除 `TEMPORAL_MEMORY` family、importance/recency 专用 contract 与 artifact。
- 旧配置直接报 unsupported method，不提供兼容路径。
- 扫描当前活动代码与配置，确保 `memory_stream`、`MemoryStream`、`TemporalMemoryRankingRequest`、`tune_memory_stream` 为零命中。

### Phase 2：去除 beam，恢复原始 R-GCN node scoring

- 恢复独立 node logit + BCE 训练与一次性 top-k inference。
- 删除 beam decoder/oracle/loss/config/metrics/tests/docs。
- 更新 checkpoint schema，明确要求重训。
- 保留后续已经修复的 batching、device、request-first 与 registry 行为。

### Phase 3：EvidenceGraph 命名与 Registry 输入语义化

- `MemoryGraph` -> `EvidenceGraph`。
- `GraphRankingRequest` -> `EvidenceGraphRankingRequest`。
- `GraphBuildRequest` -> `EvidenceGraphBuildRequest`。
- 引入 method request type、task family、required artifact、native trace capability。
- 将通用 graph payload 改为 evidence-specific payload。
- 对现有 HotpotQA/2Wiki/MuSiQue projector 只做机械迁移，不改变数据语义。

### Phase 4：用 GraphRAG 替换两个 graph rerank

- 按当前主分支架构移植旧 FastGraphRAG 的 entity extraction、index、PPR 与 candidate projection。
- 新增 `GraphRAGRequest`、`GraphRAGBuildPayload`、`graphrag` registry entry/config。
- 删除 `bm25_graph_rerank`、`dense_graph_rerank`、graph-rerank tuning/config/search space。
- 更新传统 evidence profile 为 6 个方法。

### Phase 5：Execution provenance domain 与 ours retriever

- 新增 `ExecutionProvenanceGraph` contract。
- 新增 `ExecutionProvenanceRankingRequest`。
- 实现不可训练的 semantic seed + typed provenance expansion + path scoring。
- 新增 registry/build payload/config 与 domain-level programmatic tests。
- 不在领域包内新增具体 dataset projector。

### Phase 6：通过独立 change 适配 execution-provenance benchmark

- 使用独立 dataset ID，避免改变标准 evidence dataset 的语义。
- 从可恢复的 gold evidence chain 离线构造正确分支，再加入结构匹配的错误分支。
- 保持 ranking/label 分离并审计拓扑泄漏、划分、过滤计数与结构统计。
- 启用 execution provenance 的 5-method profile 与对应评价。

## 12. 预期文件所有权

建议目标结构：

```text
graph_memory/
  retrieval/
    requests/
      text.py
      evidence_graph.py
      graphrag.py
      execution_provenance.py
      __init__.py
    methods/
      flat/
      graphrag/
      execution_provenance/
      trainable_graph.py
  graphs/
    evidence/
      contracts.py
      requests.py
      construction/
    provenance/
      contracts.py
      validation.py
  registry/
    methods.py
    retrieval.py
    retrieval_builders.py
```

边界要求：

- `retrieval/requests/*` 拥有 retriever consumer contracts；
- `graphs/evidence/*` 只拥有传统 evidence graph；
- `graphs/provenance/*` 只拥有 execution provenance graph；
- `methods/graphrag/*` 拥有 EntityKnowledgeGraph 的构建与搜索；
- dataset adapter 只拥有 raw record parsing 与到 consumer request 的 projection；
- registry 只做方法元数据、依赖声明、精确 builder 路由和兼容性校验，不解释 dataset record。

是否把当前单文件 `retrieval/requests.py` 拆成 package 可在实现时一次完成；不允许为了渐进迁移长期同时保留同名旧/新 request 兼容层。

## 13. 验收条件

领域重构完成必须满足：

1. Registry 中传统 evidence profile 恰好包含 6 个目标公开方法。
2. execution provenance profile 恰好包含 4 个目标公开方法。
3. Registry 能在执行前拒绝 `ExecutionProvenanceRankingRequest -> R-GCN`。
4. Registry 能在执行前拒绝 `EvidenceGraphRankingRequest -> Execution-Provenance Retriever`。
5. GraphRAG 不依赖 EvidenceGraph artifact，也不读取 ExecutionProvenanceGraph 的 typed edges。
6. 两个 R-GCN 都只接受 EvidenceGraphRankingRequest。
7. R-GCN 代码、配置、checkpoint 和 trace 中不存在 beam/stop/frontier/oracle 行为。
8. Execution-Provenance Retriever 只接受原生 ExecutionProvenanceRankingRequest。
9. 缺少 Claim/Verification 的 trajectory 可以合法运行，且系统不会合成这些节点。
10. `feeds` 与 `precedes` 被严格区分；相邻 step 不自动升级为 data dependency。
11. BM25/Dense/GraphRAG 在两类 dataset 上共享相同领域实现，不复制 dataset-specific retriever。
12. 现有 HotpotQA、2Wiki、MuSiQue 行为除方法集合与 R-GCN 去 beam 外保持一致。
13. 当前活动代码与配置扫描对 `memory_stream`、`MemoryStream`、`TemporalMemoryRankingRequest`、`tune_memory_stream` 为零命中。
14. 旧 Memory Stream 配置明确失败，不存在 alias、deprecated、静默迁移或旧 artifact 兼容加载。
15. 全量测试、ruff、basedpyright、compileall、OpenSpec strict validation 与最小真实 workflow smoke 全部通过。

## 14. 明确不在本计划中做的事情

- 不在 domain 层生成或清洗 benchmark 数据。
- 不让 execution-provenance dataset 使用 evidence R-GCN。
- 不把 2Wiki/HotpotQA/MuSiQue 人工伪装成 tool trajectory。
- 不引入 Claim extractor、LLM verifier 或 NLI support classifier。
- 不把 revision cases 当作第一版 Execution-Provenance Retriever 的前置条件。
- 不实现 trainable execution provenance model。
- 不直接复用或兼容 beam R-GCN checkpoint。
- 不把 GraphRAG entity edge 用作 provenance gold edge。
- 不保留任何 Memory Stream 兼容代码、配置或活动 OpenSpec 工件。

## 15. 审查时应重点确认的四个决定

1. 第一版 provenance schema 是否接受“核心五类节点 + 来源提供时才启用 Claim/Verification 等扩展”。是的
2. 第一版 ours 是否确认只做不可训练 typed path retriever。是的
3. R-GCN 是否确认以 `036d12d^` 的 node-wise BCE/top-k 行为为目标，并强制重训。可以
4. GraphRAG 是否确认使用旧 FastGraphRAG 的非 LLM entity search 设计，但只保留一个 `graphrag` 方法 ID。可以
