## Context

当前实现以 `MemoryGraph`、`GraphRankingRequest` 和 `GraphInputSource` 表达所有 graph-backed retrieval，但实际已经存在三种不同所有权：dataset-derived evidence graph、GraphRAG 方法内部 entity graph、未来数据集原生 execution-provenance graph。当前两个 graph rerank 只是 evidence-neighbor propagation；R-GCN 又叠加了高成本 beam decoder；Memory Stream 还占据独立 request、配置、调参和 workflow 分支。

本设计的根约束是 [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。实施过程中每次遇到局部代码便利性与该 plan 冲突时，必须保持 plan 锁定的领域结论，并在需要时先修订本 change 工件，不能新增长期兼容层。

## Goals / Non-Goals

**Goals:**

- 用具体 request 与 graph 类型保持 dataset/retriever 解耦，同时让错误组合在 Registry/build 边界失败。
- 将公开方法收敛为 7 个唯一 ID，并形成 evidence 6-method 与 provenance 4-method 矩阵。
- 恢复可训练 R-GCN 的 node-wise BCE/top-k 行为，保留 batching、device 和 request-first 修复。
- 以单一、可复现的非 LLM entity-search GraphRAG 替换两个 heuristic graph rerank。
- 建立原生 ExecutionProvenanceGraph 和不可训练 typed-path retriever，不依赖具体 TRAJECT-Bench adapter。
- current-only 删除 Memory Stream 和旧 beam/graph-rerank 契约，不留 alias、deprecated 或 artifact compatibility。

**Non-Goals:**

- 不实现 TRAJECT-Bench parser/projector、revision augmentation 或新 benchmark。
- 不把 HotpotQA、2Wiki、MuSiQue 投影成 execution provenance。
- 不合成 Claim/Verification，不引入 LLM/NLI extractor。
- 不实现 trainable execution-provenance model。
- 不兼容旧 Memory Stream 配置/工件或 beam R-GCN checkpoint。

## Decisions

### 1. Request 使用封闭的具体 union，不使用通用 graph request

领域层定义 `TextRankingRequest`、`EvidenceGraphRankingRequest`、`GraphRAGRequest` 与 `ExecutionProvenanceRankingRequest`。共享字段可以由小型 value objects 复用，但不得出现 `graph: object`、`graph_kind` 或 retriever 内部类型分支。这样 dataset adapter 只负责生成消费者所需 request，Registry 可以在执行前校验准确类型。

备选方案是单一 generic request 加 graph discriminator；拒绝原因是它把 graph 解释责任重新推给 retriever，无法阻止 cross-projection。

### 2. EvidenceGraph 与 ExecutionProvenanceGraph 分别拥有 contract

EvidenceGraph 继续承载传统 question/evidence node 和 sequential/query/entity/bridge edges，作为两个 R-GCN 的预构建工件。ExecutionProvenanceGraph 使用 Task、Agent、ToolCall、ToolOutput、Answer 核心节点与 contains/invokes/returns/feeds/grounds/precedes 边；Claim/Verification/revision 节点和边仅在来源真实提供时出现。

`feeds` 必须具有显式 field binding evidence；相邻事件只能产生 `precedes`。GraphRAG entity graph 由方法包私有构建，不进入 dataset artifact contract。

### 3. Registry 直接描述语义输入与能力

`MethodDefinition` 持有 `MethodInputSpec(request_type, required_artifact, supported_families)` 和 `RetrievalCapabilities`。`RequiredArtifact` 只有 `NONE` 与 `EVIDENCE_GRAPH`；task family 只有 evidence 与 execution provenance。Builder 接收具体 payload 并立即核对 method definition，不再通过 lifecycle 或 `GraphInputSource` 猜测。

共享 BM25/Dense 在 provenance benchmark 上仍消费由 adapter 产生的 flat text request；GraphRAG 消费 GraphRAGRequest；专用 provenance retriever 只消费原生 provenance request。

### 4. R-GCN 回到 independent node scoring

保留 frozen dense features、numeric features、typed R-GCN message passing、hard-negative pair sampling、batch/device 修复和两个 encoder seed 变体。删除 beam decoder/oracle/frontier/stop action 及所有 beam loss/config/trace。训练对 node logits 使用 BCE；推理一次评分全部 evidence candidates 并 top-k。

不在 loader 中识别旧 beam checkpoint。checkpoint schema 随模型更新，旧 checkpoint 失败并要求重训。

### 5. GraphRAG 是单一 method-owned entity-search pipeline

从旧 `feature/fast-graph-rag` 手工移植确定性 entity extraction/normalization、alias catalog、co-occurrence relation index、query linking、lexical/dense seeds、PPR 和 candidate projection，不 merge/cherry-pick 旧分支。首版无 LLM 和 community summaries；dense score只作受控混合或 entity signal 缺失时 fallback。

GraphRAG 返回完整 candidate ranking，可附加与 evidence/provenance edges 类型隔离的 entity trace。Planner 不为它构建 EvidenceGraph。

### 6. Execution-Provenance Retriever 先做不可训练 typed path search

先以 dense query-node similarity 选 ToolOutput/有文本 ToolCall seeds，再沿合法的 returns/feeds/grounds/supports/depends_on 依赖转移双向扩展。contains/precedes 仅作低权上下文；invalidates/supersedes/lifecycle invalid state 施加惩罚或阻止旧支持生效。

path score 组合 semantic relevance、binding consistency、provenance completeness、explicit grounding、path length penalty 与 invalidation penalty。输出为完整 candidate ranking、实际选中 paths 和实际 traversed typed edges，不能以 induced subgraph 冒充检索轨迹。

### 7. Workflow 只按真实依赖调度 EvidenceGraph

EvidenceGraph stage 只在选择两个 R-GCN、构建训练 pairs 或训练 R-GCN 时调度。GraphRAG 和 execution-provenance retriever 不触发该 stage。Dense-FT training 只服务 evidence profile 的 Dense-FT 与 Dense-FT R-GCN。

### 8. Memory Stream 使用 current-only 删除

删除实现包、TemporalMemory request、recency/importance contract、Registry/builders、tuning/CLI、dataset temporal projection、config、tests 和未完成的 `add-memory-stream-retrieval` change。旧 method/config 不做迁移，必须在 validation 边界报 unsupported。只有有非 Memory Stream 调用者的真正通用代码可以保留，并去掉 temporal/memory-stream 命名。

## Risks / Trade-offs

- [大范围 breaking rename 可能产生遗漏] → 分阶段执行 repo-wide symbol scans，并让 type check/behavior tests 覆盖每个 public boundary。
- [GraphRAG 移植可能与旧分支漂移] → 只移植算法，不移植旧 wiring；添加 entity extraction、PPR、fallback 和无 EvidenceGraph 依赖的程序化测试。
- [R-GCN 去 beam 后旧实验不可直接续训] → 明确更新 checkpoint schema并重训；不以双 loader 增加长期复杂度。
- [Provenance path scoring 首版缺少真实 dataset 校准] → 保持不可训练、参数可配置且 contract 测试完备；真实数据适配与调参留到后续 change。
- [删除 Memory Stream 影响 LongMemEval 旧配置] → 这是有意的 current-only migration；旧配置明确失败，避免静默退化成 Dense。
- [一次性变更过大] → 任务按 deletion、R-GCN、domain/registry、GraphRAG、provenance、workflow/docs 六个可验证阶段推进，每阶段立即更新 tasks checkbox。

## Migration Plan

1. 建立失败优先 contract tests，并删除 Memory Stream 活动面和旧 change。
2. 去除 R-GCN beam 行为与 checkpoint schema，验证训练/推理最小闭环。
3. 机械迁移 EvidenceGraph/request 命名，建立 semantic registry/payload validation。
4. 移植单一 GraphRAG，删除旧 graph rerank/tuning/config。
5. 新增 provenance contracts 与 typed path retriever。
6. 调整 workflow/config/docs/tests，运行全量质量门与最小真实 evidence workflow smoke。

回滚以 Git commit/branch 为单位；运行时不提供兼容开关。

## Open Questions

无阻塞问题。TRAJECT-Bench 的原始字段到 request projection、provenance 指标启用条件及可训练版本均留给后续独立 OpenSpec change。
