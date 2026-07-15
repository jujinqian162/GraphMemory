## Why

当前领域层把 evidence graph、entity-search graph 和 execution-provenance graph 混成了同一种“graph retrieval”，导致 Request、Registry、workflow 和指标能力无法表达真实输入语义。与此同时，低收益的 beam R-GCN 与已经退出研究矩阵的 Memory Stream 持续增加训练时间、配置分支和兼容负担，因此需要一次 current-only 的领域重构。

本 change 的根约束文档是 [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。Proposal、design、specs、tasks 与实现必须持续引用并遵守该 plan；若实现细节需要调整，必须先同步更新 OpenSpec 工件，不能绕开 plan 中锁定的方法矩阵、graph 所有权、无交叉 projection、Memory Stream 全量删除与 R-GCN 去 beam 要求。

## What Changes

- **BREAKING**：彻底删除 Memory Stream 的方法、请求、Registry、配置、调参、工作流、CLI、测试与活动 OpenSpec 工件；旧配置直接报 unsupported method，不提供 alias、deprecated 或 artifact 兼容加载。
- **BREAKING**：把传统 `MemoryGraph`/`GraphRankingRequest` 明确迁移为 `EvidenceGraph`/`EvidenceGraphRankingRequest`，并禁止通用 graph request 或运行时 `graph_kind` 分派。
- **BREAKING**：移除 R-GCN beam decoder、stop/frontier/oracle、beam loss/config/trace，恢复独立 node logit、BCE 训练和一次前向 top-k；旧 beam checkpoint 不兼容并要求重训。
- **BREAKING**：删除 `bm25_graph_rerank` 与 `dense_graph_rerank`，新增唯一 `graphrag` baseline；其内部构建非 LLM entity/relation index 并通过 PPR 投影回候选。
- 新增 `ExecutionProvenanceGraph`、`ExecutionProvenanceRankingRequest` 和不可训练的 `execution_provenance_retriever`，以 semantic seeds、typed dependency expansion 与 path scoring 返回 ranked candidates 和真实 traversed paths。
- Registry 显式声明 request type、task family、required artifact 与 native trace capability，并在执行前拒绝 evidence/provenance 两类专用方法的错误交叉输入。
- Workflow 只为两个 evidence R-GCN 构建 EvidenceGraph；GraphRAG 和 execution-provenance retriever 不消费该工件。
- 传统 evidence profile 固定为 6 个方法；未来 execution-provenance profile 固定为 4 个方法。本 change 不实现 TRAJECT-Bench adapter，也不把传统 QA 数据伪装成工具轨迹。

## Capabilities

### New Capabilities

- `typed-retrieval-requests`: 定义 Text、EvidenceGraph、GraphRAG 与 ExecutionProvenance 四种互不替代的请求及 graph contract。
- `semantic-method-registry`: 定义 7 个公开 method ID、两类 task family、具体 build payload、能力声明与预执行兼容性校验。
- `evidence-rgcn-retrieval`: 定义两个 evidence-only R-GCN 的 node-wise BCE/top-k 行为、EvidenceGraph 工件依赖与无 beam checkpoint 契约。
- `entity-search-graphrag`: 定义唯一、非 LLM 的 entity extraction/linking、entity graph PPR 和候选投影 GraphRAG baseline。
- `execution-provenance-retrieval`: 定义原生 execution/dataflow graph、typed path expansion、path scoring、invalidation handling 与真实 trace 输出。
- `retrieval-workflow-matrix`: 定义两类 benchmark 的方法矩阵、按需 EvidenceGraph 构建、训练依赖与 Memory Stream current-only 删除行为。

### Modified Capabilities

无。当前仓库没有已发布的 OpenSpec base specs；本 change 将现有代码行为收敛为以上新 capability contracts。

## Impact

主要影响 `graph_memory/retrieval/`、`graph_memory/graphs/`、`graph_memory/models/graph_retriever/`、`graph_memory/registry/`、`graph_memory/experiment/`、`graph_memory/stages/`、`graph_memory/datasets/`、`scripts/`、`configs/`、测试与活动文档。方法 ID、request 类型、graph artifact 名称、R-GCN checkpoint schema 和若干 CLI/config 都是有意的 breaking changes；不提供历史兼容层。
