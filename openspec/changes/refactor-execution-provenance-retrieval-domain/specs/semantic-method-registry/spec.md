## ADDED Requirements

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。

### Requirement: Exact public method registry
Registry SHALL 只公开 `bm25`、`dense`、`dense_ft`、`graphrag`、`dense_rgcn_graph_retriever`、`dense_ft_rgcn_graph_retriever` 和 `execution_provenance_retriever` 七个 method ID。

#### Scenario: Enumerate public methods
- **WHEN** 调用方枚举当前 public retrieval methods
- **THEN** 结果与七个目标 ID 完全一致，且不包含 Memory Stream 或旧 graph rerank ID

### Requirement: Semantic input metadata
每个 `MethodDefinition` SHALL 显式声明 request type、supported task families、required artifact 与 ranked-node/native-trace/trainable capabilities，而 SHALL NOT 通过 lifecycle 或 generic graph source 推断输入语义。

#### Scenario: Inspect R-GCN definition
- **WHEN** 调用方检查任一 R-GCN method definition
- **THEN** definition 声明 EvidenceGraph request、evidence family 和 EvidenceGraph artifact

#### Scenario: Inspect GraphRAG definition
- **WHEN** 调用方检查 `graphrag`
- **THEN** definition 声明 GraphRAGRequest、两类 task family 和无预构建 graph artifact

### Requirement: Pre-execution compatibility validation
Registry builder MUST 在方法运行前验证 concrete payload、request type、task family 与 required artifact 的一致性。

#### Scenario: Provenance request sent to R-GCN
- **WHEN** execution-provenance request 被路由到任一 R-GCN builder
- **THEN** builder 在加载 checkpoint 或执行 retriever 前给出明确的不支持错误

#### Scenario: Evidence request sent to provenance retriever
- **WHEN** evidence graph request 被路由到 execution-provenance retriever
- **THEN** builder 在检索前给出明确的不支持错误

### Requirement: Concrete build payloads
系统 SHALL 使用 flat、GraphRAG、Evidence R-GCN 和 ExecutionProvenance 四类具体 build payload，且 MUST NOT 保留 generic graph 或 Memory Stream payload。

#### Scenario: Wrong payload class
- **WHEN** method builder 收到不符合 method definition 的 payload class
- **THEN** builder 明确拒绝而不是把错误对象传入 retriever
