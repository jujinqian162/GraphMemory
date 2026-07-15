## ADDED Requirements

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。

### Requirement: Single GraphRAG method
系统 SHALL 公开唯一 `graphrag` method，并 MUST NOT 将旧 `bm25_graph_rerank` 或 `dense_graph_rerank` 作为 alias、seed variant 或隐藏方法保留。

#### Scenario: Resolve GraphRAG method
- **WHEN** Registry 解析 GraphRAG baseline
- **THEN** 只解析到 `graphrag`

### Requirement: Deterministic non-LLM entity index
GraphRAG SHALL 对 candidate text 执行确定性的 entity extraction、normalization 和 alias catalog 构建，并根据 entity co-occurrence 建立方法内部 relation graph；该流程 MUST NOT 调用 LLM。

#### Scenario: Repeat index construction
- **WHEN** 相同 request 和配置被构建两次
- **THEN** entity catalog、relations 与 candidate mappings 保持一致

### Requirement: Entity search and projection
GraphRAG SHALL 使用 query entity linking、lexical match 和 dense entity similarity 构造 seeds，在 entity graph 上执行 Personalized PageRank，并将 entity/relation score 投影回完整 candidate ranking。

#### Scenario: Entity-connected evidence
- **WHEN** query-linked entity 通过 relation graph 连接到另一个 candidate entity
- **THEN** PPR propagation 可以提升该 candidate 的投影分数

#### Scenario: No entity signal
- **WHEN** request 无可用 entity extraction/linking signal
- **THEN** GraphRAG 使用受控 dense fallback 并仍返回完整 candidate ranking

### Requirement: GraphRAG graph isolation
GraphRAG MUST NOT 读取 EvidenceGraph artifact 或 ExecutionProvenanceGraph typed edges；可选 trace MUST 使用独立 entity/relation trace 类型。

#### Scenario: Build without EvidenceGraph stage
- **WHEN** workflow 只选择 GraphRAG
- **THEN** 方法能够在没有 EvidenceGraph artifact 的情况下构建和检索
