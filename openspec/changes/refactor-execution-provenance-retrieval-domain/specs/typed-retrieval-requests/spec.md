## ADDED Requirements

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。

### Requirement: Concrete retrieval request types
系统 SHALL 提供 Text、EvidenceGraph、GraphRAG 与 ExecutionProvenance 四种具体 ranking request，且 SHALL NOT 提供带 `graph_kind` discriminator 的通用 graph request。

#### Scenario: Evidence R-GCN receives evidence graph request
- **WHEN** dataset projector 为传统 evidence retrieval 构造 R-GCN 输入
- **THEN** 输入类型为 `EvidenceGraphRankingRequest` 并携带对应 `EvidenceGraph`

#### Scenario: Provenance retriever receives native provenance request
- **WHEN** dataset adapter 为 execution-provenance retriever 构造输入
- **THEN** 输入类型为 `ExecutionProvenanceRankingRequest` 并携带原生 `ExecutionProvenanceGraph`

### Requirement: Separate graph ownership
系统 SHALL 将 `EvidenceGraph` 和 `ExecutionProvenanceGraph` 定义为不同 contract；GraphRAG entity graph MUST 由 GraphRAG 方法内部拥有且不得作为 dataset graph artifact 暴露。

#### Scenario: Graph type cannot be substituted
- **WHEN** 调用方将 ExecutionProvenanceGraph 放入 EvidenceGraph request
- **THEN** request 或 Registry validation 在 retriever 执行前拒绝该输入

### Requirement: Native provenance schema
ExecutionProvenanceGraph SHALL 支持 Task、Agent、ToolCall、ToolOutput、Answer 核心节点和 contains、invokes、returns、feeds、grounds、precedes 核心边，并 SHALL 允许来源真实提供的 Claim、Verification 与 revision 扩展。

#### Scenario: Missing claims remain valid
- **WHEN** 一条 trajectory 没有 Claim 或 Verification source records
- **THEN** graph validation 成功且系统不合成相应节点

#### Scenario: Binding and chronology stay distinct
- **WHEN** 两个 tool steps 仅在时间上相邻但没有参数引用或 field binding
- **THEN** adapter 只能建立 `precedes` 而不得建立 `feeds`

### Requirement: Request union has no temporal compatibility branch
公开 ranking request union MUST NOT 包含 `TemporalMemoryRankingRequest` 或任何 Memory Stream compatibility request。

#### Scenario: Legacy temporal request import
- **WHEN** 调用方尝试从当前 retrieval request API 导入 `TemporalMemoryRankingRequest`
- **THEN** 该 symbol 不存在且系统不提供兼容 re-export
