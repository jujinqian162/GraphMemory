## ADDED Requirements

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。

### Requirement: Native provenance input only
`execution_provenance_retriever` SHALL 只接受原生 `ExecutionProvenanceRankingRequest`，并 MUST NOT 从 flat evidence passages 合成 provenance graph。

#### Scenario: Evidence dataset input
- **WHEN** 传统 evidence request 被提交给 provenance retriever
- **THEN** 系统在执行前拒绝且不尝试事实抽取或 tool-trajectory 合成

### Requirement: Semantic seed selection
Retriever SHALL 对可检索 execution nodes 计算 query-node dense similarity，并优先从 ToolOutput 与具有文本结果的 ToolCall 中选择 top-s seeds；Task/Agent 默认不参与 seed competition。

#### Scenario: Select execution seed
- **WHEN** ToolOutput 与 query 高度相关而 Agent label 仅匹配通用词
- **THEN** ToolOutput 可成为 seed，Agent 不因通用文本进入默认 seed 集

### Requirement: Typed dependency expansion
Retriever SHALL 沿合法的 returns、feeds、grounds、supports、depends_on dependency transitions 恢复上下游；contains 和 precedes MUST NOT 默认等价于强依赖。

#### Scenario: Follow field binding
- **WHEN** ToolOutput 通过带 output-field/input-parameter metadata 的 `feeds` 连接下游 ToolCall
- **THEN** expansion 将该边作为高置信 data dependency

#### Scenario: Chronology only
- **WHEN** 两节点只存在 `precedes`
- **THEN** expansion 最多将其作为低权上下文，不把它计为完整 dependency segment

### Requirement: Provenance path scoring
Retriever SHALL 组合 semantic relevance、binding consistency、provenance completeness、explicit grounding、path length penalty 和 invalidation penalty 对 paths 打分，并将 top path score 投影到 candidate ranking。

#### Scenario: Prefer bound complete path
- **WHEN** 两条语义分数相近的路径中只有一条具有完整 ToolCall→ToolOutput→feeds→ToolCall 绑定
- **THEN** 完整绑定路径获得更高 provenance score

#### Scenario: Invalidation information exists
- **WHEN** 一条候选路径包含被 invalidates 或 supersedes 标记为失效的支持节点
- **THEN** retriever 惩罚该路径或停止把旧节点作为有效支持

### Requirement: Trace records actual search
Retriever SHALL 返回完整 ranked candidate list，并可返回实际选中的 provenance paths 与实际用于打分的 traversed typed edges；MUST NOT 用 top-k induced subgraph 冒充检索路径。

#### Scenario: Untraversed edge among selected nodes
- **WHEN** top-k nodes 之间存在一条搜索过程未遍历的 edge
- **THEN** 该 edge 不出现在 native provenance trace 中
