## ADDED Requirements

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。

### Requirement: Evidence method matrix
传统 evidence retrieval profile SHALL 恰好包含 BM25、Dense、Dense-FT、GraphRAG、Dense R-GCN 和 Dense-FT R-GCN 六个方法。

#### Scenario: Resolve evidence profile
- **WHEN** workflow 解析 HotpotQA、2Wiki 或 MuSiQue 的默认正式 methods
- **THEN** 方法集合与六个目标方法完全一致

### Requirement: Execution provenance method matrix
未来 execution-provenance profile SHALL 恰好包含 BM25、Dense、GraphRAG 和 Execution-Provenance Retriever 四个方法，且本 change SHALL NOT 添加具体 TRAJECT-Bench adapter。

#### Scenario: Validate provenance family support
- **WHEN** Registry 查询 execution-provenance family capabilities
- **THEN** 只返回四个目标方法且不包含 R-GCN、Dense-FT 或 Memory Stream

### Requirement: Conditional evidence graph workflow
Workflow SHALL 只在两个 evidence R-GCN、R-GCN training pairs/training 或明确 evidence-only evaluation 需要时构建 EvidenceGraph；GraphRAG 和 provenance retriever MUST NOT 触发该 stage。

#### Scenario: Flat and GraphRAG run
- **WHEN** run 只选择 BM25、Dense 和 GraphRAG
- **THEN** planner 不调度 EvidenceGraph construction

#### Scenario: R-GCN run
- **WHEN** run 选择任一 R-GCN
- **THEN** planner 调度 EvidenceGraph construction 并向 builder 传入 evidence-specific artifact

### Requirement: Memory Stream is completely removed
当前活动代码、配置、Registry、请求、调参、workflow、CLI、tests、docs 和 OpenSpec changes MUST NOT 提供 Memory Stream 能力或兼容路径。

#### Scenario: Legacy method config
- **WHEN** 用户加载包含 `memory_stream` 的旧配置
- **THEN** validation 明确报告 unsupported method，且不 alias 到 Dense、不静默忽略、不读取旧 artifact

#### Scenario: Active source scan
- **WHEN** 对当前活动代码和配置扫描 `memory_stream`、`MemoryStream`、`TemporalMemoryRankingRequest` 与 `tune_memory_stream`
- **THEN** 结果为零命中

### Requirement: Shared methods remain dataset agnostic
BM25、Dense 与 GraphRAG SHALL 在 evidence 和 provenance family 中共享同一领域实现；dataset adapter SHALL 只负责投影到对应 request，不复制 retriever。

#### Scenario: Add future provenance adapter
- **WHEN** 后续 adapter 生成 flat text 和 GraphRAG requests
- **THEN** 现有 shared method implementation 无需新增 dataset-specific branch 即可执行
