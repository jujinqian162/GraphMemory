> 实施根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。执行每组任务前后均须以该 plan 核对方法矩阵、graph 所有权、无交叉 projection、Memory Stream 全量删除和 R-GCN 去 beam 要求。

## 1. Baseline and contract guards

- [x] 1.1 盘点当前 request、graph、Registry、R-GCN beam、graph rerank、Memory Stream、workflow/config/test 所有活动引用并记录迁移清单
- [x] 1.2 新增失败优先的 request/Registry/方法矩阵行为测试，锁定七个 method ID 和错误交叉输入拒绝
- [x] 1.3 新增失败优先的 R-GCN node-wise、GraphRAG 无 EvidenceGraph、provenance feeds/precedes 与真实 trace 行为测试

## 2. Remove Memory Stream completely

- [x] 2.1 删除 Memory Stream 方法包、tuning 模块、TemporalMemory request 与 recency/importance 专用 contracts
- [x] 2.2 删除 Memory Stream Registry/build payload/builder、dataset projector 和 experiment/stage/validation 分支
- [x] 2.3 删除 Memory Stream CLI、配置、专用 tests/docs 以及未完成的 `add-memory-stream-retrieval` OpenSpec change
- [x] 2.4 验证旧 `memory_stream` 配置明确 unsupported，且活动代码/配置目标 symbol 扫描为零命中

## 3. Restore node-wise R-GCN

- [x] 3.1 将 R-GCN model/training 恢复为独立 node logits 与 BCE，保留 typed message passing、batching 和 device 路径
- [x] 3.2 将 R-GCN retrieval 恢复为一次前向完整 ranking/top-k，并删除 beam decoder/oracle/frontier/stop/trace 实现
- [x] 3.3 删除 beam config、metrics、ablation 和测试分支，更新 node-wise checkpoint schema 并拒绝旧 beam checkpoint
- [x] 3.4 验证两个 R-GCN 的最小训练、checkpoint round-trip 和 retrieval 行为

## 4. Split graph and request domain contracts

- [x] 4.1 将 `MemoryGraph` 及 construction/serialization API 机械迁移为 `EvidenceGraph`
- [x] 4.2 将 `GraphBuildRequest`/`GraphRankingRequest` 迁移为 evidence-specific request，并拆分 concrete retrieval request package
- [x] 4.3 新增 ExecutionProvenance node/edge/graph value objects、binding metadata 与 validation
- [x] 4.4 新增 GraphRAGRequest 与 ExecutionProvenanceRankingRequest，删除 generic/temporal compatibility exports
- [x] 4.5 机械迁移 HotpotQA、2Wiki、MuSiQue projectors 和 R-GCN consumers，不改变 evidence 数据语义

## 5. Semantic Registry and payload routing

- [x] 5.1 新增 task family、required artifact、method input spec 与 retrieval capability metadata
- [x] 5.2 将 Registry 收敛为七个公开 method ID，删除旧 graph rerank/Memory Stream definitions 与 generic GraphInputSource
- [x] 5.3 实现 flat、GraphRAG、Evidence R-GCN 与 ExecutionProvenance concrete build payloads
- [x] 5.4 在 builder 入口实现 request/family/artifact compatibility validation 和清晰错误

## 6. Implement entity-search GraphRAG

- [x] 6.1 按当前 request-first 架构移植确定性 entity extraction/normalization、alias catalog 与 co-occurrence index
- [x] 6.2 实现 query entity linking、lexical/dense seeds、Personalized PageRank 和 candidate score projection/fallback
- [x] 6.3 实现独立 entity trace，注册唯一 `graphrag` builder/config 并删除旧 graph-rerank tuning/search space
- [x] 6.4 验证 GraphRAG 确定性、entity propagation、fallback、完整 ranking 及不依赖 EvidenceGraph artifact

## 7. Implement Execution-Provenance Retriever

- [x] 7.1 实现 provenance dense semantic seed selection，并限制默认 seed node types
- [x] 7.2 实现 typed dependency transition、binding-aware expansion、chronology low-weight handling 与 revision invalidation
- [x] 7.3 实现 path scoring、candidate projection、完整 ranking 以及 actual paths/traversed edges trace
- [x] 7.4 注册 `execution_provenance_retriever` builder/config 并验证缺少 Claim/Verification 的合法运行

## 8. Workflow, configs, and documentation

- [x] 8.1 将 graph artifact/stage 命名迁移为 evidence-specific，并让 planner 只在 R-GCN 依赖存在时构建 EvidenceGraph
- [x] 8.2 将 HotpotQA、2Wiki、MuSiQue 正式 profile 更新为六方法矩阵并移除废弃方法/config
- [x] 8.3 更新 CLI、operations/design/config docs 与示例，确保活动文档持续指向本 plan 且无 Memory Stream/beam/旧 graph-rerank 可用性声明
- [x] 8.4 验证未来 execution-provenance family 的四方法 capability matrix，不新增具体 dataset adapter

## 9. Final verification

- [x] 9.1 运行相关 targeted pytest 并修复所有行为回归
- [x] 9.2 运行全量 pytest、ruff、basedpyright 与 compileall
- [x] 9.3 运行最小真实 evidence workflow smoke、OpenSpec strict validation、symbol scan 与 git diff check
