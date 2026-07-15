## ADDED Requirements

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。

### Requirement: Independent evidence node scoring
两个 R-GCN 方法 SHALL 使用 frozen text/numeric features、typed R-GCN message passing 和独立 node scorer，为全部 evidence candidates 一次性产生 logits，并按 logit 返回 top-k。

#### Scenario: R-GCN inference
- **WHEN** R-GCN 对一个 EvidenceGraphRankingRequest 推理
- **THEN** 每个 candidate 获得一个独立 logit，完整 ranking 由单次前向结果降序产生

### Requirement: Node-wise BCE training
R-GCN 训练 SHALL 对正负 node pairs 使用 binary cross-entropy，并保留 Dense 与 Dense-FT 两种 seed/encoder 来源。

#### Scenario: Training batch loss
- **WHEN** trainer 接收含正负 evidence nodes 的 batch
- **THEN** loss 直接由 node logits 与 binary labels 计算，不调用 sequence decoder 或 oracle

### Requirement: Beam behavior is absent
R-GCN 代码、配置、checkpoint 和 trace MUST NOT 包含 beam decoder、select/stop action、frontier state、dynamic oracle、beam/stop/frontier/coverage loss、beam size、max steps、length penalty 或 beam hypothesis。

#### Scenario: Inspect model configuration
- **WHEN** 调用方解析任一 R-GCN model config
- **THEN** 不存在 beam、stop、frontier 或 oracle 配置字段

#### Scenario: Retrieve top-k
- **WHEN** retriever 请求 top-k evidence
- **THEN** 系统直接截取排序后的 node logits，不运行 autoregressive decoding

### Requirement: Beam checkpoints are incompatible
新 R-GCN checkpoint schema SHALL 只描述 node-wise model，并 MUST NOT 提供旧 beam checkpoint compatibility loader。

#### Scenario: Load legacy beam checkpoint
- **WHEN** 调用方尝试加载缺少当前 node-wise schema 或包含旧 beam-only schema 的 checkpoint
- **THEN** loader 明确失败并要求重新训练
