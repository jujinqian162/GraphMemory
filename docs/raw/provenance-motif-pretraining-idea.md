# 基于真实 Agent 轨迹的 Provenance Motif 自监督训练设想

> 状态：Idea 记录，尚非最终实验设计。

## 核心思路

使用包含原生 `tool_call`、arguments、tool output 和 call ID 的真实 Agent execution trajectory 作为数据源，优先考虑 Toolathlon、ISETrace 和 General-AgentBench，而不再将普通 action–observation 数据包装成工具轨迹。

每条 trajectory 先以 **query-independent、label-blind** 的方式构造成执行 provenance graph。原生关系包括 `contains`、`invokes`、`returns`、`precedes`；对结构化 JSON arguments/results，可进一步提取高置信的字段级 `feeds` 数据流。

## 子图与标签构造

不把训练集限定为固定的“两节点一条边”，也不均匀随机抽取任意连通子图，而是从预定义的 provenance motif 库中动态采样，例如：

- 单跳 output-to-call 数据流；
- 多跳跨工具数据流；
- 多源 output 汇入一个 downstream call；
- 分支与聚合；
- 失败、诊断、参数修正与重试；
- 有原生记录时的状态更新、覆盖和多 Agent handoff。

每个样本保存 `anchor_node_ids`、`target_node_ids`、完整 `support_node_ids`，以及有可靠标签时的 `support_edge_ids`。采样过程只生成监督标签，不改变原始图。

## Train / Validation / Test

### Train

使用 schema-aware 模板将 motif 自动 verbalize 成 query，无需人工 QA 标签，也无需调用 LLM。通过多模板、不同询问方向、slot masking、路径长度 curriculum 和结构匹配的 hard negatives，训练 query-conditioned R-GCN 完成节点排序与完整支持集恢复。

该过程更准确地称为：**利用已有图结构的自监督 / pseudo-label 训练**，而不是完全无监督训练。

### Validation / Test

Validation 不能只复用训练模板，应同时包含：

1. held-out 模板；
2. held-out task、tool 和 domain；
3. LLM 根据采样 motif 与真实 trace 内容生成的自然 query。

Test query 可由 LLM 低成本批量生成，但不得包含 node ID、turn ID、edge type 或直接答案泄露。通过 support-only、leave-one-out、distractor-only 和 query-only 检查过滤，并人工复核一个较小子集。

## 预期贡献

潜在贡献不是“首次从图路径生成问题”，而是：

> 从真实 Agent tool-execution graph 中动态采样 typed provenance motifs，以 LLM-free 模板自动构造大规模 query–support supervision，并验证 relation-aware retriever 能否迁移到 LLM/人工生成的自然 memory queries。

若该方法能在 unseen templates、tools、tasks、domains 和跨数据集测试中保持增益，可进一步讨论其在具有文本节点和可解释关系类型的大型图上的低成本训练价值。

## 主要风险

- 模板词汇或固定遍历方向造成 shortcut；
- benchmark 只包含图方法偏好的 dependency query；
- 自动 `feeds` lineage 存在误匹配；
- template-train 与 natural-query test 的分布差异过大；
- 同一 task 的多模型、多次 rollout 跨 split 泄露。

后续正式设计需补充 motif 定义、数据划分、标签验证、flat/iterative retrieval 基线及 relation/topology ablation。
