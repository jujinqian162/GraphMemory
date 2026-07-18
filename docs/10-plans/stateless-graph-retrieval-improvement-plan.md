# 两个非训练图检索方法的局部排序改进计划

> 状态：method-only 实施边界
> 日期：2026-07-18
> 基线分支：main
> 目标方法：graphrag、execution_provenance_retriever

## 1. 目标

本计划只改进两个非训练 retrieval method：

1. GraphRAG 不再将 Dense 分数与全局 PPR 分数混合；
2. Execution-Provenance Retriever 不再把结构奖励加到大量路径节点后做全局重排；
3. 两个方法都只允许高置信局部 partner promotion；
4. 没有有效 proposal 时，node 顺序、score 和 RankedNode 对象严格回退为 Dense；
5. 保持现有 method ID、prepare/retrieve/evaluate/aggregate workflow 和输出主结构。

## 2. 硬性范围

本 change 不得修改：

- graph_memory/datasets/ 下的任何 schema、converter、parser、projector 或 scorer；
- configs/dataset/、prepared-data identity、manifest 或 cache key；
- 现有 twowiki_provenance fixture 和数据生成脚本；
- Dense-FT、R-GCN、training pairs、checkpoint 或 model cache；
- 共享 evaluation metric 列、evaluation artifact role 或 aggregate 输出；
- planner、workflow 或公共 method ID。

现有数据集和 request 是两个方法的输入。即使数据集存在需要单独研究的问题，也必须使用后续独立 change 和独立数据身份处理，不能在本 change 中迁移原数据。

## 3. 共同排序不变量

两个方法分别在自己的实现内满足：

1. Dense 完整排序先产生；
2. 只有原始 Dense top-S candidate 可以作为 anchor；
3. 每个 anchor 最多选择一个 partner；
4. 同一 partner 的冲突按 proposal confidence、anchor 原始 Dense rank、partner 原始 Dense rank、node ID 决定；
5. preserve_dense_top_n 指定的前缀成员及顺序不可变化；
6. partner 已位于 anchor 之前、位于保护前缀或不能产生实际移动时，proposal 是 no-op；
7. 有效 partner 插入 anchor 之后，同时服从保护前缀；
8. 除被提升 partner 外，其他 candidate 的相对 Dense 顺序不变；
9. 没有实际移动时原样返回 Dense；
10. 只有实际 promotion 对应的 candidate edge 可以进入 retrieved_edges。

发生 promotion 后，使用原 Dense 排序的降序 score 槽位为新顺序重新赋值，保证输出 score 单调且确定。

## 4. Execution-Provenance Retriever

### 4.1 配置

保留以下方法字段：

- seed_top_s
- beam_width
- max_hops
- max_paths_per_seed，固定为 1
- max_path_expansions
- min_path_confidence
- preserve_dense_top_n
- hop_penalty

删除旧的全局加权字段，如 semantic_weight、dependency_weight、binding_weight、grounding_weight、top_paths 和 invalidation_penalty。严格配置应拒绝这些字段。

### 4.2 路径合法性

路径 proposal 必须：

- 从原始 Dense seed 开始；
- 在另一个可检索 candidate 结束；
- 满足 FEEDS binding 与 endpoint metadata；
- 具有所需 FEEDS/RETURNS 结构；
- 不经过失效生命周期。

binding、completeness 和 lifecycle 只作为布尔 gate，不作为加分项。

### 4.3 路径置信度

方法只消费现有 ExecutionProvenanceRankingRequest 图中的 edge.weight，不重新生成或迁移数据集图。

路径置信度使用已有语义边权重一次，并仅附加超过基础路径长度后的 hop penalty。不得再次叠加 semantic_rank、semantic_score、Dense endpoint score 或固定结构奖励。

### 4.4 搜索与输出

搜索受 seed、beam、hop、expansion 和每 seed 一个 partner 的上限约束。方法只输出实际提升 partner 的 candidate-level feeds edge。结构内部 call/output edge 保留在 native trace，不进入 retrieved_edges。

## 5. GraphRAG

### 5.1 Typed entity evidence

GraphRAG 在现有 TextRankingRequest 上构造方法私有的：

- title entity mention；
- body entity mention；
- title entity group；
- sentence resolver evidence；
- directed candidate bridge proposal。

entity normalization、alias prior、document frequency 和 normalized IDF 必须确定。歧义 alias 和高频 hub 可以被记录，但不得触发 promotion。

### 5.2 Sentence resolver

title group 只作为候选句组，不能直接展开为多个 bridge。

GraphRAG 私有 Frozen Dense resolver：

- 复用 Dense ranker 的同一个 encoder 实例；
- 批量编码唯一 candidate passage；
- 对每个 anchor/group 选择 top-1 candidate；
- 分数相同时按原始 Dense rank 和 node ID 决定；
- 多 candidate group 的 top-1/top-2 margin 低于阈值时 abstain。

resolver score 只负责选择和 gate，不与 Dense score 做全局融合。

### 5.3 Bridge 与排序

bridge confidence 是 anchor/entity/title evidence 的 pair-local 置信度。每个 anchor 最多接受一个 partner。方法不得运行 PPR、RRF 或全局 graph-score fusion。

只有 resolver 唯一选中且实际被稳定插入的 partner 才生成 bridge_to edge。

## 6. Native trace

两个方法只增加自己的 native trace：

- 原始 Dense rank/score 和最终 rank；
- seed 与保护前缀；
- proposal confidence；
- gate 结果；
- accepted/rejected/conflict/no-op 原因；
- partner displacement；
- exact Dense fallback；
- 实际 emitted candidate edges。

trace 扩展不得修改 dataset record、prepare/cache identity、共享 metric 列或 evaluation artifact。

为保护旧缓存和训练方法：

- 旧 execution_provenance trace 保持原义；
- 新非训练方法使用独立 execution_provenance_local trace；
- GraphRAG 新实现使用 typed_local_bridge trace；
- validator 继续接受旧 trace，同时校验新 method-local trace 的核心引用和状态一致性。

## 7. 文件所有权

允许修改：

- configs/method/graphrag.yaml
- configs/method/execution_provenance_retriever.yaml
- graph_memory/retrieval/methods/graphrag/
- graph_memory/retrieval/methods/execution_provenance/
- graph_memory/retrieval/requests/graphrag.py
- graph_memory/retrieval/contracts.py
- graph_memory/retrieval/execution/results.py
- graph_memory/registry/retrieval_builders.py
- graph_memory/experiment/config.py
- graph_memory/stages/retrieve.py
- graph_memory/validation/ranking.py
- 两个方法的聚焦测试和直接契约文档

其他文件默认禁止修改。

## 8. 必须验证的行为

Execution-Provenance：

- binding/completeness/lifecycle 任一失败时 abstain；
- confidence 只消费现有 edge.weight 一次；
- beam/hop/expansion/per-seed 上限生效；
- conflict 确定；
- 保护前缀和非 promoted 相对顺序保持；
- 无移动时精确 Dense fallback；
- retrieved_edges 只含真实 promotion。

GraphRAG：

- title/body mention 分离；
- hub 和 alias gate 生效；
- singleton、top-1 tie 和低 margin abstention 生效；
- Dense ranker 与 resolver 共享 encoder 实例；
- 不再运行全局 PPR/fusion；
- 局部稳定插入和精确 Dense fallback；
- retrieved_edges 只含真实 promotion。

兼容性：

- 原 twowiki_provenance dataset、fixture 和 dataset tests 不变；
- Provenance R-GCN 训练和 inference tests 不变；
- 原 evaluation 输出契约不变；
- 旧缓存不会因 dataset/schema/version 变化而失效。

## 9. 完成定义

1. Git diff 不包含 dataset、training method 或共享 evaluation 文件；
2. 两个方法的聚焦行为测试通过；
3. 原 twowiki_provenance 与 Provenance R-GCN 回归测试通过；
4. 全量 pytest、Ruff、basedpyright、compileall 和 git diff --check 通过；
5. OpenSpec strict validation 通过；
6. change 文档明确记录 method-only 边界。
