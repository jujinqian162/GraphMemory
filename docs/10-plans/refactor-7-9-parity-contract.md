# 7-9 重构行为等价验收契约

日期：2026-07-10

本文件固定 `refactor-experiment-config-workflow` 迁移前后的比较规则。它不改变
`refactor-7-9-analysis.md` 的设计决策，只把“等价”变成可执行的验收口径。

## 必须精确相等

- dataset/profile/method 选择、split source/count/offset/seed 和所有 resolved scientific config；
- stage DAG、隐藏依赖、stage/method/split/variant 顺序、artifact role 和低层 script；
- prediction、failure case、selected config、checkpoint metadata、metric 和 aggregate table schema；
- BM25、固定输入的图构建、evaluation、aggregate 等确定性路径的结构化输出与逻辑指标；
- cache 状态分类、ablation alias/invalidation 和 completed-prefix resume 决策。

时间戳、临时绝对根路径、环境版本字符串和测量型 latency/memory 数值不进入确定性内容比较；它们的
字段存在性和类型仍须精确匹配。

## 可容差比较

只有实际重新训练得到的浮点 scalar 允许容差：相对误差 `1e-3`，绝对误差 `1e-4`。适用对象包括
训练 loss、dev loss、grad norm、learning rate 镜像值、Dense-FT eval scalar 和 best-dev scalar。
epoch/global-step/count、best epoch、negative 类型计数、方法名、artifact path/role、模型结构、排序
结果和所有 categorical 值必须精确相等。

同一 checkpoint 的 retrieval/evaluation 指标不按“训练浮点值”处理，必须精确相等。若一次重新训练
使排名或 aggregate 指标超出上述约束，不能通过扩大容差解决；必须记录为有意行为变化并先更新
分析文档与 OpenSpec。

对应的测试 helper 位于 `tests/refactor_7_9_assertions.py`，冻结夹具位于
`tests/fixtures/refactor_7_9/`。
