## Why

导师要求主结果满足三条统计口径：所有可训练方法统一跑 5 个 seeds、确定性检索方法只跑一次、所有方法共享同一个固定测试集，并额外用 paired bootstrap 给出 95% 置信区间。当前实现有偏移：

- **测试集未固定。** 通用数据集（hotpotqa/twowiki/musique）的 test split 由 `sample_split` 用顶层 `config.seed` 做 `random.Random(seed).shuffle` 采样（`graph_memory/datasets/splits.py`、`graph_memory/experiment/workflow.py:_prepare_config`）。不同 seed 会选出不同的 test 子集（fixed-count profile）或不同顺序（all-available profile），使得测试集与训练 seed 耦合。这直接破坏「可训练方法跨 5 seed 共享同一 test」以及「确定性方法结果对 seed 不变」两个前提。`twowiki_provenance` 走 transform 路径，test 已由 `deterministic_dev_test_partition` 按固定 `transform.seed` 切分，但该 seed 与 test 拆分口径需要与通用路径统一为同一个固定值语义。
- **paired bootstrap 数据源缺失且作用域过窄。** 逐查询配对置信区间只在 `graph_memory/analysis/provenance_ablation.py` 存在，且仅服务 provenance 消融。配对需要 per-task 指标，但 evaluate 阶段（`graph_memory/evaluation/suites.py`）算出的 `per_task_rows` 只用于聚合后即丢弃，没有落盘，主结果无法做 paired bootstrap。

本 change 修复固定测试集语义、落盘 per-task 指标、并把 paired bootstrap 95% CI 泛化到主结果任意方法对比。多方法/多 seed 的批量编排（哪些方法跑几次）不在本 change 范围内，由操作者手动选择运行。

## What Changes

- **固定测试集（解耦 test 与训练 seed）。** 为 test split 引入一个与 `config.seed` 无关、全局固定的 `split_seed`（默认 `13`）。test 的采样只用 `split_seed`，train/dev 维持现有基于 `config.seed` 的行为。`twowiki_provenance` 的 transform dev/test 切分复用同一个固定 `split_seed` 语义，使所有数据集的 test 在任意 run（任意训练 seed、任意方法）下逐记录恒定。
- **落盘 per-task 指标。** evaluate 阶段保留并持久化 `per_task` 指标（按 `task_id` 索引，含各 Recall/F1/Full Support/MRR 等），作为 paired bootstrap 的数据源；投影到 run 输出目录。
- **通用化 paired bootstrap CI。** 从 provenance 消融逻辑抽出可复用的 paired bootstrap，扩展到主结果的任意「方法 vs baseline」比较：读取每个方法的 per-task 指标，对可训练方法给出跨 seed 的 mean±std，并给出与 baseline 的 full-minus-method 逐查询配对 95% CI；确定性方法单次值直接进表。要求所有被比较方法共享同一 test（task_id 集合一致），否则显式报错。

## Capabilities

### New Capabilities

- `experiment-fixed-test-split`: test split 使用与训练 seed 解耦的固定 `split_seed`，保证跨方法、跨 seed 逐记录一致的共享测试集；通用数据集与 `twowiki_provenance` transform 路径统一该语义。
- `evaluation-per-task-metrics`: evaluate 阶段落盘按 `task_id` 索引的 per-task 指标，作为配对统计的数据源，并投影到 run 输出。
- `paired-bootstrap-main-results`: 主结果聚合支持跨 seed mean±std（可训练方法）、单次值（确定性方法）、以及方法对 baseline 的逐查询 paired bootstrap 95% CI，并强制共享测试集对齐检查。

### Modified Capabilities

None. 仓库当前 `openspec/specs/` 下无已 promote 的基线规格；本 change 以 ADDED 形式确立上述契约。

## Impact

- 数据准备与拆分：`graph_memory/datasets/splits.py`、`graph_memory/stages/prepare.py`、`graph_memory/experiment/workflow.py`、`graph_memory/experiment/config.py`（新增固定 `split_seed`）。
- transform 路径：`graph_memory/stages/transform.py`、`configs/dataset/twowiki_provenance.yaml`（dev/test 切分 seed 与固定 test 语义对齐）。
- 评测落盘：`graph_memory/evaluation/suites.py`、`graph_memory/stages/evaluate.py`、`graph_memory/experiment/output.py`（per-task 指标持久化与投影）。
- 统计分析：`graph_memory/analysis/provenance_ablation.py`（抽出通用 paired bootstrap）、新增或扩展主结果聚合脚本 `scripts/`。
- 缓存影响：test prepare / transform 的 artifact digest 会变化，已生成的 prepared/predictions/metrics 需要重跑或接受缓存失效。
- 测试：数据拆分固定性、per-task 落盘契约、paired bootstrap 与共享 test 对齐检查的聚焦测试。不含多方法/多 seed 的自动编排。
