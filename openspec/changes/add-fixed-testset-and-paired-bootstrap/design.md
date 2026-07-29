## Context

主结果需要满足三条统计口径：可训练方法跑 5 seeds、确定性方法跑 1 次、所有方法共享同一固定 test，并给出 paired bootstrap 95% CI。现有链路：

- `experiment/run.py` → `run_experiment`（`graph_memory/experiment/workflow.py`），单 run = 单方法 + 单 `config.seed`。
- test split 由 `_prepare_config(config, "test")` 把顶层 `config.seed` 传入 `prepare_split` → `sample_split`，后者 `random.Random(seed).shuffle` 后按 offset/count 取样。
- `twowiki_provenance` 走 transform（`graph_memory/stages/transform.py`），test 由 `deterministic_dev_test_partition(seed=config.seed, dev_fraction=...)` 从 dev 源切出，`transform.seed` 默认 13。
- evaluate（`graph_memory/evaluation/suites.py`）算出 `per_task_rows` 但仅用于聚合后丢弃；落盘只有聚合 `metrics.csv` 与 `failure_cases.jsonl`。
- paired bootstrap 仅存在于 `graph_memory/analysis/provenance_ablation.py`，服务 provenance 消融，输入需外部拼装 `per_task`。

用户决策：test 采用固定 `split_seed`（保留随机打散，但对所有 run 恒定，默认 13）；transform 也复用同一固定 seed；不做多方法/多 seed 自动编排（块 C 跳过，操作者手动选方法运行）。

## Goals / Non-Goals

Goals:
- test 逐记录固定，与 `config.seed` 解耦；train/dev 仍随 `config.seed` 变化。
- evaluate 落盘 per-task 指标作为配对数据源。
- 通用 paired bootstrap，服务主结果任意「方法 vs baseline」，含共享 test 对齐强校验。

Non-Goals:
- 自动编排哪些方法跑几次（手动）。
- 新增方法 trainable/deterministic 的注册元数据。
- 改动 train/dev 的采样语义或训练随机性。

## Decisions

### D1: 固定 split_seed 的注入位置
在 `ResolvedExperimentConfig` / `ExperimentConfig` 顶层新增 `split_seed: ScientificInt = 13`（`configs/config.yaml` 提供默认值）。`_prepare_config` 对 `split == "test"` 传 `seed=config.split_seed`，train/dev 继续传 `config.seed`。

理由：最小侵入，保留 `sample_split` 现有 shuffle 语义（方案 2），只换 seed 来源。`split_seed` 作为顶层字段可被所有 stage 一致引用，避免散落。

备选（已否决）：test 完全不 shuffle（方案 1）——用户已选方案 2，保留打散。

### D2: transform 路径复用固定 split_seed
`transform_twowiki_task` 里 `deterministic_dev_test_partition` 的 `seed` 改为取固定 `split_seed`（而非 `transform.seed`/`config.seed`）。`transform.seed` 继续用于转换内部的候选/边打散（与 test 拆分无关），保持转换产物 identity 稳定；仅 dev/test 边界的 seed 对齐到 `split_seed`。

注意缓存：若 `split_seed` 默认与既有 `transform.seed=13` 一致，则 transform test 输出保持不变，缓存不失效；通用数据集 test digest 因 seed 来源变更（值仍为 13）在实现上等价，但需确认 digest 计算是否读 `config.seed` 字段——若 digest 只吃实际 seed 值 13，则无变化。

### D3: per-task 指标落盘形态
evaluate suite 的 `evaluate()` 保留 `per_task_rows`（附 `task_id`）随聚合行一起返回；`materialize_evaluation` 增写 `per_task.jsonl`（每行 `{task_id, <metric>...}`）到 publisher，并加入 artifact 输出映射；`output.py` 投影到 run 输出 `metrics/per_task.jsonl`。仅落每查询可定义的指标；micro edge precision/recall 等聚合专属指标不入 per-task。

### D4: 通用 paired bootstrap
从 `provenance_ablation.py` 抽出 `paired_bootstrap_ci(baseline_per_task, method_per_task, *, metric, samples, seed)` 到可复用位置（`graph_memory/analysis/`），provenance 消融改为调用它。新增主结果聚合入口（`scripts/` 下脚本 + `graph_memory/analysis/` 逻辑）：输入多个方法/seed 的 per-task 指标，输出可训练方法 mean±std、确定性单值、以及各方法 vs baseline 的 full-minus-method 95% CI。共享 test 对齐检查：所有被比较 per-task 的 `task_id` 集合必须完全一致，否则报错。

## Risks / Trade-offs

- **缓存失效**：test prepare 的 origin 里 seed 值不变（仍 13）则 digest 不变；但若实现改变了 origin 记录的字段（如记录 `split_seed` 名），digest 会变，需重跑 test prepare 及下游 predictions/metrics。倾向让 digest 仅取决于实际生效的 seed 值，避免无谓失效。
- **per-task 落盘体积**：每方法多一份 per-task jsonl，量级 = test 记录数 × 指标数，可接受。
- **对齐强校验**：不同数据准备口径可能导致 task_id 不一致；固定 test 后应恒一致，校验失败即暴露真实配置错误，符合预期。

## Migration

- 已生成的 prepared/predictions/metrics 若 test digest 变化则视为失效，需重跑受影响 run；provenance transform 在 seed=13 对齐下应保持不变。
- provenance 消融分析改用抽出的通用 bootstrap，行为不变（同 seed、同 samples 应给出一致区间）。
