# 7-9 重构计划分析与设计反馈

日期：2026-07-10

分析基线：`main`，commit `cd48048`

原始需求：`docs/raw/REFACTOR-7-9.md`

## 结论

这次重构值得做，也可以在 Python 3.10 上完成。Hydra、Pydantic V2、MLflow
三者的职责互补，选型本身合理；但不能把它们当作一个会自动解决 workflow、缓存和产物依赖的
“实验框架”。正确边界应当是：

```text
Hydra                Pydantic              仓库自己的 experiment core       MLflow
组合配置和 CLI 覆写 -> 校验完整输入契约 -> 解析 DAG、产物、缓存和执行顺序 -> 记录参数、指标和可视化
```

四项直接判断：

1. **采用 Hydra，但只让 experiment 入口做 Hydra composition。** 不要让高层
   `run.py` 启动一个 Hydra job 后，每个低层 script 又各自创建第二层 Hydra job。
2. **采用 Pydantic V2，替换当前手写的 config structuring/unknown-field/profile
   validation。** Pydantic 不替代数据泄漏校验、artifact 校验、checkpoint 校验和 workflow
   依赖校验。
3. **采用 MLflow，但不能把 MLflow run status 当作 cache truth。** 缓存原样保留当前
   “artifact + run summary 状态检查 + completed-prefix resume”语义，不扩展为内容寻址缓存。
4. **本轮不使用 `hydra.utils.instantiate()` 重写任何 workflow/runtime 组件。** 当前 registry 已经
   明确记录了 method lifecycle、训练依赖和 artifact 类型。用 `_target_` 重写这套逻辑会产生第二份
   运行时注册表，反而更难读。未来若出现两个以上实现的无状态叶子组件，应作为另一项变更评估，
   不属于本轮范围。

截至分析日期，稳定版 `hydra-core 1.3.4` 明确覆盖 Python 3.10；Pydantic V2 和当前 MLflow
也覆盖 Python 3.10。项目应锁定稳定大版本并提交 `uv.lock`，不要采用 Hydra 1.4 dev 版本。
参考：[hydra-core releases](https://pypi.org/project/hydra-core/)、
[Hydra 1.3 output directory](https://hydra.cc/docs/1.3/configure_hydra/workdir/)、
[Hydra instantiate](https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/)、
[Pydantic ConfigDict](https://docs.pydantic.dev/latest/api/config/)、
[OmegaConf interpolation/to_container](https://omegaconf.readthedocs.io/en/latest/usage.html)、
[MLflow backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)。

### 决策状态

本文从本节开始是本轮重构的设计基线，不再是候选方案列表。文中的“必须”“锁定”“删除”均是
实现和验收要求；“未来”只表示明确排除在本轮之外的扩展，不是实现时可临时选择的第二条路径。

本轮还锁定以下边界：

- 使用 Hydra BasicLauncher，single-run 和 multirun 都顺序执行；
- cache 只保留当前 named run 内的 completed-prefix resume，不增加 hash、跨 run cache 或逐 DAG
  node cache；
- MLflow 使用仓库内统一 SQLite store，不提供静默关闭或自动切换 backend；
- 不保留旧 JSON、旧 argparse 或旧 `scripts/workflow` 的兼容入口；stage run summary 作为 cache
  证据保留，但改成 typed contract 并集中公共写入逻辑；
- 除本文“有意改变的公开行为”明确列出的项目外，当前能力全部保留并进入验收矩阵。

## 当前系统的真实形态

原计划说 `experiment.py` 职责庞大，这个判断方向正确，但当前职责不只在
`scripts/experiment.py`。该文件只有 331 行，主要复杂度已经分散到 `scripts/workflow/`：

```text
JSON experiment config + argparse
              |
              v
scripts/experiment.py
              |
              v
scripts/workflow/manifest.py
  - profile/default/CLI merge
  - run layout
  - resolved method config
  - generated stage config
              |
              v
scripts/workflow/planner.py + registry.py + workflows.py
  - method lifecycle
  - train/tune dependencies
  - from/to stage selection
  - ablation invalidation boundary
              |
              v
subprocess -> scripts/<stage>.py --config <generated JSON>
              |
              v
artifact + adjacent run_summary.json
              |
              v
scripts/workflow/status.py -> resume.py
  - missing / complete / stale / alias
  - prune completed prefix
```

因此，本次不是简单的“移动 `scripts/experiment.py` 并把 JSON 改成 YAML”。真正需要重构的是：

- config composition；
- typed experiment/run/stage contract；
- workflow DAG；
- run layout；
- artifact status 和 cache resume；
- stage CLI adapter；
- tracking/observability。

## `experiment.py` 当前已经提供的能力

以下列表是新实现的默认验收基线。只有本文“有意改变的公开行为”明确列出的项目允许改变，其余
能力不得删减或在迁移中无声丢失。

### 1. 入口与配置

1. 默认加载 HotpotQA evidence retrieval 配置。
2. 支持以完整路径或配置名选择 experiment config。
3. 支持 profile；未显式指定时当前代码最终回落到 `quick`。
4. 支持 `--top-k` 覆写。
5. 支持重复 `--method` 或逗号分隔 `--methods` 选择方法。
6. 支持 named run 和自定义 `run_root`。
7. 生成并持久化 effective config。
8. 生成 method-resolved config 和每个 stage/method 的执行配置。
9. 对已存在 run 拒绝无意的 config/profile/method 变更。
10. 支持显式 `--force` 删除旧 run 并完整重建。

### 2. 数据集、profile 与 split

1. 当前 workflow 支持 HotpotQA、2WikiMultiHopQA、MuSiQue 三套 dataset adapter。
2. 为 train/dev/test 分别处理 raw source、sample count、seed 和 offset。
3. 保持 dev tuning 与 test evaluation 的 split 边界。
4. profile 同时影响数据量和 trainable method 的训练配置。
5. Memory Stream 支持外部 importance artifact、特殊 split source 和覆盖数量上限。

### 3. 方法和依赖

当前 registry 有八个 public method：

1. `bm25`
2. `dense`
3. `memory_stream`
4. `bm25_graph_rerank`
5. `dense_graph_rerank`
6. `dense_rgcn_graph_retriever`
7. `dense_ft`
8. `dense_ft_rgcn_graph_retriever`

默认 HotpotQA quick experiment 运行其中七个，明确不包含 `memory_stream`。

依赖能力包括：

1. 根据 method lifecycle 自动决定所需 stages。
2. graph-rerank retrieval 依赖 dev tuning 产出的 selected config。
3. trainable retrieval 依赖 pairs 和 checkpoint。
4. `dense_ft_rgcn_graph_retriever` 即使不是公开选择的 `dense_ft` run，也会自动先执行
   `dense_ft` pairs/train，并把其 best model 作为 seed checkpoint。
5. dependency artifact 已存在时允许从后续 stage 开始。
6. dependency artifact 缺失时在执行前失败，并给出缺失路径。
7. checkpoint 同时支持单文件 R-GCN checkpoint 和 Dense-FT model directory 两种形态。

### 4. 计划与执行

1. `plan` 打印所有低层 script、method/split/variant 和完整 argv，不执行 stage。
2. `run` 在执行每条命令之前打印与 `plan` 相同的命令块。
3. 使用 subprocess 顺序执行，任意 stage 非零退出即停止。
4. 支持 `--from` / `--to` 选择合法 workflow 范围。
5. 支持方法子集和 stage 子集组合。
6. `plan` / `run` 在 manifest 不存在时会隐式初始化。
7. `status` 可独立查看当前 run 状态。

### 5. 缓存与中断恢复

1. 状态不是只检查“文件是否存在”，而是区分 `missing`、`complete`、`stale`、`alias`。
2. prepare、graphs、pairs、tune、train、retrieve、evaluate、aggregate 都检查自己的
   expected input/output/effective config。
3. 输出存在但缺少成功 run summary 时会判定为 stale。
4. 默认从 plan 中剪掉已经完成的连续前缀，从第一个 pending/stale stage 恢复。
5. `--no-cache` 可强制显示或执行完整选择范围。
6. run 结束后把 live status 写回 manifest。
7. 失败 stage 的残留输出不能单凭存在性变成 cache hit。

注意：当前语义就是“跳过完成的连续前缀”，不是任意 DAG node 的独立 content-addressed cache。
新实现必须保持该语义，不升级 cache；同时不能退化成 MLflow 显示 FINISHED 就跳过。

### 6. 调参与 ablation

1. graph rerank 和 Memory Stream 使用独立 search-space config。
2. selected tuning config 写入 run-local `tuned/`，不修改全局 config。
3. 支持注册的 ablation suite 和 variant 筛选。
4. 支持 `ablations-only`，并检查 ordinary baseline metric 前置条件。
5. 根据 variant 的 changed dimensions 计算最早 invalidated stage。
6. 未失效 artifact 通过 alias 复用，失效 artifact 写入 variant namespace。
7. baseline alias 与被选 variant 一起进入 ablation table。

### 7. 结果与可检查性

1. 每个 method 生成 prediction、metric、failure case。
2. trainable method 生成 pair summary、training metrics、best checkpoint/model。
3. aggregate 生成 main、path、efficiency 和可选 ablation table。
4. 当前支持列出 stages、methods、configs、profiles、recipes、ablations；新实现按下文决策删除
   recipe 概念和 recipe list。
5. manifest 固定记录选中 methods、stages、artifact paths、resolved configs 和状态。
6. 所有 run-specific artifact 都隔离在 named run 下面。

## 有意改变的公开行为

| 当前行为 | 锁定后的行为 |
|---|---|
| 位置参数和 argparse subcommand | 删除；只保留 Hydra 入口和 `key=value` override |
| 以 JSON 路径或 config 名选择 experiment | 删除任意 JSON 路径入口；普通实验使用 dataset/profile/method override，少数特殊组合使用 Hydra root experiment preset |
| 自定义 `run_root` | 删除；run root 固定为 `runs/`，单次实验固定为 `runs/${name}` |
| `--force` | 删除；唯一 destructive 操作是 `experiment/reset.py name=...` |
| `--no-cache` | 改为 `cache.enabled=false`；忽略 completed-prefix，仍写原语义的 stage run summary |
| `--from` / `--to` | 改为 `stages.from=<stage>` / `stages.to=<stage>` |
| list subcommands | 改为 `experiment/inspect.py kind=stages|methods|datasets|profiles|configs|ablations|jobs`；删除 recipe list |
| JSON effective/stage configs | 改为 resolved YAML 和 Pydantic stage YAML |
| `manifest.json` 和相邻 `*.run_summary.json` | 保留现有职责，分别收敛为 typed `RunState` 与 `StageRunSummary`，不增加新 cache record |

除上表外，默认方法集合、依赖补全、plan parity、cache resume、ablation、三套 dataset 和最终交付
表格语义都保持。

## 锁定的目标调用链

```text
experiment/run.py or experiment/plan.py
  |
  | Hydra compose: dataset/profile/method-config/search-space + CLI overrides
  v
ExperimentConfig.model_validate(resolved primitive container)
  |
  +--> RunLayout: 唯一负责 run 内所有派生路径
  |
  +--> WorkflowPlanner: method registry -> typed StageInvocation DAG
  |       |
  |       +--> hidden train dependencies
  |       +--> tune dependencies
  |       +--> ablation invalidation/alias
  |
  +--> ResumePlanner: artifact + StageRunSummary status
          |
          +--> complete/alias prefix: skip
          |
          +--> first missing/stale: execute remaining ordered commands
                         |
                         +--> write resolved stage YAML
                         |
                         v
                subprocess scripts/<stage>.py --config <stage.yaml>
                         |
                         v
                graph_memory/stages/<stage>.py
                         |
                         +--> artifact outputs
                         +--> atomic StageRunSummary
                         +--> MLflow metrics/tags/curated artifacts
```

这个结构保留了用户可见的 script plan、进程隔离、GPU 资源释放和低层独立调试能力；同时把
config、DAG、cache、tracking 的职责拆开。

## 新入口与独立 script CLI

### Experiment 入口

统一使用 Hydra 的 `key=value` 形式，不再支持位置参数别名：

```powershell
uv run python experiment/plan.py name=demo
uv run python experiment/run.py name=demo
uv run python experiment/run.py name=demo dataset=2wiki profile=cloud-full methods='[bm25,dense]'
uv run python experiment/run.py name=demo device=cuda seed=14
uv run python experiment/run.py name=demo stages.from=retrieve stages.to=aggregate
uv run python experiment/run.py name=demo cache.enabled=false
uv run python experiment/run.py name=dense-ft-lr methods='[dense_ft]' method_configs.dense_ft.train.trainer.learning_rate=3e-5
uv run python experiment/run.py name=rgcn-ablation methods='[dense_rgcn_graph_retriever]' ablation.variants=all
```

公开入口锁定为五个：

- `experiment/plan.py`：compose、validate、解析 DAG、检查 cache、打印计划；不启动 MLflow run。
- `experiment/run.py`：与 plan 共享同一个 planner，然后执行。
- `experiment/status.py`：按 artifact 和 typed run summary 展示 `missing/complete/stale/alias`。
- `experiment/inspect.py`：替代现有 stages/methods/configs/profiles/ablations 列表命令，并以 `kind=jobs name=<sweep>` 列出 concise selectors；不提供 recipes。
- `experiment/reset.py`：删除一个指定 named run 或 exact multirun job selector；这是唯一 destructive 入口。

不保留独立 `init`。`plan` 和 `run` 在 run 不存在时创建目录、resolved config、`RunState` 和 stage
configs；`plan` 的这些本地写入是其唯一副作用，不创建 MLflow run、不执行 stage。普通 run 参数
不能删除任何现有内容。

experiment root contract 固定包含：`name`、dataset/profile groups、`methods`、`method_configs`、
`seed`、`device`、`top_k`、`stages.from`、`stages.to`、`cache.enabled` 和 `ablation`。业务默认值全部来自
YAML；代码不得补默认值。`name` 必填，其余默认行为必须等价于 HotpotQA + quick + 默认七方法。

### 独立 scripts

不要让每个 script 都成为新的 Hydra app。每个 script 统一只接受一个完整、已解析的 stage config：

```powershell
uv run python scripts/build_graphs.py --config runs/.../stage-configs/graphs-dev.yaml
uv run python scripts/train_method.py --config runs/.../stage-configs/train-dense_ft.yaml
uv run python scripts/run_retrieval.py --config runs/.../stage-configs/retrieve-dense.yaml
```

script 的职责固定为：

1. 读取 YAML；
2. 用对应 Pydantic model 校验；
3. 做 artifact IO 和 artifact/domain validation；
4. 调用 `graph_memory.stages`；
5. 原子写输出和 typed stage run summary；
6. 始终写本地科学输出和指标文件；若收到 MLflow parent context，再镜像记录该 stage；
7. 用退出码报告成功或失败。

script 不再拥有业务默认值、profile merge、method dispatch、artifact path 推导或 cache 判断。直接
运行 script 时不要求 MLflow context，仍必须产生与 workflow 相同的 artifact 和指标语义。

## 锁定的文件职责

```text
experiment/
  plan.py                 # Hydra 入口；只调用 application service
  run.py                  # Hydra 入口；只调用 application service
  status.py               # 状态入口
  inspect.py              # registry/config group 检查入口
  reset.py                # 唯一 destructive 入口

graph_memory/experiment/
  config.py               # ExperimentConfig 和 composition -> Pydantic 边界
  planning.py             # typed DAG 与 StageInvocation
  execution.py            # subprocess execution、失败传播
  layout.py               # RunLayout，唯一派生 artifact path 的位置
  status.py               # artifact/run-summary verification
  resume.py               # completed-prefix resume decision
  state.py                # RunState / StageRunSummary
  tracking.py             # MLflow adapter；不包含 cache 决策

graph_memory/config/
  base.py                 # 项目统一 StrictConfigModel
  stages.py               # discriminated stage config models
  artifacts.py            # typed artifact references

scripts/
  prepare_*.py            # 每个 stage/dataset 的薄 adapter
  build_graphs.py
  build_train_pairs.py
  tune_*.py
  train_method.py
  run_retrieval.py
  evaluate_retrieval.py
  aggregate_tables.py
```

迁移完成后，当前 `scripts/workflow/` 应整体消失；可复用主逻辑进入
`graph_memory/experiment/`，不是继续留一份“新名字的 scripts/workflow”。

## 锁定的 Hydra config group 设计

作者侧文件结构固定为：

```text
configs/
  experiment.yaml
  2wiki_tiny.yaml                 # 少量特殊 root preset；由 Hydra --config-name 选择
  hotpotqa_memory_stream.yaml
  hotpotqa_dev_full.yaml
  dataset/
    hotpotqa.yaml
    2wiki.yaml
    musique.yaml
  profile/
    smoke.yaml
    quick.yaml
    full.yaml
    cloud-quick.yaml
    cloud-full.yaml
  graph/
    default.yaml
    spacy.yaml
  method_configs/
    bm25.yaml
    dense.yaml
    memory_stream.yaml
    bm25_graph_rerank.yaml
    dense_graph_rerank.yaml
    dense_rgcn_graph_retriever.yaml
    dense_ft.yaml
    dense_ft_rgcn_graph_retriever.yaml
  search_spaces/
    default.yaml
```

根 config 示例：

```yaml
defaults:
  - dataset: hotpotqa
  - method_configs:
      - bm25
      - dense
      - memory_stream
      - bm25_graph_rerank
      - dense_graph_rerank
      - dense_rgcn_graph_retriever
      - dense_ft
      - dense_ft_rgcn_graph_retriever
  - profile: quick
  - graph: default
  - search_spaces: default
  - _self_

name: ???
seed: 13
device: cuda
methods:
  - bm25
  - dense
  - bm25_graph_rerank
  - dense_graph_rerank
  - dense_rgcn_graph_retriever
  - dense_ft
  - dense_ft_rgcn_graph_retriever
top_k: 10
stages:
  from: prepare
  to: aggregate
cache:
  enabled: true
ablation:
  variants: []
  only: false
tracking:
  backend_store_uri: sqlite:///runs/.mlflow/tracking.db
  artifact_root: runs/.mlflow/artifacts
```

### 为什么不再使用 `method@method_configs.<name>`

`method@method_configs.dense_ft: dense_ft` 的原意，是把同一个 config group 的多个选项分别搬到
`method_configs.<name>`，避免相互覆盖。这个目标合理，但逐项写 package override 可读性差，也没有
必要。Hydra 1.3 原生支持从同一 group 选择多个 config，因此锁定为上面的 multi-select defaults：
`method_configs: [bm25, dense, ...]`。参考：[Hydra selecting multiple configs](https://hydra.cc/docs/1.3/patterns/select_multiple_configs_from_config_group/)。

每个文件自己带 method key，例如：

```yaml
# configs/method_configs/dense_ft.yaml
dense_ft:
  kind: dense_ft
  encoder: ...
  pairs: ...
  train: ...
```

组合结果自然是 `method_configs.dense_ft`。root 组合全部八份小 config，`methods` 普通列表只决定本次
执行哪些 public methods；这样 CLI 可临时改变方法集合，registry 也始终能取得
`dense_ft_rgcn_graph_retriever` 所隐含依赖的 Dense-FT config。没有动态补 compose、没有
`method@...` package 语法，也不把八份内容合并成一个大 YAML。

### Config 内容和归属

| 位置 | 固定拥有的内容 |
|---|---|
| root `experiment.yaml` | `name`、`seed`、`device`、`top_k`、选中 `methods`、stage 范围、cache 开关、ablation 选择和 Hydra/MLflow 根设置 |
| `dataset/<name>.yaml` | dataset id/task、raw paths、split source/offset/capacity、prepare adapter、projector/validator、capability constraints |
| `profile/<name>.yaml` | train/dev/test 规模 policy，以及 R-GCN、Dense-FT、pair sampling 的 profile patch |
| `method_configs/<name>.yaml` | 一个 baseline 的完整基础科学参数；只有真正消费的 encoder/scoring/pairs/train/selection 字段 |
| `graph/<name>.yaml` | graph builder 参数和实现选择 |
| `search_spaces/default.yaml` | graph rerank 与 Memory Stream 的 tuning search spaces |
| root preset | 仅用于 `2wiki_tiny`、HotpotQA Memory Stream 等少数特殊组合；选择已有 groups 并写少量显式 override |

每个 profile YAML 使用 `# @package _global_`，同时写入 `profile` 本身和少量
`method_configs.<method>` patch。profile 在 method configs 之后组合，因此这些 patch 覆盖 method
基础值；CLI override 最后生效。例如：

```yaml
# @package _global_
# configs/profile/cloud-full.yaml
profile:
  name: cloud-full
  splits:
    train: {kind: all_available}
    dev: {kind: fixed, count: 500}
    test: {kind: all_available}
method_configs:
  dense_rgcn_graph_retriever:
    train:
      model: {hidden_dim: 256, num_layers: 2}
      trainer: {batch_size: 128, epochs: 15}
  dense_ft:
    train:
      trainer: {train_batch_size: 16, eval_batch_size: 64, epochs: 1}
```

特殊 root preset 使用 Hydra 原生 `--config-name`，不定义 recipe 字段、recipe group 或 recipe list：

```powershell
uv run python experiment/run.py --config-name=2wiki_tiny name=demo
```

### Method config 与临时参数覆写

method config 的 shape 按 lifecycle 固定：

| method 类型 | 允许的主要字段 |
|---|---|
| BM25 | `scoring` |
| Dense | `encoder` |
| Memory Stream | `encoder`、`scoring`、external importance binding |
| graph rerank | seed scorer、graph scoring；search space 由 `search_spaces` 持有 |
| R-GCN | `encoder`、`pairs`、`train.model`、`train.trainer`、`train.selection` |
| Dense-FT | `encoder`、`pairs`、`train.data`、`train.trainer`、`train.selection` |
| Dense-FT seeded R-GCN | 与 R-GCN 相同；Dense-FT seed dependency 只在 registry 声明 |

临时修改一个 baseline 的 learning rate，直接覆写最终字段，不创建新 profile/config：

```powershell
# 只跑 Dense-FT，并把当前 2e-5 临时改为 3e-5
uv run python experiment/run.py name=dense-ft-lr methods='[dense_ft]' method_configs.dense_ft.train.trainer.learning_rate=3e-5

# 只跑普通 R-GCN，并把当前 1e-4 临时改为 5e-4
uv run python experiment/run.py name=rgcn-lr methods='[dense_rgcn_graph_retriever]' method_configs.dense_rgcn_graph_retriever.train.trainer.learning_rate=5e-4

# 修改 Dense-FT seeded R-GCN 的 R-GCN trainer learning rate
uv run python experiment/run.py name=dense-ft-rgcn-lr methods='[dense_ft_rgcn_graph_retriever]' method_configs.dense_ft_rgcn_graph_retriever.train.trainer.learning_rate=5e-4
```

如果仍要运行默认七方法，只临时改变其中一个 baseline，省略 `methods='[...]'` 即可；字段 override
只作用于对应 `method_configs.<name>`。

### Ablation 的合法值

`ablation` 不是 config group，而是 root typed config：

```yaml
ablation:
  variants: []   # Literal["all"] | list[AblationVariantId]
  only: false
```

- `ablation.variants=[]`：默认，不执行 ablation；
- `ablation.variants=all`：对选中的 ablation-capable methods 执行全部非 baseline variants；
- `ablation.variants='[wo_bridge,wo_graph]'`：只执行列出的 variants；
- `ablation.only=true`：只执行 variants，不重跑 ordinary baseline；要求当前 run 已有 baseline metrics。

合法 variant id 固定为：`wo_bridge`、`wo_entity_overlap`、`wo_sequential`、`wo_query_overlap`、
`wo_graph`、`wo_edge_type`、`wo_edge_weight`、`wo_seed_score`、`wo_hard_negatives`。只支持
`dense_rgcn_graph_retriever` 和 `dense_ft_rgcn_graph_retriever`。`full_rgcn` 是 ordinary run 的
baseline alias，会自动进入 ablation table，不是需要用户填写或额外执行的 variant。

```powershell
# 全部九个非 baseline variants
uv run python experiment/run.py name=rgcn-ablation methods='[dense_rgcn_graph_retriever]' ablation.variants=all

# 只跑两个 variants
uv run python experiment/run.py name=rgcn-ablation-small methods='[dense_rgcn_graph_retriever]' ablation.variants='[wo_bridge,wo_graph]'

# ordinary baseline 已存在时，只补跑一个 variant
uv run python experiment/run.py name=rgcn-run methods='[dense_rgcn_graph_retriever]' ablation.variants='[wo_seed_score]' ablation.only=true
```

### Dataset 必须是 config group，不是普通字符串

`dataset=2wiki` 不只是把 `dataset` 字段从 `hotpotqa` 改成 `2wiki`。它必须同时替换：

- raw train/dev/test paths；
- 默认 split source、offset 和逻辑 window；
- dataset validation 后每个逻辑 window 的声明容量；
- prepare adapter id；
- dataset-specific validation/projector id；
- 可用 method/capability 约束。

因此 `dataset` 必须是 Hydra config group。公开 config 名可以是 `2wiki`，内部 Python package
仍可叫 `twowiki`；dataset registry 负责把 typed dataset config 连接到 adapter。

### Profile 只选择一次

当前每个 method JSON 都有自己的 `profiles`，experiment JSON 又有一份 `profiles`，这是重复源。
新结构中只选择一次 `profile=quick`。profile group 固定提供：

- train/dev/test 的规模策略；
- R-GCN 训练规模参数；
- Dense-FT 训练规模参数；
- pair sampling 规模参数。

method config 用插值或 profile patch 取得 profile 值。dataset 与 profile 对 split 的职责锁定为：

- dataset group 定义普通实验的 raw source、seeded-sampling offset 和逻辑 window，并声明该 window
  在 dataset-specific invalid-example validation 后的 `capacity`；
- profile group 只定义与 dataset 无关的 typed policy：`fixed(count)` 或 `all_available`；
- Hydra composition 和 Pydantic validation 完成后，由唯一的 typed `SplitResolver` 将 window 与
  policy 合成为 stage config 的 `offset` 和 `count`；
- `fixed.count > capacity` 必须 fail fast，不能静默 `min(count, capacity)`；
- `all_available` 解析为当前 logical window 的 `capacity`；
- prepare adapter 在过滤 invalid examples 后重新核对声明 capacity；不一致视为 raw input/config
  drift 并失败，同时保留 raw/valid/dropped/reason counts。

profile 的 split policy 固定为：

| profile | train | dev | test |
|---|---|---|---|
| `smoke` | `fixed(1)` | `fixed(1)` | `fixed(1)` |
| `quick` | `fixed(100)` | `fixed(100)` | `fixed(100)` |
| `full` | `fixed(5000)` | `fixed(500)` | `fixed(1000)` |
| `cloud-quick` | `fixed(1000)` | `fixed(500)` | `fixed(1000)` |
| `cloud-full` | `all_available` | `fixed(500)` | `all_available` |

因此 `profile=cloud-full` 在 HotpotQA、2Wiki 和 MuSiQue 下分别读取自己的 train/test capacity，
但 dev 始终固定为 500；profile 不知道 dataset 名，dataset 也不知道 profile 名。

`2wiki_tiny` 不改变全局 `quick`/`smoke` 的定义，而由同名 root preset 显式覆盖对应 split policy 和
test window。Memory Stream 同理由 root preset 覆盖 external source/window，并由 capability validation
限制在支持的 dataset。**本轮不引入 recipe 或 `split_plan` 第四组，也不允许 resolver 出现
dataset-name 条件分支。**

### seed 与 device

顶层只保留一份用户可覆写的 `seed` 和 `device`：

```yaml
method_configs:
  dense_ft:
    pairs:
      random_seed: ${seed}
    train:
      trainer:
        random_seed: ${seed}
        device: ${device}
```

插值只放到真正消费该字段的 stage。BM25 不应为了“统一”而获得一个无意义的 `device` 字段。
也不要实现 `cuda` 失败后自动回落到 CPU；配置写 `cuda` 就应失败，用户需要 CPU 时显式
`device=cpu`。

只统一整数值还不够。所有随机 stage 在入口调用同一个 `seed_everything(seed)`，固定处理 Python
`random`、NumPy、PyTorch CPU/CUDA、DataLoader generator 和 worker seed。为保持当前训练算子与
性能语义，本轮明确不启用 `torch.use_deterministic_algorithms(True)`，也不新增第二套 determinism
mode；GPU 训练验收使用预先声明的数值容差，不承诺 bitwise identical。当前 R-GCN 和 Dense-FT 的
seed 处理必须收敛到这一入口。

### 插值的边界

适合 Hydra interpolation 的内容：

- root seed/device -> consumer config；
- dataset/profile -> resolved split window 与 count；
- run root/name ->少量根路径；
- encoder model/prefix ->需要共享的 method config。

不适合把所有 split/method/variant artifact path 都写成 YAML 插值。那些路径数量大、依赖 method
lifecycle，并且 ablation 需要 alias。应由单一 `RunLayout` 以 typed API 生成，避免把 Python
workflow 逻辑搬进 YAML。

## Pydantic 的使用边界

统一基类锁定为：

```python
class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
```

Hydra 入口只做一次边界转换：

```python
payload = OmegaConf.to_container(
    cfg,
    resolve=True,
    throw_on_missing=True,
)
config = ExperimentConfig.model_validate(payload)
```

不在 model 级开启 `strict=True`：Hydra primitive container 到 `Path`、Enum 的正常解析需要保留。
seed、count、batch size、epoch 使用 strict integer + range constraint，learning rate/权重使用 finite
number + range constraint，bool 禁止被整数或字符串代替。也就是说，结构统一 `extra="forbid"`、
`frozen=True`，科学标量逐字段 strict；实现中不得再保留两套可选策略。

### 可以交给 Pydantic 的校验

- required/unknown fields；
- Literal/Enum method 和 dataset id；
- 正数、非负数、概率和权重范围；
- method-specific discriminated union；
- cross-field config 约束；
- path 类型和 name 的 path-safe 格式；
- profile 组合后配置完整性；
- config serialization。

### 不得删除的校验

- ranking/label/graph artifact schema；
- task id 对齐和 dataset projector 校验；
- dev/test leakage 边界；
- Memory Stream importance digest/coverage；
- checkpoint metadata、encoder metadata 和文件/目录形态；
- stage dependency 是否已经满足；
- selected tuning config 是否来自对应 dev 输入；
- cached output、run summary 与当前 expected inputs/outputs/config 是否一致；
- ablation visible edge 与 gold label 边界；
- 原始数据 checksum。

### 用 discriminated union 删除无意义的 `None`

当前 `RetrieveIO` 同时包含 optional `graphs`、`selected_config`、`importance`；这是 wide config
造成的 optional，不是业务本身不可避免。新 config 应拆成 method-specific union：

- BM25/Dense：只需要 tasks/output；
- Memory Stream：importance 必填；
- graph rerank：graphs 和 selected config 必填；
- R-GCN：graphs 和 checkpoint file 必填；
- Dense-FT：model directory 必填。

同理，普通 R-GCN 和 Dense-FT-seeded R-GCN 的 train config 应分开，使后者的
`seed_checkpoint` 为 required field，而不是所有 R-GCN 都携带 `Path | None`。

Pydantic 只用于外部/跨层 contract。内部 graph node、ranking request、训练 batch 等已经清晰的
domain dataclass 不需要为了“统一技术栈”改成 Pydantic。

## `instantiate()` 判断

本轮禁止使用 `instantiate()` 的位置：

- workflow stage；
- method dependency graph；
- dataset projector；
- artifact layout；
- cache policy；
- checkpoint/model loader；
- 需要运行时 tasks、graphs、model parameters 的 trainer/retriever。

这些对象需要大量 runtime dependency，`_target_` 只会把显式 registry 调用变成 YAML 中的隐式
import path。

以下位置即使未来可能适合，也明确不在本轮引入：

- 同一 Protocol 下的无状态 optimizer/scheduler factory；
- 不依赖 artifact 的可替换 graph rule；
- 可替换的 tracking sink。

原则是：先有两个真实实现和稳定 Protocol，再引入 `_target_`。本轮验收不应把
`instantiate()` 使用数量当作目标。

## output、run identity 与 multirun

single-run 与 multirun 布局分别锁定为：

```text
runs/
  .mlflow/
    tracking.db
    artifacts/
  <name>/
    .hydra/
    run_state.yaml
    config/resolved.yaml
    config/overrides.yaml
    config/stages/...
    inputs/...
    graphs/...
    learned/...
    tuned/...
    predictions/...
    metrics/...
    tables/...
    debug/...
    # 每个主 artifact 相邻保留 typed *.run_summary.yaml

  <sweep-name>/
    multirun.yaml
    0_num_layers=2/
      .hydra/
      run_state.yaml
      config/resolved.yaml
      config/overrides.yaml
      ...
```

Hydra 明确配置：

```yaml
hydra:
  run:
    dir: runs/${name}
  sweep:
    dir: runs/${name}
    subdir: '${hydra.job.num}_${concise_override:${hydra.job.override_dirname}}'
  job:
    chdir: false
```

### Run identity 与 cache 规则

`runs/${name}` 是锁定后的 run layout。实验名同时是 single-run 的稳定身份，不再让当前日期参与
路径或 resume 判断：

1. `runs/<name>` 不存在时创建新 run；
2. 已存在时，把新 composed Pydantic config 与持久化 resolved config 直接比较；完全相等才恢复；
3. resolved config 不相等时直接失败，不能覆盖或局部复用；
4. 重建必须走显式 `experiment/reset.py name=...`；
5. 不计算 config hash、source hash 或 artifact digest。

`run_state.yaml` 包含固定的 `mode`。同一个 `<name>` 不能在 single-run 和
multirun 两种 mode 间复用；mode 不匹配时必须 reset 或更换 name。

multirun 中 `<name>` 是 sweep 名，不是每个 job 的唯一 id。`layout.py` 的唯一 formatter 在 Hydra
composition 前注册，并同时供 `hydra.sweep.subdir` 与 `RunLayout` 使用。它排除固定 root identity
overrides，把嵌套 sweep path 收敛为 leaf key，保留确定性 assignment 顺序，清洗非法路径字符，并在
leaf key 冲突时直接失败。完整 resolved config 与 override list 仍保存在每个 job 内；concise leaf
只是 selector，不是配置序列化格式，也不增加额外 cache identity。

## Cache 与中断恢复：保持当前能力

本轮不新增 cache 能力。新实现移植当前经过验证的语义：

```text
拓扑有序 StageCommand 列表
  -> 读取 artifact + 相邻 StageRunSummary
  -> 判定 missing / complete / stale / alias
  -> 跳过连续的 complete/alias 前缀
  -> 从第一个 missing/stale command 开始执行剩余后缀
```

`StageRunSummary` 是现有 run summary 的 typed Pydantic 版本，继续与主 artifact 相邻保存，固定保留：

- stage/script id；
- `status=success|failed`；
- expected inputs 和 outputs paths；
- 本 stage 实际消费的 effective config；
- started/finished/timing、counts 和失败信息。

status 检查保持当前规则：artifact 必须存在且 file/directory kind 正确，summary 必须存在、成功、
stage/script 匹配，并且 inputs、outputs、effective config 与 planner 当前期望值一致；否则为 stale。
失败 attempt 的残留输出不能成为 cache hit。ablation alias 继续视为可跳过的已完成节点。

`cache.enabled=false` 等价于当前 `--no-cache`：不剪 completed prefix，执行用户选择的完整 stage
范围，但仍写普通 `StageRunSummary`。不做跨 run cache、不做逐 DAG node 独立跳过，也不因为某个
后续 method 已完成而跳过 first pending 之后的命令。

这意味着同一路径的 raw data 或代码被原地修改时，当前 summary 规则未必能识别；本轮接受这一现有
边界，用户通过新 run name、reset 或 `cache.enabled=false` 明确重跑。MLflow status 只用于展示，
不能替代上述本地判断。

## MLflow 设计

### 存储位置

每个 `runs/<name>` 内禁止创建独立 tracking store。全项目固定使用：

```text
runs/.mlflow/tracking.db      # SQLite backend metadata
runs/.mlflow/artifacts/       # MLflow-managed curated artifacts
runs/<name>/...               # Hydra/experiment 的真实 workflow artifacts
```

本轮固定使用 SQLite backend、local artifact root 和 Hydra BasicLauncher 顺序执行。MLflow 初始化、
参数/指标写入或 child-run 结束失败都使当前 experiment attempt 失败，不能静默关闭 tracking、降级为
file store 或自动切换 backend。`plan/status/inspect/reset` 不创建 MLflow run；直接运行低层 script
且没有 parent context 时只写本地 artifacts，不自行创建孤立 MLflow run。

### Run 层级

- 一个 Hydra single-run/multirun job = 一个 MLflow parent run；
- 一个 user-visible selected baseline 或 executable ablation variant = 一个 nested child run；
- prepare、graphs、aggregate 是 parent 的 workflow/result observations，不创建 child；
- pairs、tune、train、retrieve、evaluate 都写入所属 baseline 的同一个 child；
- tuning candidate table 是 artifact，selected config 是 parameters，tuning 分数不进入 model metrics；
- resume 精确相同配置时复用 parent 和唯一 baseline child，cache hit 也从本地事实源补全该 child；
- config 变化必须是新 job，不能在同一个 MLflow run 中改参数。

### 记录内容

parent 只记录 name、dataset、profile、seed、device、methods、top-k、stage bounds、cache、ablation 和
multirun identity 等 concise parameters；resolved config/overrides 放在 `config/`，aggregate CSV 放在
`results/`，shared summaries 放在 `workflow/`。aggregate 成功后，所有 baseline results 合并为一个
`mlflow.note.content` Overview 表。MLflow 3.14 只保证该 plain-text Markdown 内容可持久化；即使 UI
渲染有限，`results/*.csv` 仍是 durable fallback。parent 没有 native result metrics、training series、
saved chart config 或 repository-generated comparison image。

baseline child 记录 method-relative encoder/pairs/tuning/model/trainer/scoring parameters、统一的 `final.*`
数值和 trainable method 的 epoch-indexed `train.*` series。counts、timings、status、error、artifact path/kind/
size 只进入 tags、parameters、Overview 或 summary artifacts，不冒充 model metrics。相同 evaluation 值不从
aggregate 再写一份；`N/A`、non-numeric、NaN 和 infinity 不写 numeric metric。用户在 MLflow Compare Runs
中选择 baseline children，直接比较相同的 `final.*` keys。

本地 workflow artifacts 是科学结果和交付的事实源；MLflow 是其精选展示镜像，不是唯一存储。
resolved config、override list、selected tuning config、candidate table、train metrics、evaluation CSV、最终
tables、stage summaries 和小型 failure summary 可以按 `config/`、`tuning/`、`training/`、`evaluation/`、
`workflow/`、`dependencies/` 上传。raw/processed dataset、完整 graph、train pairs、完整 predictions、
checkpoint/model directory 禁止上传，只记录 path、size、kind 和 role。本轮不提供“上传全部 artifact”的
fallback 开关。隐藏 Dense-FT prerequisite 的 summaries/references 放到 dependent baseline 的
`dependencies/dense_ft/`；若 Dense-FT 也被显式选择，则它拥有独立 child，dependent child 只记录引用。

切换不增加 schema/version tag、compatibility reader、legacy branch、dual write、old-run migration 或
database rewrite。existing SQLite rows 保持历史记录；former local run name 必须改用新 name，或先执行
normal explicit reset，再运行新 projection。

## 引入 MLflow 后可以删除什么

MLflow 不能替代 cache 所需的 stage run summary。下面只删除重复的观测样板，不改变 summary/status/
resume 语义。

### 可以彻底删除

1. 每个 prepare/build/tune/train/retrieve/evaluate/aggregate script 中重复的
   `started_at -> try -> build summary -> write -> except -> failed summary` 样板；改由一个共享 helper
   生成同一 `StageRunSummary` contract。
2. manifest 中重复复制的 counts、timings、environment 和 notes 观测字段；这些由 stage summary 与
   MLflow 分别持有。
3. `graph_memory/observability.py` 中既不参与 typed stage summary、也不参与 MLflow 的纯转发 API。
4. 仅为了内部训练观察自动生成的静态 training curve 流程；训练曲线由 MLflow 展示。

### 可以简化，但不能完全删除

1. `RunSummary`：替换成 Pydantic `StageRunSummary`，删除宽泛 `dict[str, Any]` 和重复构造代码，但
   每个主 artifact 相邻的 summary 文件继续保留。
2. `manifest.json`：替换成小型 typed `RunState`，保留 resolved config path、artifact layout、选中
   methods/stages、MLflow run id 和创建日期。
3. `stage_status`：不再复制回 manifest；由 artifact + typed summary 实时推导。
4. `scripts/workflow/status.py`：保留 method/stage-specific expected values，抽取公共 summary 比较逻辑，
   不引入 hash/digest layer。
5. environment 收集：改成 tracking adapter 一次性记录，不在每个 script 重复实现。
6. metric files：stage-local metric JSON/JSONL（包括 `train_metrics.jsonl`）继续作为 direct-script、
   cache validation 和离线交付的事实源；MLflow 只镜像。可删除仅用于重复展示的 method-level CSV，
   但 `main_results.csv`、`path_results.csv`、`efficiency_results.csv`、`ablation_results.csv` 必须保留，
   因为它们是论文/报告/交付的可移植产物。
7. training curve renderer：从核心 run 中移除；如需要报告 PNG/SVG，保留独立 export 工具。

### MLflow 不能让这些东西消失

- prepared input/label artifacts；
- graph artifacts；
- train pairs；
- selected tuning config；
- checkpoints/model directories；
- ranked predictions；
- failure cases；
- 最终 aggregate tables；
- typed `StageRunSummary`、status 检查和 completed-prefix resume；
- domain/leakage/checkpoint validation；
- dependency DAG 和 ablation alias 规则。

## 引入 Hydra + Pydantic 后可以删除什么

### 可以彻底删除

1. `graph_memory/config/codec.py` 的 JSON-only codec。
2. `graph_memory/config/converter.py` 的手写 dataclass/union structuring。
3. `graph_memory/config/patches.py` 的 profile/CLI deep merge。
4. `graph_memory/config/loader.py` 的 argparse provided-option 扫描和 layer merge。
5. `StageConfigSpec` 中 parser factory、config path、profile patch、registry patch、CLI patch 组合体系。
6. `_set_provided_options()`、`build_effective_config()`、`_profile_patch()`、
   `_validate_method_config_profiles()`、`_require_all_dataclass_fields()` 等配置结构校验。
7. experiment JSON 内的 `defaults`、`default_profile`、`profiles` 手工协议。
8. method JSON 内重复的 `default_profile`、`profiles` 手工协议。
9. experiment config 中 method -> JSON path 的 `method_configs` 映射。
10. 所有旧 JSON config；迁移完成后不保留 JSON compatibility loader。
11. `effective_config.json`；Hydra 已写 resolved YAML 和 overrides。
12. 当前 generated stage JSON；替换为 Pydantic dump 的 resolved stage YAML。
13. scripts 中科学参数的 argparse defaults 和大批独立 flags。
14. `GenericStageConfig(args: dict[str, object])` 这种只是包住 argparse namespace 的假类型。
15. config/workflow 中大量 `dict[str, Any]` 和通过 `get()` 猜字段存在性的代码。
16. 为 BM25/Dense 等方法携带无意义 optional graph/checkpoint/selected-config 字段的 wide config。
17. 只为旧 CLI/config 形态存在的 compatibility tests 和 adapters；本次明确不做兼容层。

### 可以简化

1. `scripts/workflow/stage_configs.py`：大部分手工 dict/dataclass 投影和 JSON 写入删除，剩余
   artifact binding 进入 typed `StageInvocation` builder。
2. `scripts/workflow/manifest.py`：profile merge、默认值、config discovery、artifact path 展开删除；
   run state/layout 分到对应模块。
3. `scripts/experiment.py`：subcommand argparse 删除，变成几个极薄 Hydra 文件入口。
4. config discovery/listing：由 Hydra config groups + `experiment/inspect.py` 展示，不再扫描三种 JSON
   目录并猜 kind。
5. config dataclass：外部配置模型改为 Pydantic；稳定 domain dataclass 保留。

### 不应删除

- method registry 的 lifecycle、dependency、artifact kind、seed method、train dependencies；
- workflow stage order 和 dependency validation；
- artifact/data/domain validators；
- train/retrieve registry 里真正的 runtime builder；
- script/domain 边界；
- output layout 的单一 owner；
- status/cache 的 artifact verification。

## 细节冲突与锁定决策

| 冲突 | 当前事实 | 锁定决策 |
|---|---|---|
| Python 版本 | `.python-version` 是 3.12，服务器是 3.10；现有测试主要检查声明和部分 AST | 在真实 Python 3.10 环境运行完整 smoke、Hydra composition、Pydantic、MLflow 和类型测试 |
| multirun 同名 | 一个 name 对应多个 Hydra jobs | 注册唯一 concise formatter；Hydra 与 `RunLayout` 共同产生 `<job-num>_<varying-leaf>=<value>` |
| single/multirun 复用 name | 两种 mode 的根目录语义不同 | `RunState.mode` 固定；mode 不匹配时拒绝，必须 reset 或改名 |
| `dataset=2wiki` | 当前 dataset 字符串还伴随多组 raw/split/adapter 配置 | dataset 必须是 config group，不能只覆写一个 scalar |
| profile | dataset count 和 method training profile 分散在多份 JSON | dataset 提供已验证 logical-window capacity，profile 提供 `fixed`/`all_available` policy；typed resolver 合成并对越界 fail fast |
| 全局 seed | 目前 prepare、pairs、R-GCN、Dense-FT 的 seed 行为不同 | 统一 `seed_everything`；本轮不启用 strict deterministic algorithms，训练比较使用声明容差 |
| 全局 device | CPU method 不消费 device；checkpoint retrieval 从 method config 取 device | root 唯一 device，只注入需要它的 discriminated config；不 fallback |
| Pydantic validation | 当前很多 validate 是 artifact/domain validation | 只删除 config-shape validation，保留所有科学和 artifact 语义校验 |
| optional 泛滥 | wide IO 让不同 method 共享无关 optional fields | method-specific discriminated unions，让依赖字段在对应 variant 中必填 |
| Hydra 嵌套 | experiment 需要启动低层 scripts | 只有 experiment 做 composition；script 接受 resolved YAML |
| Hydra cwd | 相对路径大量存在，Hydra 会创建 output dir，历史版本 chdir 行为不同 | 显式 `hydra.job.chdir=false`，RunLayout 产出规范绝对路径供 subprocess 使用 |
| method 动态选择 | 临时 `methods=[...]` 和隐藏 Dense-FT dependency 都需要 config | 用 Hydra multi-select 一次组合八份 `method_configs`，普通 `methods` 列表只决定执行；不使用 `method@...` |
| MLflow 存储 | 每个 named run 一套 store 会失去跨 run 比较 | 全局 SQLite metadata + artifact root；BasicLauncher 顺序执行，workflow artifact 仍在 run dir |
| 指标事实源 | direct script 无 parent context 仍必须独立工作 | 本地 metric JSON/JSONL 是事实源，MLflow 只镜像，不删除 `train_metrics.jsonl` |
| 大 artifact | `log_artifact` 会形成 MLflow 管理的副本 | 大数据只登记 path/size/kind/role；精选 config/table/summary 才上传，不生成 comparison plot |
| cache correctness | MLflow FINISHED 不证明本地输出匹配输入 | 保留 artifact + typed run summary 检查和 completed-prefix resume，不新增 cache 能力 |
| code/raw 原地变化 | 当前 summary 未必识别同路径内容变化 | 接受现有边界；使用新 run name、reset 或 `cache.enabled=false` 明确重跑 |
| config 变化恢复 | MLflow param 和旧 artifact 不应在同 run 被改写 | 新旧 resolved Pydantic config 直接比较；不同即改名或 reset，绝不混用 |
| Dense-FT seeded R-GCN | public method selection 隐含 Dense-FT pairs/train dependency | dependency 留在 typed method registry，不写到 YAML `_target_` |
| Memory Stream | external importance、Hotpot-only source、coverage cap 是特殊真实约束 | 建模为独立 typed workflow/capability，不用 optional/fallback 隐藏 |
| 调参与泄漏 | tune 使用 dev labels，test retrieval 必须使用固定 selected config | stage contract 显式区分 dev tune/test retrieve，run summary 记录并核对 selected config path |
| Ablation | variant 可从不同 stage 失效，并复用 alias | 保留 changed-dimension invalidation 模型；Hydra multirun 不能代替 domain ablation |
| Ablation CLI | baseline alias 与 executable variants 容易混淆 | `variants=[]|all|[wo_*]`；`full_rgcn` 自动作为 alias，不接受为执行项 |
| checkpoint 形态 | R-GCN 是 file，Dense-FT 是 directory | typed `ArtifactRef(kind=file|directory)`，cache/status 依 kind 验证 |
| force | 当前 `--force` 会删除整个 run | 改成单独 reset command；普通 run 永不隐式清理 |
| 默认等价 | GPU 训练可能非严格 bitwise deterministic | 先比 plan/resolved config/artifact schema，再对 deterministic 方法做 exact，对训练指标设预先声明容差 |
| SQLite 并发 | 高并发 sweep 会增加锁冲突 | 本轮只支持 BasicLauncher 顺序 multirun；并行 launcher 和其他 backend 不在本轮范围 |

## 新实现的验收矩阵

### 配置与 CLI

- `name` 缺失时 Hydra/OmegaConf fail fast。
- unknown config key 被 Pydantic 拒绝。
- `dataset=2wiki` 替换完整 dataset group。
- `profile=cloud-full` 对 train/test 读取所选 dataset capacity、对 dev 固定 500，并同时影响 trainable method 设置。
- profile 的 `fixed.count` 超过 dataset split capacity 时在执行前失败。
- `--config-name=2wiki_tiny` 选择特殊 root preset；inspect configs 可发现它，但不存在 recipe/recipe list。
- `methods='[bm25,dense]'` 只执行两个 public methods。
- resolved config 中八份 `method_configs` 位于各自 key，不出现 `method@method_configs.*`。
- `method_configs.dense_ft.train.trainer.learning_rate=3e-5` 到达且只影响 Dense-FT trainer。
- `ablation.variants=[]|all|[wo_*]` 和 `ablation.only=true|false` 按本文列出的合法值解析。
- `seed=14` 到达 prepare/pairs/train 等所有随机消费者。
- `device=cpu|cuda|cuda:N` 到达所有 device consumer，不影响 BM25 config。
- resolved YAML 不含 `???`，没有 config code default 参与结果。
- 全局 Pydantic model 不启用 strict mode，但科学标量拒绝隐式字符串/bool coercion。

### Plan parity

- 新默认 plan 与当前 HotpotQA quick 的七个默认 methods 和 stage 依赖一致。
- `dense_ft_rgcn_graph_retriever` plan 明确先列 Dense-FT pairs/train。
- graph rerank retrieve 在 selected config 缺失时失败。
- train/retrieve 从中间 stage 开始时验证 pairs/checkpoint。
- 计划仍展示低层 script、stage、split/method/variant 和 resolved config path。
- HotpotQA、2Wiki、MuSiQue 的默认组合及保留的特殊 root presets 都有 plan snapshot/parity test。
- Memory Stream dedicated workflow 和 R-GCN ablation workflow 单独覆盖。

### Cache 与恢复

- artifact 存在、typed run summary 成功且 inputs/outputs/effective config 匹配时为 complete。
- 输出存在但 run summary 缺失、失败或不匹配时为 stale。
- 中断后跳过连续的 complete/alias 前缀，从第一个 missing/stale command 恢复后续执行。
- 保持当前 ordered-prefix 行为；不升级为逐 DAG node 独立 cache。
- ablation alias 和 invalidation boundary 与当前行为一致。
- `cache.enabled=false` 不剪 completed prefix，重跑所选 stage 范围并写普通 run summary。
- single-run/multirun mode 不匹配时拒绝复用同名目录。
- 不存在 hash、source fingerprint、artifact digest 或跨 run cache 验收项。

### MLflow

- 一个 Hydra job 对应一个 parent run。
- 每个 selected baseline/executable variant 对应一个稳定 child；shared stages 没有 child。
- fully cached baseline 仍从 authoritative local files 填充 child；exact resume 复用唯一 child。
- epoch metrics 使用正确 step。
- parent Overview、tuning candidate table、selected config、baseline final metrics 和 aggregate tables 可在 UI 找到。
- parent metrics 为空；baseline children 共享 `final.*` keys，并可直接 Compare Runs。
- MLflow 不复制默认禁止的大型 artifacts。
- run state 记录 MLflow ids；experiment run 的 MLflow 初始化或写入失败即失败，不静默丢 tracking。
- direct script 没有 parent context 时仍写完整本地 metric artifacts，不创建孤立 MLflow run。
- BasicLauncher 顺序 multirun 可完整写入统一 SQLite store。
- UI 可跨不同 name/multirun job 比较。

### 行为等价与实际 workflow

- direct stage scripts 只用 resolved YAML 仍可独立运行和测试。
- HotpotQA smoke 覆盖全部八个 registry methods；Memory Stream 可使用专用 fixture/artifact。
- 默认 HotpotQA quick 覆盖原计划要求的七个 methods，不含 Memory Stream。
- 2Wiki 和 MuSiQue 对当前支持的方法分别跑完整 workflow smoke。
- main/path/efficiency/ablation table schema 与必要语义保持。
- prediction schema、checkpoint metadata、failure case schema 不变。
- leakage tests、dataset projector tests、request-boundary tests 全部通过。

### 工程门槛

- 在真实 Python 3.10 下安装 lock 并运行验收，而不只是 AST 检查。
- 全量 pytest。
- Ruff。
- basedpyright error-level。
- strict OpenSpec validation。
- compileall。
- `git diff --check`。
- residual scan 确认旧 JSON config loader、旧 argparse defaults、重复 run-summary 样板、旧
  `scripts/workflow` 和兼容 adapter 已删除。

## 锁定的实施顺序

这应作为新的 OpenSpec change 执行，不应在一个无验收锚点的大提交中直接重写。

1. **冻结当前行为。** 保存各 dataset/config/method 的 plan snapshots、resolved configs、smoke
   artifacts 和可比 metrics。
2. **验证依赖。** 在 Python 3.10 上加入并锁定 Hydra/Pydantic/MLflow，做最小 composition、
   multirun output-dir、SQLite tracking spike。
3. **建立 Pydantic contracts 和 YAML groups。** 暂不改执行器，先证明所有当前 JSON 可以无损表达，
   并证明 root seed/device/profile 覆写正确。
4. **建立新的 experiment core。** 先实现 RunLayout、typed DAG、plan parity；此时仍可调用旧 scripts。
5. **迁移现有 status/resume。** 用 typed `StageRunSummary` 复现当前状态检查和 completed-prefix
   resume；不得加入 hash、逐节点或跨 run cache。
6. **逐 stage 迁移 script。** 每次先写 direct-script failing test，再把 CLI 收敛到 resolved YAML，
   删除该 stage 的旧默认值和 config loader 分支。
7. **接入 MLflow。** 记录 parent/child run、metrics、params 和精选 artifacts；验证中断恢复。
8. **迁移 aggregate/delivery。** 确保最终 CSV 和 mentor/report artifact 仍可独立交付。
9. **删除旧系统。** 一次性删除 JSON config、手写 converter/patch/loader、重复 run-summary 样板、
   `scripts/workflow`、旧 CLI tests 和所有 compatibility remnants；typed stage summaries 保留。
10. **跑完整 workflow 验收。** 不以 unit tests/typecheck 代替 `experiment/run.py` 的真实 smoke/quick
    workflow。

## 最终锁定方案

原计划的目标和技术方向批准，并锁定以下不可妥协的设计约束：

- Hydra 是 composition/CLI，不是 workflow engine；
- Pydantic 是 config boundary，不是 artifact/domain validator；
- MLflow 是 tracking UI，不是 cache truth；
- method dependency 只存在于 typed registry，不复制进 YAML `_target_`；
- dataset 定义 split window/capacity，profile 定义规模 policy；少数特殊组合只使用 Hydra root preset，
  不引入 recipe 概念；
- `method_configs` 使用 Hydra multi-select 组合，`methods` 普通列表决定执行；
- `runs/${name}` 是稳定的 single-run identity，日期不参与路径和 cache identity；
- single-run 和 multirun 使用不同 mode，multirun 使用 Hydra job num 加 concise varying-leaf suffix；
- cache 保持 artifact + typed run summary + completed-prefix resume，不增加任何能力；
- 大 artifact 不默认复制进 MLflow；
- 本地 metric artifacts 是事实源，MLflow 是 parent/baseline presentation mirror；
- root seed 统一所有 RNG 入口，本轮明确不启用 strict deterministic algorithms；
- direct script 只接受 resolved stage YAML；
- 不保留旧 JSON/argparse/config compatibility；
- 最终完成标准是三套 dataset、默认七方法、Memory Stream、ablation、resume 和 multirun 的真实
  workflow 验证，而不是配置文件成功加载。

至此本文没有留给实现阶段临时选择的阻塞性设计问题；任何需要改变上述唯一决策的发现，都必须先
更新本文和对应 OpenSpec change，不能在代码中加入第二条 fallback 路径。
