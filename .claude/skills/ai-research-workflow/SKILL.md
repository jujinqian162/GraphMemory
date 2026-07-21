---
name: ai-research-workflow
version: 2.0.0
description: "指导 Agent 为 AI 科研项目设计简单、可缓存、可追踪的实验工作流。默认使用 Hydra 管理配置、Prefect 编排和缓存阶段、MLflow 记录实验与持久化资产、PyTorch/Lightning 执行训练。强调单一最终方法、直观 Flow、准确缓存边界和最小基础设施复杂度。"
---

# AI 科研工作流设计技能

## 1. 目标

使用本技能时，优先实现一套**简单、直观、可复用**的 AI 科研工作流，而不是建设通用工作流平台。

默认技术分工：

- **Hydra**：组合实验配置与命令行覆盖。
- **Prefect Flow**：表达一次实验从输入到最终指标的控制流。
- **Prefect Task**：承载昂贵、可缓存、可重试的计算阶段。
- **Prefect Result Storage**：持久化 Task 返回的小型结果对象，使缓存可跨运行复用。
- **MLflow**：记录一次实验的完整配置、最终指标、报告和长期资产。
- **PyTorch / Lightning**：执行单个训练阶段内部的训练循环。

首要原则：

1. Flow 必须像普通科研代码一样容易阅读。
2. 缓存边界由 Task 的输入和输出自然表达。
3. 不自行实现缓存数据库、DAG 注册表、共享锁或通用 Planner 框架。
4. 一次实验默认只对应一个最终方法和一个顶层 MLflow run。
5. 可复用计算产物不按 run name 隔离。

---

## 2. 默认实验抽象

一次实验应当抽象为：

```text
实验配置
  ↓
选择一个最终方法
  ↓
执行该方法所需的阶段链
  ↓
得到最终评估结果
  ↓
记录到一个 MLflow run
```

默认关系：

```text
一个 Hydra job
= 一个 Prefect flow run
= 一个最终方法
= 一个顶层 MLflow run
```

`run name` 只是方便人类识别实验的显示标签。它可以用于：

- MLflow run name；
- Prefect flow-run name；
- 报告标题；
- 人工筛选 tag。

它不得影响任何计算 Task 的缓存身份，也不得决定长期资产路径。

### EXAMPLE：单一最终方法

```bash
python run.py dataset=<dataset-name> method=<method-name> name=<display-name>
```

这里的 `<dataset-name>`、`<method-name>` 和 `<display-name>` 都是抽象占位符。

### EXAMPLE：某个图检索项目

```bash
python run.py dataset=musique method=dense_ft_rgcn name=test
```

这是具体项目示例，不应被复制为通用 Skill 的固定命名。

---

## 3. Flow 的设计原则

### 3.1 Flow 应直接表达实验逻辑

Flow 负责：

- 根据最终方法选择一条阶段链；
- 调用或提交对应 Task；
- 等待最终结果；
- 将最终结果记录到当前 MLflow run。

优先使用直接、可读的普通 Python 分支：

```python
@flow
def run_experiment(cfg):
    prepared = prepare_dataset(cfg.dataset, cfg.prepare)

    if cfg.method == "method_a":
        model = train_method_a(prepared, cfg.method_a)
        result = evaluate_method_a(prepared, model, cfg.evaluation)

    elif cfg.method == "method_b":
        base_model = train_base_model(prepared, cfg.base_model)
        model = train_method_b(
            prepared,
            base_model,
            cfg.method_b,
        )
        result = evaluate_method_b(prepared, model, cfg.evaluation)

    else:
        raise ValueError(f"Unsupported method: {cfg.method}")

    log_experiment_result(cfg, result)
    return result
```

这种写法优先于：

- 自定义 DAG 注册表；
- 通用节点解析器；
- Future 去重注册表；
- 通用 Planner 框架；
- 为少量方法建立复杂插件系统。

当方法数量仍然可读时，显式分支比高度抽象的依赖解析更适合科研代码。

### 3.2 可以在分支中显式调用依赖阶段

某个最终方法依赖基础方法时，直接在该分支中先完成基础阶段：

```python
elif cfg.method == "extended_method":
    base_model = train_base_model(
        prepared,
        cfg.base_model,
    )
    extended_model = train_extended_model(
        prepared,
        base_model,
        cfg.extended_method,
    )
    result = evaluate_extended_method(
        prepared,
        extended_model,
        cfg.evaluation,
    )
```

不要仅为了“自动解析依赖”引入额外框架。Prefect 缓存会负责跨实验复用 `train_base_model` 的结果。

### EXAMPLE：具体方法链

```python
elif cfg.method == "dense_ft_rgcn":
    train_pairs = build_train_pairs(
        prepared,
        cfg.train_pairs,
    )
    dense_ft = train_dense_ft(
        train_pairs,
        cfg.dense_ft,
    )
    rgcn = train_rgcn(
        prepared,
        dense_ft,
        cfg.rgcn,
    )
    result = evaluate_rgcn(
        prepared,
        rgcn,
        cfg.evaluation,
    )
```

这里的 `dense_ft_rgcn`、`train_pairs` 和 `rgcn` 是项目示例，不是通用约束。

---

## 4. Task 的设计原则

### 4.1 哪些阶段应成为 Task

把满足以下条件的阶段定义为 Prefect Task：

- 计算耗时明显；
- 结果未来可能被复用；
- 失败后值得单独重试；
- 希望独立观察状态和日志。

典型抽象阶段：

```text
prepare_data
build_training_data
build_features
build_index
train_base_model
train_derived_model
generate_predictions
evaluate_model
```

不要把以下细粒度操作做成 Prefect Task：

- batch；
- epoch；
- optimizer step；
- 单个模型层；
- 毫秒级配置转换。

这些仍由普通 Python、PyTorch 或 Lightning 管理。

### 4.2 Task 签名定义缓存边界

每个 Task 只接收真正影响其结果的输入。

正确的抽象形式：

```python
train_base_model(
    training_data,
    base_model_config,
)

train_derived_model(
    prepared_data,
    base_model,
    derived_model_config,
)
```

避免：

```python
train_base_model(
    training_data,
    full_experiment_config,
)
```

完整配置中与该阶段无关的字段会污染缓存键，使上游结果因为下游配置变化而错误失效。

Hydra 可以维护完整配置树，但传入 Task 时应当只传该阶段需要的配置切片。

### 4.3 默认跨运行缓存

对可复用 Task 开启结果持久化和跨运行缓存：

```python
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE

CACHE_POLICY = INPUTS + TASK_SOURCE


@task(
    persist_result=True,
    cache_policy=CACHE_POLICY,
)
def prepare_data(...):
    ...
```

缓存的基本语义：

```text
相同 Task 实现
+ 相同有效输入
→ 复用历史返回结果
```

不要把以下信息传给可缓存 Task：

- run name；
- 当前 MLflow run ID；
- 当前 Prefect flow-run ID；
- 与该 Task 输出无关的下游配置。

### 4.4 实现变化必须能使缓存失效

如果 Task 只是调用普通 helper，而 helper 代码变化未被缓存策略感知，应采用一种简单、明确的版本策略，例如：

```python
@task(
    persist_result=True,
    cache_policy=CACHE_POLICY,
)
def prepare_data(
    dataset,
    config,
    implementation_version="prepare-v2",
):
    ...
```

只在确实需要时加入实现版本。不要把整个 Git commit 无差别传给所有 Task。

---

## 5. 缓存行为的通用不变量

Agent 实现后，应验证以下抽象行为：

1. 完全相同的有效输入再次运行时，应命中缓存。
2. 只修改某个下游阶段的配置时，不应重算无关上游阶段。
3. 修改上游阶段配置时，应重算该阶段及其下游。
4. 修改评估配置时，不应重新训练模型。
5. 修改仅用于展示的 run name 时，不应使任何计算缓存失效。

### EXAMPLE：派生模型依赖基础模型

假设：

```text
prepare
  ↓
train_base
  ↓
train_extension
  ↓
evaluate
```

若只修改 `train_extension` 的配置，应当：

```text
prepare          Cached
train_base       Cached
train_extension  Recomputed
evaluate         Recomputed
```

### EXAMPLE：具体项目

在某个项目中，若 `RGCN` 依赖 `dense-ft`，只修改 RGCN 参数时，`dense-ft` 应命中缓存。

这些示例用于说明通用不变量，不应成为 Skill 中写死的方法名或测试脚本。

---

## 6. 资产与存储

### 6.1 不按 run name 保存计算资产

不要使用以下模式保存未来可能被下游读取的结果：

```text
runs/<run-name>/prepared/
runs/<run-name>/graph/
runs/<run-name>/checkpoint.pt
```

可复用计算资产属于其输入、配置和实现，而不属于某次展示命名的实验。

`runs/<run-name>` 可以：

- 完全不存在；或
- 只保存不会再进入程序的 summary、报告和展示日志。

### 6.2 Task 使用临时工作目录

需要先写文件的 Task 应使用临时 workspace：

```text
临时 workspace
  ↓
执行计算
  ↓
校验最终产物
  ↓
上传长期存储
  ↓
返回资产引用
  ↓
清理临时 workspace
```

临时 workspace 可以包含：

- 下载碎片；
- 解压文件；
- 训练中的临时 checkpoint；
- profiler 输出；
- 当前 attempt 的调试文件。

这些内容默认不作为跨实验缓存资产。

### 6.3 Prefect 缓存小型结果对象

Prefect Result Storage 优先保存小型、稳定、可序列化的结果对象，而不是直接保存超大数据目录。

抽象形式：

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    kind: str
    digest: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageResult:
    artifact: ArtifactRef | None
    metrics: dict[str, float]
    metadata: dict[str, Any]
```

Task 返回 `StageResult` 或其他领域结果对象。真正的大文件保存在：

- MLflow Artifact Store；或
- 共享文件系统；或
- 对象存储。

### 6.4 何时使用 MLflow 保存资产

第一版可以直接把以下长期产物放入 MLflow Artifact Store：

- prepared data；
- features / graph / index；
- 最终 checkpoint；
- predictions；
- evaluation tables；
- reports。

当资产过大、需要随机访问、mmap、分片读取或频繁跨节点共享时，将实际内容迁移到共享文件系统或对象存储，MLflow 只记录 URI、digest 和 manifest。

### 6.5 最终产物与恢复状态分离

区分：

```text
最终可复用产物
    例如 best model、prepared data、index

一次执行的恢复状态
    例如 last checkpoint、optimizer state、当前 epoch
```

最终产物可以成为 Task 返回的长期资产。

恢复状态只服务于当前训练 attempt；是否上传和保留由训练可靠性需求决定，不应默认纳入跨实验缓存设计。

---

## 7. MLflow 记录策略

### 7.1 一个最终方法对应一个顶层 run

普通实验默认只创建一个顶层 MLflow run。

不要因为 Flow 内部有多个 Task 就自动创建 child runs。Child run 不会自动把指标汇总到 parent，容易降低实验比较体验。

只有在 trial、fold、replicate 等天然存在多次同类运行时，才考虑 parent/child；即使使用，也必须显式把需要比较的汇总指标记录到 parent。

### 7.2 每次实验都必须记录完整可比较信息

每个顶层 MLflow run 至少记录：

- resolved 实验配置；
- 最终方法和数据集标识；
- 最终评估指标；
- 关键中间结果摘要；
- 使用的资产 URI 或 ID；
- 最终报告或汇总表。

这些信息无论来自真实计算还是 Prefect 缓存，都必须出现在当前 run 中。

### 7.3 将实验级 logging 放在缓存 Task 外

缓存命中时，Task 函数体不会执行。因此用于横向比较实验的 MLflow logging 不应只写在 Task 内部。

推荐模式：

```python
@flow
def run_experiment(cfg):
    with mlflow.start_run(run_name=cfg.name):
        result = execute_method(cfg)

        mlflow.log_params(flatten_config(cfg))
        mlflow.log_metrics(result.metrics)
        log_result_artifacts(result)

        return result
```

Task 返回足够完整的结果，使 Flow 在缓存命中时仍能记录全部最终指标。

### 7.4 Task 内只记录真实生产过程

只有在 Task 真正执行时才有意义的信息，可以保留在 Task 内：

- epoch 级训练曲线；
- 学习率变化；
- 真实训练耗时；
- profiler；
- 实际生成 checkpoint 的过程信息。

缓存复用时，不应把过去的资源耗时伪装成本次实验的真实执行耗时。

第一版优先使用简单的 MLflow fluent API。只有在并发写入、跨进程控制或明确需要指定其他 run 时，才引入 `MlflowClient`。

---

## 8. `@materialize` 的定位

`@materialize` 是带资产语义的 Prefect Task。

它额外帮助 Prefect 记录：

- 某个逻辑资产被生成或更新；
- 资产 materialization 历史；
- 上下游资产 lineage；
- 资产状态和附加元数据。

它不会自动：

- 保存文件；
- 上传 MLflow artifact；
- 启用缓存；
- 验证外部 URI；
- 提供 DVC 式文件恢复。

因此：

```text
@task / @materialize
    定义计算执行语义

persist_result + cache_policy
    定义返回结果缓存语义

MLflow / 文件系统 / 对象存储
    定义实际文件存储语义
```

第一版可以全部使用 `@task`。只有当资产目录和 lineage 对项目确实有帮助时，再把重要长期资产节点改为 `@materialize`。

不要为了“看起来完整”强制使用 `@materialize`。

---

## 9. 推荐的抽象代码骨架

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE


CACHE_POLICY = INPUTS + TASK_SOURCE


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    kind: str


@dataclass(frozen=True)
class StageResult:
    artifact: ArtifactRef | None
    metrics: dict[str, float]
    metadata: dict[str, Any]


@task(persist_result=True, cache_policy=CACHE_POLICY)
def prepare_data(
    dataset_config: dict[str, Any],
    prepare_config: dict[str, Any],
) -> StageResult:
    ...


@task(persist_result=True, cache_policy=CACHE_POLICY)
def train_base_model(
    prepared: StageResult,
    model_config: dict[str, Any],
) -> StageResult:
    ...


@task(persist_result=True, cache_policy=CACHE_POLICY)
def train_derived_model(
    prepared: StageResult,
    base_model: StageResult,
    model_config: dict[str, Any],
) -> StageResult:
    ...


@task(persist_result=True, cache_policy=CACHE_POLICY)
def evaluate_model(
    prepared: StageResult,
    model: StageResult,
    evaluation_config: dict[str, Any],
) -> StageResult:
    ...


@flow
def run_experiment(cfg: Any) -> StageResult:
    with mlflow.start_run(run_name=cfg.name):
        prepared = prepare_data(
            cfg.dataset,
            cfg.prepare,
        )

        if cfg.method == "base_method":
            model = train_base_model(
                prepared,
                cfg.base_method,
            )

        elif cfg.method == "derived_method":
            base_model = train_base_model(
                prepared,
                cfg.base_method,
            )
            model = train_derived_model(
                prepared,
                base_model,
                cfg.derived_method,
            )

        else:
            raise ValueError(
                f"Unsupported method: {cfg.method}"
            )

        result = evaluate_model(
            prepared,
            model,
            cfg.evaluation,
        )

        mlflow.log_params(flatten_config(cfg))
        mlflow.log_metrics(result.metrics)
        log_result_artifacts(result)

        return result
```

该代码是抽象骨架：

- `base_method` 和 `derived_method` 是占位符；
- 项目应替换为自己的领域阶段；
- 不要照搬无关字段；
- 优先保持 Flow 易读。

---

## 10. 实现策略

Agent 在实现或重构项目时，按以下顺序推进：

1. 先把一次实验收敛为一个最终方法。
2. 用直观 `if / elif` 写出每个最终方法的阶段链。
3. 把昂贵且可复用的阶段包装成 Prefect Task。
4. 为 Task 传入准确的配置切片。
5. 开启结果持久化和跨运行缓存。
6. 让 Task 返回完整的结果对象或资产引用。
7. 在 Flow 外层统一记录当前 MLflow run 的最终指标。
8. 最后再决定是否需要 `@materialize`、共享对象存储或更复杂执行基础设施。

不要在第一版引入：

- 通用 Planner 框架；
- 节点注册表；
- 分布式缓存锁；
- 多层 MLflow child runs；
- 自定义内容寻址存储；
- 复杂并行调度；
- 无实际需求的资产抽象。

---

## 11. 设计审查

审查一个工作流时，只检查最关键的问题：

1. Flow 是否能直接读出某个最终方法会执行哪些阶段？
2. 每个昂贵阶段是否是独立 Task？
3. Task 是否只接收真正影响其结果的配置？
4. 下游配置变化是否会错误地使上游缓存失效？
5. 可复用资产是否脱离 run name 存储？
6. Task 返回值是否足以在缓存命中时继续执行下游？
7. 当前 MLflow run 是否总能得到完整最终指标？
8. 是否引入了超出当前需求的基础设施复杂度？

当“自动化程度”和“代码可读性”冲突时，优先选择可读性。

---

## 12. 版本敏感性

Prefect、Hydra、MLflow 和 Lightning 的 API 会演进。实现具体 API 时应查询当前官方文档，尤其是：

- Prefect cache policy 与 Result Storage；
- Prefect Flow / Task 调用语义；
- Prefect `@materialize` 与 Asset API；
- MLflow artifact 与 logging API；
- Hydra 配置组合与启动方式。

API 可以变化，但本技能的核心边界保持不变：

```text
Hydra 管配置
Prefect Flow 管易读控制流
Prefect Task 管可缓存计算阶段
MLflow 管实验记录和长期资产
训练框架管训练内部
```
