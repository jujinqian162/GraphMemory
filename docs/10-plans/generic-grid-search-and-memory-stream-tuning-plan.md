# 通用 Grid Search 与 Memory Stream 调参实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 使用
> `superpowers:subagent-driven-development`（推荐）或
> `superpowers:executing-plans` 按任务实施。所有生产代码必须遵循 TDD：
> 先写失败测试，确认失败原因正确，再写最小实现。

**状态：** 已按本计划实施。Task 1-9 已完成，最终验证记录见对应提交与运行日志。

**Current note:** `scripts/tune_memory_stream.py` is now dataset-aware. LongMemEval V1 workflows pass `--dataset longmemeval` and do not pass `--importance` unless a non-gold external artifact is explicitly configured; when no artifact is supplied, tuning uses request-owned importance maps and phase-1 LongMemEval configs fix `importance_weight` to `0.0`.

**目标：** 抽取一个不感知 Graph Rerank、Memory Stream、检索任务或指标名称的
通用 Grid Search 内核；让 Graph Rerank 和 Memory Stream 分别通过明确的
method-specific adapter 构建候选、复用昂贵信号、评估候选并序列化最终配置。

**架构：** 通用层只负责“展开参数组合、逐个调用评价函数、按注入的 selection
key 确定性选优”。检索调参层负责把 `MetricRow` 转换为 selection key。Graph
Rerank 和 Memory Stream 各自负责输入验证、领域配置解析、信号缓存、候选运行和
输出配置，不通过继承共享 method-specific 行为。

**技术栈：** Python 3.10、标准库 `dataclasses` / `itertools` / `typing`、
必要时使用现有 `typing-extensions` 兼容类型、现有 retrieval/evaluation
contracts、`pytest`、JSON 配置与 run summary。

---

## 0. 实施前置约束

当前 `phase2-implement` 工作树中已经存在未提交的 Memory Stream method、registry、
workflow 和测试修改。本计划实施时必须：

1. 将这些修改视为用户现有工作，不得回退、覆盖或用旧版本文件替换。
2. 修改重叠文件前先读取完整 diff，基于当前内容继续演进。
3. 每个任务提交前使用 `git diff -- <本任务文件>` 审查实际范围。
4. 禁止使用 `git add tests`、`git add graph_memory` 等宽泛暂存命令；必须逐文件
   暂存本任务实际修改。
5. 若现有 Memory Stream 实现尚未独立提交，优先在同一工作树连续完成并统一验证；
   不为本计划强行创建 worktree 或移动用户修改。

## 1. 问题定义

当前 `graph_memory/retrieval/tuning/service.py` 同时承担以下职责：

1. Graph Rerank method 校验。
2. Graph Rerank grid 构建。
3. dense/BM25 seed score 缓存。
4. 候选运行和 retrieval evaluation。
5. objective 计算和最优候选选择。
6. Graph Rerank config 序列化。

其中只有第 4 项的循环骨架和第 5 项的“根据 key 选最大值”是跨 method
可复用的。其余职责都依赖具体 method、输入 artifact 或领域 config。

Memory Stream 的最终分数为：

```python
score = (
    relevance_weight * relevance[node_id]
    + recency_weight * recency[node_id]
    + importance_weight * importance[node_id]
)
```

需要搜索的字段完全由 search-space JSON 决定。代码不得假设
`relevance_weight` 固定为 `1.0`，也不得自动归一化、约束权重和为 `1.0`，或删除
同比缩放的候选。若只希望某字段固定，配置中将该字段写成单元素数组即可。

本次必须避免的直接实现是：复制 `tune_graph_rerank()`，再创建一个结构几乎相同的
`tune_memory_stream()` 循环。这样会继续复制候选遍历、空 grid 校验、选择规则和
确定性 tie-break 语义。

## 2. 方案比较

### 方案 A：函数回调驱动的通用 `GridSearchRunner`，推荐

通用 runner 接收：

- 已经解析成领域 config 的候选序列。
- `evaluate(candidate) -> evaluation` 回调。
- `selection_key(evaluation) -> tuple[...]` 回调。

runner 返回：

- 按输入顺序保存的所有 `EvaluatedCandidate`。
- 被选中的 `EvaluatedCandidate`。

优点：

- 通用层没有 Graph、Memory Stream、retrieval、metric 或 JSON 依赖。
- method adapter 可以自由决定如何缓存 dense、graph index、importance。
- selection policy 可注入，未来其他实验不必复用当前 retrieval objective。
- 单元测试可以只使用整数或字符串候选，不需要构造 retrieval fixture。

缺点：

- Graph 和 Memory Stream 仍各自需要一个薄 service；这是有意保留的领域边界，
  不是重复。

### 方案 B：抽象基类 `BaseTuner` + 子类覆写 hook，不采用

例如定义 `build_candidates()`、`prepare()`、`evaluate_candidate()`、
`serialize_config()` 等抽象方法。

不采用原因：

- Graph 与 Memory Stream 的输入和缓存结构差异较大，基类会逐渐积累可选 hook。
- CLI、artifact IO 和 method runtime 容易被拉入同一个对象生命周期。
- 继承关系隐藏依赖，测试必须构造完整子类，边界不如显式函数参数清楚。

### 方案 C：只创建统一 CLI，根据 `--method` 写大型分支，不采用

统一 CLI 本身不能形成可复用抽象，只会把当前 `tune_graph_rerank.py` 的分支扩展成
更大的 procedural script。脚本层应只做 IO、日志和 run summary，不应成为调参
核心。

## 3. 最终目录与命名

### 3.1 通用调参层

新增：

```text
graph_memory/tuning/
  __init__.py
  grid_search.py
```

`graph_memory/tuning/grid_search.py` 只定义：

| 名称 | 职责 |
|---|---|
| `ParameterGrid` | 保存参数候选数组和固定字段，按确定性字段顺序展开笛卡尔积 |
| `EvaluatedCandidate[ConfigT, EvaluationT]` | 将一个领域 config 与评价结果绑定 |
| `GridSearchResult[ConfigT, EvaluationT]` | 保存 selected candidate 和全部 candidates |
| `GridSearchRunner[ConfigT, EvaluationT, KeyT]` | 遍历候选、调用 evaluator、按 selection key 选优 |

这里使用 `GridSearchRunner`，不使用 `Tuner`：

- runner 只执行搜索，不知道“训练”或“检索调参”的含义。
- `tune_graph_rerank()` 和 `tune_memory_stream()` 才是领域调参 service。

### 3.2 Retrieval 调参层

调整为：

```text
graph_memory/retrieval/tuning/
  __init__.py
  selection.py
  seed_scores.py
  graph_rerank_grid.py
  graph_rerank.py
  memory_stream_grid.py
  memory_stream.py
```

| 文件 | 唯一职责 |
|---|---|
| `selection.py` | 将一个 retrieval `MetricRow` 转换为确定性 selection key |
| `seed_scores.py` | 一次计算并保存 seed ranker 的全 task score 和等价 latency |
| `graph_rerank_grid.py` | 解析 Graph Rerank search-space，并构造 `GraphRerankConfig` |
| `graph_rerank.py` | Graph 输入检查、缓存复用、候选执行和评价 |
| `memory_stream_grid.py` | 解析 Memory Stream search-space，并构造 `MemoryStreamScoringConfig` |
| `memory_stream.py` | importance 选择、Memory Stream 信号缓存、候选执行和评价 |

现有含义模糊的文件迁移：

```text
graph_memory/retrieval/tuning/grid.py
  -> graph_memory/retrieval/tuning/graph_rerank_grid.py

graph_memory/retrieval/tuning/initial_scores.py
  -> graph_memory/retrieval/tuning/seed_scores.py

graph_memory/retrieval/tuning/service.py
  -> graph_memory/retrieval/tuning/graph_rerank.py
```

不保留 `grid.py`、`service.py` 这种离开目录上下文后无法判断领域的名字。

### 3.3 Memory Stream 领域配置

新增：

```text
graph_memory/retrieval/methods/memory_stream/config.py
```

定义：

```python
@dataclass(frozen=True)
class MemoryStreamScoringConfig:
    relevance_weight: float = 1.0
    recency_weight: float = 0.0
    importance_weight: float = 0.01
    recency_decay: float = 0.99
```

命名为 `MemoryStreamScoringConfig`，而不是：

- `MemoryStreamConfig`：范围过大，容易与 encoder、top-k、artifact path 混淆。
- `MemoryStreamTuningConfig`：选出的配置还会在正式 retrieval 中使用，不只属于 tuning。
- `MemoryStreamWeights`：无法表达 `recency_decay`。

`MemoryStreamRetrievalSettings` 继续拥有 encoder、top-k 和 test cap，但改为组合：

```python
@dataclass(frozen=True)
class MemoryStreamRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    scoring: MemoryStreamScoringConfig = field(
        default_factory=MemoryStreamScoringConfig
    )
    capped_test_count: int | None = None
```

`MemoryStreamMethod` 同样接收 `scoring: MemoryStreamScoringConfig`，不再分别接收
`MemoryStreamWeights` 和 `recency_decay`。`MemoryStreamWeights` 可删除，避免两套
近似配置类型互相转换。

## 4. 通用 Grid Search 契约

### 4.1 `ParameterGrid`

建议接口：

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterGrid:
    parameters: Mapping[str, Sequence[object]]
    fixed: Mapping[str, object]

    def expand(self) -> list[dict[str, object]]:
        ...
```

规则：

1. `parameters` 的每个 value 必须是非空 sequence。
2. `str`、`bytes` 不视为候选 sequence。
3. `parameters` 与 `fixed` 不得有重名字段。
4. 按 `parameters` 的 insertion order 生成笛卡尔积。
5. 每个结果先写入 fixed fields，再写入当前 parameter values。
6. 通用层不判断数字范围、字段名或字段之间的关系。
7. 候选值保持原值，不做 float coercion、去重或权重归一化。

Graph adapter 会将 `neighbor_type_weights` 放入 `fixed`；其余 list 字段放入
`parameters`。Memory Stream adapter 当前四个字段全部放入 `parameters`。若未来
search-space schema 增加固定字段，不需要修改 runner。

### 4.2 `GridSearchRunner`

建议接口：

```python
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

ConfigT = TypeVar("ConfigT")
EvaluationT = TypeVar("EvaluationT")
KeyT = TypeVar("KeyT")


@dataclass(frozen=True)
class EvaluatedCandidate(Generic[ConfigT, EvaluationT]):
    config: ConfigT
    evaluation: EvaluationT


@dataclass(frozen=True)
class GridSearchResult(Generic[ConfigT, EvaluationT]):
    selected: EvaluatedCandidate[ConfigT, EvaluationT]
    candidates: list[EvaluatedCandidate[ConfigT, EvaluationT]]


@dataclass(frozen=True)
class GridSearchRunner(Generic[ConfigT, EvaluationT, KeyT]):
    selection_key: Callable[[EvaluationT], KeyT]

    def run(
        self,
        candidates: Iterable[ConfigT],
        evaluate: Callable[[ConfigT], EvaluationT],
    ) -> GridSearchResult[ConfigT, EvaluationT]:
        ...
```

规则：

1. 候选 iterable 只 materialize 一次。
2. 空候选立即抛出 `ValueError("Grid search requires at least one candidate.")`。
3. 每个候选严格评价一次。
4. `candidates` 保持输入顺序，用于输出和复现。
5. 以 `max(..., key=selection_key)` 选择最大 key。
6. selection key 完全相同时，选择输入顺序中最早的候选。
7. runner 不捕获 evaluator 异常；失败候选导致本次 run 失败，防止静默产生不完整
   比较。
8. runner 不写文件、不计时、不记录日志、不生成 run summary。

## 5. Retrieval Selection Policy

当前 Graph Rerank 的 objective 保持不变：

```python
def retrieval_tuning_objective(row: MetricRow) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )
```

`selection.py` 提供：

```python
def retrieval_candidate_key(
    row: MetricRow,
) -> tuple[float, float, float, float]:
    return (
        retrieval_tuning_objective(row),
        float(row.get("Full Support@10", 0.0)),
        -float(row.get("Retrieval Latency / Query", 0.0)),
        -float(row.get("Avg Retrieved Edges", 0.0)),
    )
```

Graph Rerank 和 Memory Stream 默认共享该 policy，原因是二者都在相同 evidence
retrieval 指标表上选择 dev config。共享的是 selection policy，不是 method
执行逻辑。

若未来某 baseline 使用不同 objective，应向 `GridSearchRunner` 注入另一个 key，
而不是在通用 runner 中添加 method 分支。

## 6. Seed Score Cache 边界

将 `InitialScoreCache` 重命名为：

```python
@dataclass(frozen=True)
class SeedScoreCache:
    scores_by_task_id: dict[TaskId, dict[NodeId, float]]
    latency_ms_by_task_id: dict[TaskId, float]
```

提供：

```python
def precompute_seed_score_cache(
    *,
    seed_method: RetrievalMethodId,
    task_inputs: list[MemoryTaskInput],
    dense_runtime: DenseRuntime,
) -> SeedScoreCache:
    ...
```

边界要求：

1. cache 函数显式接收 `seed_method`，不再通过 outer method 反查 seed。
2. 它只认识 BM25/dense seed ranker contract，不认识 Graph 或 Memory Stream。
3. dense bulk ranking 继续按 task group 批量执行。
4. cache 保存一次真实 seed 计算的 task-level amortized latency。
5. 每个候选的等价 retrieval latency =
   `cached_seed_latency + candidate_specific_ranking_latency`。
6. grid search 的 wall-clock 总耗时单独写入 run summary，不混入 retrieval metric。

## 7. Graph Rerank Adapter

`tune_graph_rerank()` 的外部 callable 和 `scripts/tune_graph_rerank.py` CLI 保持兼容。

内部流程：

```text
validate graph-rerank method
  -> parse/build GraphRerankConfig candidates
  -> resolve seed method from method registry
  -> precompute SeedScoreCache once
  -> create GridSearchRunner(retrieval_candidate_key)
  -> evaluate each config with run_graph_rerank_from_seed_score_cache
  -> evaluate_results
  -> return selected GraphRerankConfigRecord + flattened candidate rows
```

Graph adapter 独占以下职责：

- graph artifact 校验和 `GraphIndex` 使用。
- `GraphRerankConfig` parse/validation/serialization。
- graph-specific candidate execution。
- `Avg Retrieved Edges` 等 graph trace 语义。

通用 runner 不导入任何 graph module。

兼容要求：

- `graph_memory.retrieval.tuning.__all__` 继续导出
  `graph_rerank_grid`、`graph_rerank_grid_from_record`、
  `tune_graph_rerank`。
- 现有 Graph Rerank candidate JSON 结构不变。
- `scripts/tune_graph_rerank.py` 参数和 sidecar 命名不变。
- 现有 graph search-space JSON schema 不变。

## 8. Memory Stream Adapter

### 8.1 Search-space 配置

新增：

```text
configs/search_spaces/memory_stream.json
```

初始配置只作为可编辑默认值，不编码任何不可变的权重约束：

```json
{
  "relevance_weight": [1.0],
  "recency_weight": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
  "importance_weight": [0.0],
  "recency_decay": [0.99]
}
```

代码规则：

- 四个字段都必须存在且为非空 list。
- 每个 weight 必须是 finite 且 `>= 0.0`。
- 每个候选至少一个 weight `> 0.0`。
- `recency_decay` 必须满足 `0 < value <= 1.0`。
- 不删除数学上可能产生相同排序的候选。
- 不根据 `recency_weight == 0.0` 自动合并不同 decay 候选。
- 是否扩大搜索空间只改 JSON。

`memory_stream_grid_from_record()` 返回
`list[MemoryStreamScoringConfig]`，领域校验统一复用
`parse_memory_stream_scoring_config()`，不在 grid parser 中复制数值规则。

### 8.2 Signal cache

新增 tuning-only 类型：

```python
@dataclass(frozen=True)
class MemoryStreamSignalCache:
    relevance_by_task_id: dict[TaskId, dict[NodeId, float]]
    importance_by_task_id: dict[TaskId, dict[NodeId, float]]
    seed_latency_ms_by_task_id: dict[TaskId, float]
```

cache 中保存已经 task-local min-max normalized 的 relevance 和 importance：

- relevance 来源于一次性 `SeedScoreCache`。
- importance 来源于已选择并严格验证的 importance records。
- recency 不放入 cache，因为它依赖每个候选的 `recency_decay`，计算成本很低。

将以下纯函数放在
`graph_memory/retrieval/methods/memory_stream/scoring.py`，供正式 method 和 tuner
共同使用：

```python
def pseudo_recency_scores(
    task_input: MemoryTaskInput,
    *,
    decay: float,
) -> dict[NodeId, float]:
    ...


def score_memory_stream(
    signals: NormalizedMemoryStreamSignals,
    *,
    config: MemoryStreamScoringConfig,
) -> dict[NodeId, float]:
    ...
```

正式 `MemoryStreamMethod.rank_task()` 和 tuning adapter 必须经过相同 pure scoring
函数，防止“调参路径”和“正式 retrieval 路径”出现公式漂移。

### 8.3 候选执行

每个 candidate：

1. 对每个 task 根据 candidate decay 计算 pseudo-recency。
2. 对 pseudo-recency 做 task-local min-max normalization。
3. 从 cache 读取 normalized relevance 和 importance。
4. 调用共享 `score_memory_stream()`。
5. 按 `(-score, node_id)` 生成完整 ranking。
6. 使用现有 `assemble_ranked_result()` 构建 `RankedResult`。
7. latency 使用 cached dense latency 加本候选的信号组合与排序时间。
8. 调用现有 `evaluate_results(predictions, labels, graphs)`。

不能通过为每个 candidate 新建普通 `MemoryStreamMethod` 并调用
`run_retrieval()` 来实现，因为这会对每个候选重复执行 dense ranking。

### 8.4 Tuning service

建议接口：

```python
def tune_memory_stream(
    *,
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    importance_artifact: ImportanceArtifact,
    grid: list[MemoryStreamScoringConfig],
    top_k: int = 10,
    dense_runtime: DenseRuntime | None = None,
) -> tuple[MemoryStreamScoringConfigRecord, list[MemoryStreamTuningCandidateRow]]:
    ...
```

其中：

```python
class MemoryStreamScoringConfigRecord(TypedDict):
    relevance_weight: float
    recency_weight: float
    importance_weight: float
    recency_decay: float


class MemoryStreamTuningCandidateRow(MetricRow):
    config: MemoryStreamScoringConfigRecord
```

service 不接收 importance path，只接收已经读取的 artifact。文件读取、SHA-256 和
run summary 属于 CLI adapter。

### 8.5 CLI

新增：

```text
scripts/tune_memory_stream.py
```

参数：

```text
--tasks
--labels
--graphs
--importance
--output_config
--encoder_model
--query_prefix
--passage_prefix
--top_k
--grid_config
```

输出与 Graph Rerank 对齐：

```text
<output_config>
<output_config stem>.candidates.json
<output_config stem>.run_summary.json
```

run summary 至少记录：

- tasks、labels、graphs、importance、grid config 输入路径。
- importance SHA-256。
- encoder model、prefix、batch size、top-k。
- task count、grid size、candidate row count。
- total wall-clock seconds。
- selected scoring config。

CLI 必须先完成 task/label/graph/importance alignment 校验，再加载 dense encoder 和
进入 grid evaluation。

## 9. Workflow 与 selected config 接入

调参能力不能继续由 `RetrievalLifecycle` 隐式表示。Memory Stream 的 runtime
仍然是 stateless retrieval，但它需要 tune stage；因此不能把
`MEMORY_STREAM` 错误标记为 `GRAPH_RERANK` lifecycle。

### 9.1 Registry 能力

在 `graph_memory/registry/methods.py` 增加：

```python
class TuningKind(StrEnum):
    GRAPH_RERANK = "graph_rerank"
    MEMORY_STREAM = "memory_stream"


@dataclass(frozen=True)
class MethodDefinition:
    ...
    tuning: TuningKind | None = None
```

注册：

```text
bm25                         tuning=None
dense                        tuning=None
memory_stream                tuning=MEMORY_STREAM
bm25_graph_rerank            tuning=GRAPH_RERANK
dense_graph_rerank           tuning=GRAPH_RERANK
dense_rgcn_graph_retriever   tuning=None
dense_ft                     tuning=None
```

`RetrievalLifecycle` 继续只描述 runtime/training lifecycle；
`TuningKind` 只描述 dev 参数选择 adapter。两者不得合并。

### 9.2 Workflow

新增 `TUNED_STATELESS_RETRIEVAL_WORKFLOW`：

```text
prepare -> graphs -> tune -> retrieve -> evaluate -> aggregate
```

workflow registry 根据 `(lifecycle, tuning is not None)` 选择 workflow：

- `STATELESS + no tuning` -> `STATELESS_RETRIEVAL_WORKFLOW`
- `STATELESS + tuning` -> `TUNED_STATELESS_RETRIEVAL_WORKFLOW`
- `GRAPH_RERANK + tuning` -> `GRAPH_RERANK_WORKFLOW`

`build_tune_commands()` 根据 `TuningKind` 分派：

- `GRAPH_RERANK` -> `scripts/tune_graph_rerank.py`
- `MEMORY_STREAM` -> `scripts/tune_memory_stream.py`

分派只存在于 workflow adapter，不进入 `GridSearchRunner`。

### 9.3 Selected config artifact

将 retrieval dependency 中 graph-specific 的：

```python
GraphConfigSource.TUNED_ARTIFACT
```

替换为：

```python
class SelectedConfigSource(StrEnum):
    NONE = "none"
    TUNED_ARTIFACT = "tuned_artifact"
```

`RetrieveIO.graph_config` 重命名为：

```python
selected_config: Path | None
```

Graph Rerank 与 Memory Stream 都声明
`selected_config=SelectedConfigSource.TUNED_ARTIFACT`。

`scripts/run_retrieval.py` 负责读取 selected config artifact，并按 method：

- 将 Graph record 解析为 `GraphRerankConfig`。
- 将 Memory Stream record 解析为 `MemoryStreamScoringConfig`。

随后再构建 method。builder 不读取文件。

这一命名调整明确表达：tune stage 产物是 method scoring config，而不是 graph
artifact。它也避免未来继续增加 `memory_stream_config`、`graphrag_config` 等平行
IO 字段。

### 9.4 Experiment config

`configs/experiments/hotpotqa_evidence_retrieval.json` 增加：

```json
{
  "search_spaces": {
    "graph_rerank": "configs/search_spaces/graph_rerank.json",
    "memory_stream": "configs/search_spaces/memory_stream.json"
  }
}
```

workflow tune command 根据 `TuningKind.value` 读取对应 search space。

正式 retrieval 不再从以下 experiment-level 临时字段读取权重：

```text
memory_stream_relevance_weight
memory_stream_recency_weight
memory_stream_importance_weight
memory_stream_recency_decay
```

权重来源应唯一化为：

```text
dev search space
  -> selected config artifact
  -> test retrieval
```

smoke 或手工 retrieval 若不经过 workflow，可以显式提供一个
`MemoryStreamScoringConfig` stage config；不得在 builder 内静默回退到另一个文件。

## 10. 职责边界矩阵

| 层/模块 | 可以做 | 不可以做 |
|---|---|---|
| `ParameterGrid` | 笛卡尔积、固定字段合并、空数组校验 | 识别权重、解析 Graph/Memory config |
| `GridSearchRunner` | 遍历、调用 evaluator、保存结果、按 key 选优 | IO、缓存、指标计算、异常吞掉、日志 |
| retrieval `selection.py` | 从 `MetricRow` 生成 selection key | 运行 retrieval、读取 config |
| `seed_scores.py` | 调用 seed ranker 一次、保存 score/latency | 组合 Graph 或 Memory Stream 分数 |
| Graph grid adapter | Graph 字段和数值校验、领域 config 构造 | 评价指标选择、文件 IO |
| Memory grid adapter | Memory 字段和数值校验、领域 config 构造 | 自动固定或归一化权重 |
| Graph tuning service | graph/cache/candidate execution/evaluation | 读写 JSON、写 run summary |
| Memory tuning service | importance selection/signal cache/candidate execution/evaluation | 读取 importance path、生成 importance |
| CLI scripts | 文件读取、hash、validation、输出、日志、run summary | 实现打分公式和搜索算法 |
| workflow | 生成命令、连接 artifact path、检查 stage dependency | 运行候选或解析 scoring 公式 |
| retrieval method | 单 task 纯 ranking | artifact IO、dev label evaluation、grid search |

## 11. 数据与泄漏边界

1. Grid search 只允许在 dev tasks/dev labels 上选择参数。
2. test labels 不得进入 tune CLI 或 selected config。
3. importance artifact 仍是 query-independent 外部 artifact；tuner 只消费，不生成。
4. Graph 用于统一 connectivity 指标计算，不用于 Memory Stream scoring。
5. `MemoryStreamSignalCache` 不保存 label 或 gold evidence。
6. selected config 只包含四个 scoring 字段，不包含 metrics、task id 或 label 摘要。
7. candidate rows 可以包含 dev aggregate metrics，但只能作为 tuning run artifact，
   不得被 retrieval method 读取。

## 12. 错误处理

以下情况必须 fail fast：

- search space 缺字段、存在未知字段或候选数组为空。
- 候选值不是允许类型、非 finite、越界。
- grid 最终为空。
- task/label/graph task-id 不对齐。
- importance 缺 task、digest 不匹配、node coverage 不完整。
- seed cache 缺 task 或 node。
- evaluator 未返回恰好一个 aggregate metric row。
- selected config artifact 与 method 类型不匹配。
- workflow 运行 retrieve 但 tuned artifact 不存在。

不支持“跳过失败候选继续搜索”。某候选执行失败通常说明 config validator 不完整或
实现存在错误，静默跳过会让结果不可复现。

## 13. 测试结构

新增：

```text
tests/test_grid_search.py
tests/test_memory_stream_tuning.py
```

修改：

```text
tests/test_phase1_real_retrieval.py
tests/test_memory_stream_method.py
tests/test_phase1_real_cli_smoke.py
tests/test_workflow_orchestration.py
tests/test_registry_stage_configs.py
tests/test_config_run_retrieval.py
tests/test_public_api_exports.py
tests/test_retrieval_domain_boundaries.py
```

测试重点：

- 通用 runner 使用普通整数候选测试，不依赖 retrieval。
- exact tie 选择第一个候选。
- evaluator 每候选只调用一次。
- `ParameterGrid` 保持确定性顺序且不归一化/去重。
- Graph 迁移后结果、candidate rows 和 CLI contract 不变。
- Memory Stream dense seed 只执行一次，不随 grid size 增长。
- tuning scoring 与正式 `MemoryStreamMethod` 对同一 config 产生相同 ranking。
- selected Memory Stream config 能被 test retrieval stage 消费。
- workflow 将 Memory Stream 放入 tune stage，但 lifecycle 仍为 stateless。

## 14. 分步实施任务

### Task 1：建立通用 Grid Search 内核

**文件：**

- Create: `graph_memory/tuning/__init__.py`
- Create: `graph_memory/tuning/grid_search.py`
- Create: `tests/test_grid_search.py`

- [ ] **Step 1：先写 `ParameterGrid` 失败测试**

覆盖：

```python
def test_parameter_grid_expands_in_deterministic_order() -> None:
    grid = ParameterGrid(
        parameters={"a": [1, 2], "b": ["x", "y"]},
        fixed={"fixed": 3},
    )

    assert grid.expand() == [
        {"fixed": 3, "a": 1, "b": "x"},
        {"fixed": 3, "a": 1, "b": "y"},
        {"fixed": 3, "a": 2, "b": "x"},
        {"fixed": 3, "a": 2, "b": "y"},
    ]
```

并覆盖空数组、字段重名、字符串被误当 sequence。

- [ ] **Step 2：运行测试并确认因模块不存在而失败**

```powershell
uv run pytest -q tests/test_grid_search.py
```

- [ ] **Step 3：实现最小 `ParameterGrid`**

只使用 `itertools.product`，不增加领域校验。

- [ ] **Step 4：写 `GridSearchRunner` 失败测试**

覆盖：

```python
def test_grid_search_evaluates_each_candidate_once_and_selects_max_key() -> None:
    calls: list[int] = []
    runner = GridSearchRunner[int, int, int](selection_key=lambda value: value)

    result = runner.run(
        [1, 3, 2],
        lambda candidate: calls.append(candidate) or candidate * 10,
    )

    assert calls == [1, 3, 2]
    assert result.selected.config == 3
    assert [row.config for row in result.candidates] == [1, 3, 2]
```

另写 exact tie 选择第一个、空 grid 抛错测试。

- [ ] **Step 5：实现最小 runner 并运行测试**

```powershell
uv run pytest -q tests/test_grid_search.py
```

- [ ] **Step 6：提交**

```powershell
git add graph_memory/tuning tests/test_grid_search.py
git commit -m "refactor: add generic grid search runner"
```

### Task 2：抽取 retrieval selection policy

**文件：**

- Create: `graph_memory/retrieval/tuning/selection.py`
- Modify: `tests/test_phase1_real_retrieval.py`

- [ ] **Step 1：写 objective 和完整 tie-break key 测试**

必须覆盖 objective 相同后依次比较：

```text
Full Support@10 descending
Retrieval Latency / Query ascending
Avg Retrieved Edges ascending
```

- [ ] **Step 2：运行 focused test，确认新函数不存在**

```powershell
uv run pytest -q tests/test_phase1_real_retrieval.py -k "tuning_objective or candidate_key"
```

- [ ] **Step 3：实现 `retrieval_tuning_objective()` 和
`retrieval_candidate_key()`**

- [ ] **Step 4：运行 focused test**

```powershell
uv run pytest -q tests/test_phase1_real_retrieval.py -k "tuning_objective or candidate_key"
```

- [ ] **Step 5：提交**

```powershell
git add graph_memory/retrieval/tuning/selection.py tests/test_phase1_real_retrieval.py
git commit -m "refactor: isolate retrieval tuning selection policy"
```

### Task 3：迁移 Graph Rerank 到通用 runner

**文件：**

- Move: `graph_memory/retrieval/tuning/grid.py`
  -> `graph_memory/retrieval/tuning/graph_rerank_grid.py`
- Move: `graph_memory/retrieval/tuning/initial_scores.py`
  -> `graph_memory/retrieval/tuning/seed_scores.py`
- Move: `graph_memory/retrieval/tuning/service.py`
  -> `graph_memory/retrieval/tuning/graph_rerank.py`
- Modify: `graph_memory/retrieval/tuning/__init__.py`
- Modify: `scripts/tune_graph_rerank.py`
- Modify: `tests/test_phase1_real_retrieval.py`
- Modify: `tests/test_phase1_real_cli_smoke.py`
- Modify: `tests/test_public_api_exports.py`

- [ ] **Step 1：新增 Graph adapter 使用 runner 的行为测试**

通过 fake candidate evaluator 或现有小 fixture 证明：

- 所有 candidate row 顺序保持不变。
- selected config 仍使用原 selection policy。
- seed ranker 只执行一次。

- [ ] **Step 2：运行测试确认旧 service 尚未使用 runner**

```powershell
uv run pytest -q tests/test_phase1_real_retrieval.py -k "grid_search or tuning_reuses_seed"
```

- [ ] **Step 3：完成明确命名的文件迁移**

同时将：

```python
InitialScoreCache
precompute_initial_score_cache
run_graph_rerank_from_initial_score_cache
```

改名为：

```python
SeedScoreCache
precompute_seed_score_cache
run_graph_rerank_from_seed_score_cache
```

- [ ] **Step 4：让 `tune_graph_rerank()` 调用 `GridSearchRunner`**

Graph service 保留 validation、cache 和 evaluation；删除本地候选循环与
`select_best_config()`。

- [ ] **Step 5：运行 Graph tuning 与 CLI tests**

```powershell
uv run pytest -q `
  tests/test_phase1_real_retrieval.py `
  tests/test_phase1_real_cli_smoke.py `
  tests/test_public_api_exports.py
```

- [ ] **Step 6：提交**

```powershell
git add graph_memory/retrieval/tuning/graph_rerank.py
git add graph_memory/retrieval/tuning/graph_rerank_grid.py
git add graph_memory/retrieval/tuning/seed_scores.py
git add graph_memory/retrieval/tuning/__init__.py
git add scripts/tune_graph_rerank.py
git add tests/test_phase1_real_retrieval.py
git add tests/test_phase1_real_cli_smoke.py
git add tests/test_public_api_exports.py
git commit -m "refactor: run graph rerank tuning through generic grid search"
```

### Task 4：统一 Memory Stream scoring config

**文件：**

- Create: `graph_memory/retrieval/methods/memory_stream/config.py`
- Modify: `graph_memory/retrieval/methods/memory_stream/scoring.py`
- Modify: `graph_memory/retrieval/methods/memory_stream/method.py`
- Modify: `graph_memory/retrieval/methods/memory_stream/__init__.py`
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `tests/test_memory_stream_method.py`
- Modify: `tests/test_retrieval_registry_builders.py`

- [ ] **Step 1：写 config parse/validation 失败测试**

覆盖四个字段、未知字段、非 finite、负权重、全零权重和 decay 范围。

- [ ] **Step 2：运行测试确认 config 类型不存在**

```powershell
uv run pytest -q tests/test_memory_stream_method.py
```

- [ ] **Step 3：实现 `MemoryStreamScoringConfig`、record parser 和 serializer**

parser 是正式 retrieval 与 grid parser 的唯一数值验证入口。

- [ ] **Step 4：写共享 scoring path 等价测试**

同一个 task、signal 和 config：

```text
MemoryStreamMethod ranking
==
score_memory_stream() + rank_memory_stream_scores()
```

- [ ] **Step 5：重构 method 与 builder 使用 scoring config**

不改变现有默认公式和 tie-break。

- [ ] **Step 6：运行 focused tests**

```powershell
uv run pytest -q `
  tests/test_memory_stream_method.py `
  tests/test_retrieval_registry_builders.py
```

- [ ] **Step 7：提交**

```powershell
git add graph_memory/retrieval/methods/memory_stream/config.py
git add graph_memory/retrieval/methods/memory_stream/scoring.py
git add graph_memory/retrieval/methods/memory_stream/method.py
git add graph_memory/retrieval/methods/memory_stream/__init__.py
git add graph_memory/registry/retrieval.py
git add graph_memory/registry/retrieval_builders.py
git add tests/test_memory_stream_method.py
git add tests/test_retrieval_registry_builders.py
git commit -m "refactor: define memory stream scoring config"
```

### Task 5：实现 Memory Stream grid adapter 与缓存执行

**文件：**

- Create: `graph_memory/retrieval/tuning/memory_stream_grid.py`
- Create: `graph_memory/retrieval/tuning/memory_stream.py`
- Create: `tests/test_memory_stream_tuning.py`
- Modify: `graph_memory/retrieval/tuning/__init__.py`

- [ ] **Step 1：写 search-space parse 失败测试**

验证：

- 单元素数组可以固定任意字段。
- 多元素数组展开完整笛卡尔积。
- 不自动去重或归一化候选。
- 未知字段和缺失字段失败。

- [ ] **Step 2：运行测试确认 adapter 不存在**

```powershell
uv run pytest -q tests/test_memory_stream_tuning.py
```

- [ ] **Step 3：实现 `memory_stream_grid_from_record()`**

使用 `ParameterGrid` 展开 record，再通过统一 parser 转为领域 config。

- [ ] **Step 4：写 dense seed 只执行一次的失败测试**

使用计数 fake ranker 和至少三个候选，断言 rank 调用次数等于 task 数，不等于
`task 数 * grid size`。

- [ ] **Step 5：实现 `MemoryStreamSignalCache` 和候选执行**

复用：

- `precompute_seed_score_cache()`
- `normalize_task_signal()`
- `pseudo_recency_scores()`
- `score_memory_stream()`
- `assemble_ranked_result()`
- `evaluate_results()`

- [ ] **Step 6：写 selection 与正式 method ranking 等价测试**

- [ ] **Step 7：运行 focused tests**

```powershell
uv run pytest -q `
  tests/test_memory_stream_tuning.py `
  tests/test_memory_stream_method.py
```

- [ ] **Step 8：提交**

```powershell
git add graph_memory/retrieval/tuning/memory_stream.py
git add graph_memory/retrieval/tuning/memory_stream_grid.py
git add graph_memory/retrieval/tuning/__init__.py
git add tests/test_memory_stream_tuning.py
git commit -m "feat: add memory stream grid tuning service"
```

### Task 6：实现 Memory Stream tuning CLI

**文件：**

- Create: `scripts/tune_memory_stream.py`
- Create: `configs/search_spaces/memory_stream.json`
- Modify: `tests/test_phase1_real_cli_smoke.py`
- Modify: `tests/test_cli_contracts.py`

- [ ] **Step 1：写 CLI smoke 失败测试**

使用 tiny task/label/graph/importance fixture 和 fake dense encoder，验证：

- 读取指定 search space。
- 写 selected config。
- 写 candidates sidecar。
- 写 success run summary。
- selected config 只包含四个 scoring 字段。

- [ ] **Step 2：运行 CLI test，确认脚本不存在**

```powershell
uv run pytest -q tests/test_phase1_real_cli_smoke.py -k memory_stream_tuning
```

- [ ] **Step 3：实现 CLI IO、validation、hash 和 summary**

脚本不得实现候选循环或 scoring。

- [ ] **Step 4：增加失败 summary 测试**

importance mismatch 时：

- 抛出原始 validation error。
- run summary `status="failed"`。
- 不写 selected config。

- [ ] **Step 5：运行 CLI tests**

```powershell
uv run pytest -q `
  tests/test_phase1_real_cli_smoke.py `
  tests/test_cli_contracts.py
```

- [ ] **Step 6：提交**

```powershell
git add scripts/tune_memory_stream.py
git add configs/search_spaces/memory_stream.json
git add tests/test_phase1_real_cli_smoke.py
git add tests/test_cli_contracts.py
git commit -m "feat: add memory stream tuning command"
```

### Task 7：将 tuning 能力接入 registry 和 workflow

**文件：**

- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `scripts/workflow/registry.py`
- Modify: `scripts/workflow/planner.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/status.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Modify: `tests/test_method_registry.py`
- Modify: `tests/test_workflow_orchestration.py`
- Modify: `tests/test_registry_stage_configs.py`

- [ ] **Step 1：写 registry 正交能力失败测试**

断言：

```text
memory_stream.lifecycle == STATELESS
memory_stream.tuning == MEMORY_STREAM
dense.lifecycle == STATELESS
dense.tuning is None
```

- [ ] **Step 2：写 workflow 失败测试**

断言 Memory Stream stages 为：

```text
prepare, graphs, tune, retrieve, evaluate, aggregate
```

且 tune command 使用 `scripts/tune_memory_stream.py` 和 memory stream search space。

- [ ] **Step 3：实现 `TuningKind` 和 tuned stateless workflow**

- [ ] **Step 4：将 workflow tune command 改为按 `TuningKind` 分派**

- [ ] **Step 5：运行 registry/workflow tests**

```powershell
uv run pytest -q `
  tests/test_method_registry.py `
  tests/test_workflow_orchestration.py `
  tests/test_registry_stage_configs.py
```

- [ ] **Step 6：提交**

```powershell
git add graph_memory/registry/methods.py
git add graph_memory/registry/stage_configs.py
git add scripts/workflow/workflows.py
git add scripts/workflow/registry.py
git add scripts/workflow/planner.py
git add scripts/workflow/manifest.py
git add scripts/workflow/status.py
git add scripts/workflow/stage_configs.py
git add configs/experiments/hotpotqa_evidence_retrieval.json
git add tests/test_method_registry.py
git add tests/test_workflow_orchestration.py
git add tests/test_registry_stage_configs.py
git commit -m "feat: register memory stream tuning workflow"
```

### Task 8：统一 selected config retrieval 输入

**文件：**

- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `scripts/run_retrieval.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `tests/test_config_run_retrieval.py`
- Modify: `tests/test_registry_stage_configs.py`
- Modify: `tests/test_retrieval_registry_builders.py`
- Modify: `tests/test_retrieval_provenance.py`

- [ ] **Step 1：写 `selected_config` IO contract 失败测试**

Graph Rerank 和 Memory Stream 均应指向
`manifest["artifacts"]["tuned"][method]`；普通 dense/BM25 为 `None`。

- [ ] **Step 2：写 Memory Stream selected config override 失败测试**

stage config 中的默认 scoring 与 selected artifact 不同，最终 built method 必须使用
selected artifact。

- [ ] **Step 3：将 graph-specific config source/IO 重命名为 selected config**

完成：

```text
GraphConfigSource -> SelectedConfigSource
RetrieveIO.graph_config -> RetrieveIO.selected_config
```

- [ ] **Step 4：在 run-retrieval adapter 中解析 method-specific selected config**

IO 和 parser 调度在 script/stage adapter，builder 只接收 typed config。

- [ ] **Step 5：运行 retrieval config/provenance tests**

```powershell
uv run pytest -q `
  tests/test_config_run_retrieval.py `
  tests/test_registry_stage_configs.py `
  tests/test_retrieval_registry_builders.py `
  tests/test_retrieval_provenance.py
```

- [ ] **Step 6：提交**

```powershell
git add graph_memory/registry/methods.py
git add graph_memory/registry/stage_configs.py
git add graph_memory/registry/retrieval.py
git add graph_memory/registry/retrieval_builders.py
git add scripts/run_retrieval.py
git add scripts/workflow/stage_configs.py
git add tests/test_config_run_retrieval.py
git add tests/test_registry_stage_configs.py
git add tests/test_retrieval_registry_builders.py
git add tests/test_retrieval_provenance.py
git commit -m "refactor: generalize tuned retrieval config input"
```

### Task 9：边界、文档和全量验证

**文件：**

- Modify: `pyproject.toml`
- Modify: `docs/30-design/architecture.md`
- Modify: `docs/30-design/abstractions.md`
- Modify: `docs/40-operations/commands.md`
- Create: `docs/configs/search_spaces/memory_stream.md`
- Modify: `tests/test_python310_compatibility.py`
- Modify: `tests/test_retrieval_domain_boundaries.py`
- Modify: `tests/test_public_api_exports.py`

- [ ] **Step 1：将项目最低 Python 版本调整为 3.10**

将：

```toml
requires-python = ">=3.12"
```

改为：

```toml
requires-python = ">=3.10"
```

新增代码不得使用 Python 3.11/3.12 才提供的标准库 typing 名称；需要的兼容类型从
现有 `typing_extensions` 导入。

- [ ] **Step 2：增加 Python 3.10 compatibility tests**

扩展 `tests/test_python310_compatibility.py`，对本计划新增模块执行 AST 检查，至少
拒绝从 `typing` 导入 Python 3.10 不存在的名称，并确保源码可以由 Python 3.10
语法解析。

- [ ] **Step 3：增加 import boundary tests**

必须证明：

- `graph_memory.tuning.grid_search` 不导入 `graph_memory.retrieval`。
- generic tuning package 不导入 Graph 或 Memory Stream。
- method package 不导入 tuning service、evaluation 或 CLI。
- workflow 不导入 scoring implementation。

- [ ] **Step 4：更新架构与命令文档**

命令文档提供独立命令和 workflow 命令，明确调参只使用 dev split。

- [ ] **Step 5：运行 Python 3.10 compatibility test**

```powershell
uv run pytest -q tests/test_python310_compatibility.py
```

- [ ] **Step 6：运行静态检查**

```powershell
uv run ruff check .
uv run basedpyright
```

- [ ] **Step 7：运行 focused retrieval suite**

```powershell
uv run pytest -q `
  tests/test_grid_search.py `
  tests/test_memory_stream_tuning.py `
  tests/test_memory_stream_method.py `
  tests/test_phase1_real_retrieval.py `
  tests/test_phase1_real_cli_smoke.py `
  tests/test_workflow_orchestration.py `
  tests/test_config_run_retrieval.py
```

- [ ] **Step 8：运行全量测试**

```powershell
uv run pytest -q
```

- [ ] **Step 9：验证 OpenSpec**

若实现时为本功能新增 OpenSpec change，运行：

```powershell
openspec validate <change-name> --strict
```

- [ ] **Step 10：提交**

```powershell
git add pyproject.toml
git add docs/30-design/architecture.md
git add docs/30-design/abstractions.md
git add docs/40-operations/commands.md
git add docs/configs/search_spaces/memory_stream.md
git add tests/test_python310_compatibility.py
git add tests/test_retrieval_domain_boundaries.py
git add tests/test_public_api_exports.py
git commit -m "docs: document generic retrieval tuning architecture"
```

## 15. 验收标准

- 通用 `GridSearchRunner` 不导入 retrieval、evaluation、Graph 或 Memory Stream。
- 任意字段是否固定完全由 search-space JSON 的候选数组决定。
- Graph Rerank 现有 callable、CLI、search-space 和输出 artifact 保持兼容。
- Memory Stream 产生 selected config、candidate rows 和 run summary。
- Memory Stream dense seed 计算次数与 grid size 无关。
- tuning 路径与正式 retrieval 路径共享同一 Memory Stream scoring 函数。
- Graph 与 Memory Stream 使用同一 retrieval selection policy，但不共享领域执行代码。
- Memory Stream lifecycle 保持 stateless，tuning capability 独立注册。
- test retrieval 只消费 dev-selected config，不读取 dev labels 或 candidate metrics。
- selected config IO 使用通用命名，不再绑定 graph。
- `pyproject.toml` 声明 `requires-python = ">=3.10"`，新增代码通过 Python 3.10
  compatibility test。
- focused tests、全量 pytest、ruff 和 basedpyright 全部通过。

## 16. 自审结果

### 抽象范围

通用层只抽取了稳定且真正相同的机制：参数组合、候选遍历、评价回调和选择 key。
没有试图统一 Graph 与 Memory Stream 的输入、缓存和评分实现。

### 命名

- `GridSearchRunner` 表达执行搜索，不暗示训练。
- `ParameterGrid` 表达参数空间，不包含领域 config。
- `SeedScoreCache` 表达 BM25/dense seed 的共享语义。
- `MemoryStreamScoringConfig` 表达正式 retrieval 与 tuning 共用的评分配置。
- `SelectedConfigSource`/`selected_config` 表达 tune stage 的通用输出。
- Graph/Memory adapter 文件名均显式包含 method 名。

### 配置边界

没有把 `relevance_weight=1.0` 写入代码假设。默认 JSON 使用单元素数组只是默认搜索
空间；用户可只改配置扩展任意字段。

### 性能边界

dense seed 在 grid 外计算一次。每个候选只执行低成本的信号组合、排序和 evaluation。
candidate latency 保留算法等价 latency，run summary 单独记录整个 grid wall time。

### 兼容性

Graph Rerank 外部入口和输出保持不变。内部模块重命名通过
`graph_memory.retrieval.tuning.__init__` 保持稳定 public API，不保留旧 leaf module
作为长期兼容层。
