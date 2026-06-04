# Config Registry Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个 stage-root、typed、registry-driven 的配置与构造系统，让 `scripts/*.py` 只选择当前阶段 config、执行 artifact IO、validation 和 run summary；配置解析、CLI 覆盖、profile 合并、method/stage 分发和 factory 映射全部收敛到 `graph_memory.config` 与 `graph_memory.registry`。

**Architecture:** `graph_memory.config` 只负责配置机制：按 registry spec 解析 argv、读取配置文件、应用 profile 与 CLI 覆盖、反序列化到 typed dataclass、序列化 resolved config。`graph_memory.registry` 负责所有声明和分发：stage config spec、CLI parser、method config union、factory map、ablation patch 和 workflow-facing projection。`scripts` 不做 method 分发，不构造 dense/R-GCN 依赖对象，但继续负责 task/graph/prediction/summary 等 artifact IO 边界。

**Tech Stack:** Python dataclass、Python 3.10-compatible `StrEnum`、轻量 dataclass 序列化库优先选 `cattrs`、标准库 `argparse` 与 JSON、现有 `graph_memory.infrastructure.io`、pytest、basedpyright、OpenSpec validation。

---

日期：2026-06-04

状态：计划审阅中。本文是 `docs/10-plans/graph-memory-core-package-refactor-design.md` 的配置与 registry 子系统补充计划，不代表代码已经迁移。

## 1. 本轮审阅后的修正结论

上一版计划中有几处设计必须推翻：

- 不再暴露 `load_profiled_file()`、`load_cli_config()` 等多个公开 loader API。
- 不再设计 `ConfigSource(path=None, profile=None, variant=None)` 这种可选字段组合。
- 不再把 `profile_key`、`defaults_key` 这类一万年不变的约定做成变量。
- 不再让 `RunRetrievalConfig` 顶层同时拥有 `dense`、`graph_rerank?`、`trainable?`。
- 不再让 application 层构造一堆宽 request 再在 resolver/factory 里继续按 method string 分发。
- 不再让 scripts 解析 argparse 后再把 typed args 交给 loader；argv 直接进入 loader，parser 由 registry 的 stage spec 提供。

新的方向：

```python
config = CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)
task_inputs = read_json(config.io.tasks)
validate_memory_task_inputs(task_inputs)
predictions = run_retrieve_stage(config, task_inputs=task_inputs)
write_json(config.io.output, predictions)
write_run_summary(config.io.summary, summary)
```

也就是说：

- scripts 只知道自己需要哪个 stage root config。
- config loader 负责把 argv 变成完整 typed config。
- registry 负责所有 config/method/training builder 声明和查表分发。
- stage runner 是固定函数，不挂在 registry 上。
- stage/domain execution 只接收已经构造好的 protocol 对象或 method-specific settings，不看到全局 option bag。

## 2. 边界分工

### 2.1 `scripts/*.py` 负责什么

scripts 保留这些职责：

- 选择当前 stage root config，例如 `Registry.configs.RETRIEVE`。
- 调用 `CONFIG_LOADER.load(spec, argv)`。
- 读取本 stage 的输入 artifact，例如 tasks、graphs、labels、pairs。
- 调用已有 validator 验证 artifact。
- 调用固定 stage runner 或 domain service。
- 写输出 artifact。
- 写 run summary。
- 设置 logging 和进程退出码。

scripts 不负责：

- 构造 argparse parser。
- 解析 argv。
- 合并 config file、profile 和 CLI override。
- 判断 method 是否是 BM25、dense、graph-rerank 或 trainable。
- 构造 dense encoder runtime、checkpoint runtime 或 seed retriever。
- 读取 method 内部 config 文件。
- 管理 ablation variant patch。

### 2.2 `graph_memory.config` 负责什么

config module 负责机制，不负责业务分发：

- 从 `StageConfigSpec` 获取 parser、config type、profile 规则和 CLI override 映射。
- 用 registry-provided parser 解析 argv。
- 定位 config file 或 resolved config file。
- 按固定约定读取 root config 和 `profiles`。
- 应用 profile patch。
- 应用 registry-provided ablation patch。
- 应用 CLI patch，且 CLI 永远最后覆盖。
- 用轻量库将 mapping 反序列化为 dataclass。
- 将 dataclass 序列化成 JSON-compatible mapping。
- 写 resolved config snapshot。

config module 不负责：

- 不知道具体 method 如何构造。
- 不 import retrieval/model implementation。
- 不包含 `if method == "dense"` 这样的 public method 分支。
- 不读取 task、graph、prediction、metrics artifact。

### 2.3 `graph_memory.registry` 负责什么

registry module 负责声明和分发：

- 每个 stage 的 config spec。
- 每个 stage 的 CLI parser。
- 每个 stage 的 argv -> config patch 映射。
- 每个 retrieval method 的 settings dataclass。
- method settings union 的 discriminator。
- method settings type -> builder function 的映射。
- ablation variant -> typed config patch。
- workflow-facing method capability projection。

registry 是唯一允许集中分发的地方。

允许：

```python
builder = registry.retrieval.builders[type(settings)]
return builder(settings, deps)
```

禁止在 scripts/application/domain 中出现：

```python
if method == "dense":
    build_dense_method()
elif method == "dense_rgcn_graph_retriever":
    build_checkpoint_graph_method()
```

### 2.4 registry 的抽象层级

当前 `graph_memory/retrieval/catalog.py` 位置太低，语义也太窄。它只是 retrieval 领域内部的 method metadata 文件，不适合作为整个系统的 composition/dispatch root。

目标设计中：

- `graph_memory.registry` 是顶层 application/composition 边界。
- `graph_memory.retrieval.catalog` 不再是 source of truth；迁移期最多作为 compatibility projection。
- `graph_memory.retrieval` 只保留 retrieval domain protocol、method implementation 和 execution service。
- `scripts/workflow/registry.py` 不再独立持有 ablation/method 语义；迁移期改为从 `graph_memory.registry` 投影出 workflow 需要的 view。

迁移目标：

```text
graph_memory.registry
  -> owns public method ids
  -> owns stage config specs
  -> owns method settings union registration
  -> owns builder maps
  -> owns ablation patches
  -> exports compatibility projections

graph_memory.retrieval
  -> owns RetrievalMethod protocol
  -> owns concrete retrieval implementations
  -> owns retrieval execution loop
  -> does not own global dispatch
```

因此，AI 或开发者新增方法时应该先找 `graph_memory/registry/`，而不是在 `retrieval/catalog.py`、`workflow/registry.py`、`run_retrieval.py` 之间猜测 source of truth。

### 2.5 registry 不负责执行 stage

registry 不提供：

```python
Registry.stages.RETRIEVE.run(config)
Registry.stages.TRAIN.run(config)
```

原因：

- 这会让 registry 从“声明/查表”变成执行入口。
- stage runner 拥有业务编排职责，例如读入后的 artifact 如何传给 domain service、如何组装 stage result。
- registry 只应该告诉 stage runner “这个 settings 类型对应哪个 builder”，不应该自己运行 stage。

正确关系：

```text
scripts/run_retrieval.py
  -> CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)
  -> run_retrieve_stage(config, loaded_artifacts)

stages/retrieve.py
  -> Registry.retrieval.build(config.job, deps)
  -> retrieval.execution.run_retrieval(retrieval_method, task_inputs, top_k)
```

职责口令：

```text
Registry describes and dispatches builders.
Stage runner orchestrates.
Scripts own artifact IO.
Domain service executes protocol.
```

## 3. Stage Root Config 设计

root config 按 workflow/script stage 切分，而不是按算法内部字段切分。

目标 root config：

```text
PrepareStageConfig
GraphBuildStageConfig
PairBuildStageConfig
TuneStageConfig
TrainStageConfig
RetrieveStageConfig
EvaluateStageConfig
AggregateStageConfig
ExperimentInitConfig
ExperimentPlanConfig
```

这些 config 是平等的入口 root。它们不是互相嵌套的上帝 config。

### 3.1 `RetrieveStageConfig`

错误形态：

```python
@dataclass(frozen=True)
class RetrieveStageConfig:
    method: str
    dense: DenseConfig
    graph_rerank: GraphRerankConfig | None
    trainable: TrainableGraphRuntimeConfig | None
```

这个结构会让 BM25 也携带 dense 字段，并且重新引入 `if config.trainable is not None`。

目标形态：

```python
@dataclass(frozen=True)
class RetrieveIO:
    tasks: Path
    graphs: Path | None
    output: Path
    summary: Path


@dataclass(frozen=True)
class RetrieveStageConfig:
    io: RetrieveIO
    job: RetrievalJobSettings
```

`RetrievalJobSettings` 是 discriminated union：

```python
RetrievalJobSettings = (
    Bm25RetrievalSettings
    | DenseRetrievalSettings
    | GraphRerankRetrievalSettings
    | CheckpointGraphRetrievalSettings
)
```

method-specific fields 只出现在自己的 settings 分支里。

### 3.2 Retrieval method settings

```python
class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    BM25_GRAPH_RERANK = "bm25_graph_rerank"
    DENSE_GRAPH_RERANK = "dense_graph_rerank"
    DENSE_RGCN_GRAPH_RETRIEVER = "dense_rgcn_graph_retriever"
```

```python
@dataclass(frozen=True)
class Bm25RetrievalSettings:
    method: Literal[RetrievalMethodId.BM25]
    top_k: int
```

```python
@dataclass(frozen=True)
class DenseEncoderSettings:
    model_name: str
    query_prefix: str
    passage_prefix: str
    batch_size: int = 64
```

```python
@dataclass(frozen=True)
class DenseRetrievalSettings:
    method: Literal[RetrievalMethodId.DENSE]
    top_k: int
    encoder: DenseEncoderSettings
```

```python
@dataclass(frozen=True)
class GraphRerankSettings:
    lambda_init: float = 1.0
    lambda_query: float = 0.1
    lambda_neighbor: float = 0.2
    lambda_bridge: float = 0.1
    lambda_path: float = 0.0
    seed_top_s: int = 30
    max_hops: int = 2
    neighbor_type_weights: dict[str, float] = field(default_factory=default_neighbor_type_weights)
```

```python
@dataclass(frozen=True)
class SeedRetrievalSettings:
    method: Literal[RetrievalMethodId.BM25, RetrievalMethodId.DENSE]
    encoder: DenseEncoderSettings | None = None
```

```python
@dataclass(frozen=True)
class GraphRerankRetrievalSettings:
    method: Literal[RetrievalMethodId.BM25_GRAPH_RERANK, RetrievalMethodId.DENSE_GRAPH_RERANK]
    top_k: int
    seed: SeedRetrievalSettings
    rerank: GraphRerankSettings
```

```python
@dataclass(frozen=True)
class CheckpointGraphRetrievalSettings:
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER]
    top_k: int
    checkpoint: Path
    device: str
```

注意：

- BM25 没有 dense encoder。
- dense encoder 在 dense job 或 dense seed job 内部。
- checkpoint graph retrieval 不通过 optional `trainable` 字段表达。
- `Runtime` 这个词不用来命名可序列化 config。

### 3.3 `TrainStageConfig`

训练入口不叫 `RgcnTrainingConfig` 包 `TrainableTrainingConfig`。

目标命名：

```python
@dataclass(frozen=True)
class RgcnMethodSettings:
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER]
    encoder: DenseEncoderSettings
    model: RgcnModelSettings
    trainer: RgcnTrainerSettings
    pairs: NegativeSamplingSettings
    reporting: TrainingReportingSettings
    selection: ModelSelectionSettings
```

```python
@dataclass(frozen=True)
class TrainStageConfig:
    io: TrainIO
    job: RgcnMethodSettings
```

`TrainableModelConfig` 这类 checkpoint reconstruction record 后续应改名或明确标注为 record：

```text
TrainableModelConfig -> RgcnCheckpointModelRecord
TrainableTrainingConfig -> RgcnCheckpointTrainerRecord
```

迁移期可以保留旧名，但新计划和新代码不得继续扩大这组命名。

### 3.4 `PairBuildStageConfig`

```python
@dataclass(frozen=True)
class PairBuildIO:
    tasks: Path
    labels: Path
    graphs: Path
    output: Path
    summary: Path
    run_summary: Path
```

```python
@dataclass(frozen=True)
class PairBuildStageConfig:
    io: PairBuildIO
    job: PairBuildSettings
```

```python
@dataclass(frozen=True)
class PairBuildSettings:
    method: RetrievalMethodId
    sampling: NegativeSamplingSettings
    dense_encoder: DenseEncoderSettings | None = None
```

### 3.5 `EvaluateStageConfig`

```python
@dataclass(frozen=True)
class EvaluateIO:
    predictions: Path
    labels: Path
    graphs: Path | None
    output: Path
    failure_cases_output: Path | None
```

```python
@dataclass(frozen=True)
class EvaluateStageConfig:
    io: EvaluateIO
    failure_case_limit: int = 50
```

Evaluate 不需要知道 retrieval method 内部 config。它只消费 ranked result artifact。

## 4. 单一 ConfigLoader API

公开 API 保持小：

```python
class ConfigLoader:
    def __init__(self, registry: AppRegistry, codec: ConfigCodec | None = None) -> None:
        self.registry = registry
        self.codec = codec or JsonConfigCodec()

    def load(self, spec: StageConfigSpec[T], argv: Sequence[str] | None) -> T:
        namespace = spec.parser_factory().parse_args(argv)
        raw = self._read_stage_config(spec, namespace)
        merged = self._merge_stage_layers(spec, namespace, raw)
        return self._converter.structure(merged, spec.config_type)

    def to_json(self, config: object) -> JsonValue:
        return self._converter.unstructure(config)

    def write_resolved(self, path: str | Path, config: object) -> None:
        self.codec.write(path, ensure_json_object(self.to_json(config)))
```

不提供：

```python
load_cli_config(config_type, args)
load_profiled_file(spec, path, profile)
load_resolved_file(spec, path)
ConfigSource(path=None, profile=None, variant=None)
```

这些都是内部步骤，不应成为调用者需要理解的 API。

### 4.1 `load(spec, argv)` 内部流程

```text
1. parser = spec.parser_factory()
2. namespace = parser.parse_args(argv)
3. base = read config path if spec has config path argument
4. profile_name = spec.profile_selector(namespace, base)
5. profile_patch = base["profiles"][profile_name] by convention
6. cli_patch = spec.cli_patch(namespace)
7. registry_patch = spec.registry_patch(namespace, base)
8. merged = merge(base_without_profiles, profile_patch, registry_patch, cli_patch)
9. settings = converter.structure(merged, spec.config_type)
10. return settings
```

固定约定：

- profile key 永远叫 `profiles`。
- 默认 profile key 永远叫 `default_profile`。
- CLI override 永远最后应用。
- ablation patch 由 registry 产生，优先级低于 CLI。
- unknown field fail-fast。

### 4.2 轻量序列化库

优先使用 `cattrs`：

- 适合标准 dataclass。
- 不要求 config 类型继承第三方 base class。
- 支持 structure/unstructure。
- 可以给 `Path`、`StrEnum`、tuple、union 注册 hook。

不优先使用 `msgspec` 作为第一步：

- 性能强，但最佳体验是 `msgspec.Struct`。
- 会要求 config 类型采用第三方 struct 基类。
- 当前配置加载不是性能瓶颈。

如果后续评估发现 `cattrs` 对 union discriminator 支持不够清晰，再考虑局部手写 discriminator 或切换 `msgspec`。

## 5. Registry 设计

### 5.1 Stage config spec

```python
ArgsNamespace = argparse.Namespace
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class StageConfigSpec(Generic[ConfigT]):
    stage: StageId
    config_type: type[ConfigT]
    parser_factory: Callable[[], argparse.ArgumentParser]
    cli_patch: Callable[[ArgsNamespace], ConfigPatch]
    config_path: Callable[[ArgsNamespace], Path | None]
    profile_name: Callable[[ArgsNamespace, Mapping[str, JsonValue]], str | None]
    registry_patch: Callable[[ArgsNamespace, Mapping[str, JsonValue]], ConfigPatch]
```

这些 callable 由 registry 提供。`ConfigLoader` 只执行，不知道每个 stage 的细节。

如果某个 stage 没有 config file，只靠 CLI 和 registry defaults，也返回 `None`。

### 5.2 Retrieval registry

```python
SettingsT = TypeVar("SettingsT")


@dataclass(frozen=True)
class RetrievalBuilderSpec(Generic[SettingsT]):
    settings_type: type[SettingsT]
    build: Callable[[SettingsT, RetrievalDependencies], RetrievalMethod]
```

注册表按 settings type 分发：

```python
RETRIEVAL_BUILDERS: dict[type[object], RetrievalBuilderSpec[object]] = {
    Bm25RetrievalSettings: RetrievalBuilderSpec(Bm25RetrievalSettings, build_bm25),
    DenseRetrievalSettings: RetrievalBuilderSpec(DenseRetrievalSettings, build_dense),
    GraphRerankRetrievalSettings: RetrievalBuilderSpec(GraphRerankRetrievalSettings, build_graph_rerank),
    CheckpointGraphRetrievalSettings: RetrievalBuilderSpec(
        CheckpointGraphRetrievalSettings,
        build_checkpoint_graph_retriever,
    ),
}
```

构造入口：

```python
def build_retrieval_method(settings: RetrievalJobSettings, deps: RetrievalDependencies) -> RetrievalMethod:
    spec = RETRIEVAL_BUILDERS[type(settings)]
    return spec.build(settings, deps)
```

这里是集中分发。`run_retrieval.py`、`application/run_retrieval.py` 和 `retrieval/execution/service.py` 不再做 method string 分支。

### 5.3 Training registry

训练路径也不能继续让高层函数感知 `rgcn`。`TrainStageConfig` 的 `job` 应该是 train method settings union：

```python
TrainJobSettings = (
    RgcnMethodSettings
    | DenseFinetuneMethodSettings
)
```

训练 registry 按 settings type 分发：

```python
TrainSettingsT = TypeVar("TrainSettingsT")


@dataclass(frozen=True)
class TrainBuilderSpec(Generic[TrainSettingsT]):
    settings_type: type[TrainSettingsT]
    build: Callable[[TrainSettingsT, TrainDependencies], TrainableMethodTrainer]
```

```python
TRAIN_BUILDERS: dict[type[object], TrainBuilderSpec[object]] = {
    RgcnMethodSettings: TrainBuilderSpec(RgcnMethodSettings, build_rgcn_trainer),
    DenseFinetuneMethodSettings: TrainBuilderSpec(DenseFinetuneMethodSettings, build_dense_finetune_trainer),
}
```

训练 stage runner 只做：

```python
trainer = Registry.training.build(config.job, deps)
result = trainer.train(train_inputs, dev_inputs)
```

它不应出现：

```python
if config.job.method == RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER:
    train_graph_retriever(train_inputs=train_inputs, dev_inputs=dev_inputs)
elif config.job.method == RetrievalMethodId.DENSE_FT:
    train_dense_finetune(train_inputs=train_inputs, dev_inputs=dev_inputs)
```

这样未来加入 Dense-FT 时，`scripts/train_graph_retriever.py` 或后续通用 `scripts/train_method.py` 不需要理解 Dense-FT 内部结构。新增工作限制为：

1. 新增 `DenseFinetuneMethodSettings`。
2. 新增 Dense-FT trainer implementation。
3. 在 `registry/training.py` 注册 settings type -> builder。
4. 在 `registry/stage_configs.py` 注册 CLI/config patch。
5. 增加 focused tests 和必要 docs。

### 5.4 Registry 与 config module 的配合

`ConfigLoader` 不知道 method 分支，但它需要知道 union 如何 decode。这个知识由 registry 提供：

```python
@dataclass(frozen=True)
class UnionDecodeSpec:
    union_type: type
    discriminator_field: str
    variants: Mapping[str, type]
```

retrieval registry 提供：

```python
RETRIEVAL_JOB_UNION = UnionDecodeSpec(
    union_type=RetrievalJobSettings,
    discriminator_field="method",
    variants={
        "bm25": Bm25RetrievalSettings,
        "dense": DenseRetrievalSettings,
        "bm25_graph_rerank": GraphRerankRetrievalSettings,
        "dense_graph_rerank": GraphRerankRetrievalSettings,
        "dense_rgcn_graph_retriever": CheckpointGraphRetrievalSettings,
    },
)
```

training registry 提供：

```python
TRAIN_JOB_UNION = UnionDecodeSpec(
    union_type=TrainJobSettings,
    discriminator_field="method",
    variants={
        "dense_rgcn_graph_retriever": RgcnMethodSettings,
        "dense_ft": DenseFinetuneMethodSettings,
    },
)
```

`ConfigLoader` 只把这些 union decode spec 注册给 converter。它不写任何 public method 字符串判断。

## 6. Scripts 目标形态

### 6.1 `scripts/run_retrieval.py`

目标：

```python
def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    started_at = now_iso()
    start_time = time.perf_counter()

    config = CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)
    task_inputs = read_json(config.io.tasks)
    validate_memory_task_inputs(task_inputs)
    graphs = read_json(config.io.graphs) if config.io.graphs is not None else []

    result = run_retrieve_stage(
        config,
        task_inputs=task_inputs,
        graphs=graphs,
    )

    validate_ranked_results(result.predictions, {task["task_id"]: task for task in task_inputs})
    write_json(config.io.output, result.predictions)
    write_run_summary(
        config.io.summary,
        build_run_summary(script="run_retrieval.py", status="success", effective_config=config_summary),
    )
    return 0
```

scripts 中不出现：

```python
DenseConfig(model_name=args.encoder_model)
DenseRuntime(config=dense_config)
TrainableGraphRuntime(checkpoint_path=args.checkpoint, device=args.device)
if config.job.method == RetrievalMethodId.DENSE
if config.job.trainable is not None
```

### 6.2 `scripts/build_train_pairs.py`

目标：

```python
config = CONFIG_LOADER.load(Registry.configs.PAIRS, argv)
tasks = read_json(config.io.tasks)
labels = read_json(config.io.labels)
graphs = read_json(config.io.graphs)
result = run_pair_build_stage(config, tasks=tasks, labels=labels, graphs=graphs)
write_json(config.io.output, result.pairs)
write_json(config.io.summary, result.summary)
write_run_summary(config.io.run_summary, run_summary)
```

sampling config 和 dense hard negative config 已经在 `PairBuildStageConfig.job` 中完整结构化。

### 6.3 `scripts/train_graph_retriever.py`

目标：

```python
config = CONFIG_LOADER.load(Registry.configs.TRAIN, argv)
train_tasks = read_json(config.io.train_tasks)
train_labels = read_json(config.io.train_labels)
train_graphs = read_json(config.io.train_graphs)
train_pairs = read_json(config.io.train_pairs)
dev_tasks = read_json(config.io.dev_tasks)
dev_labels = read_json(config.io.dev_labels)
dev_graphs = read_json(config.io.dev_graphs)

result = run_train_stage(
    config,
    train_tasks=train_tasks,
    train_labels=train_labels,
    train_graphs=train_graphs,
    train_pairs=train_pairs,
    dev_tasks=dev_tasks,
    dev_labels=dev_labels,
    dev_graphs=dev_graphs,
)
```

训练脚本不再调用多个 `*_from_training_config(dict)` helper。

## 7. Domain Execution 目标形态

### 7.1 retrieval execution service

`retrieval/execution/service.py` 应保持低层而纯粹：

```python
def run_retrieval(
    *,
    retrieval_method: RetrievalMethod,
    task_inputs: list[MemoryTaskInput],
    top_k: int,
) -> list[RankedResult]:
    return rank_all_tasks_with_method(retrieval_method, task_inputs, top_k)
```

它不应知道：

- BM25。
- dense。
- R-GCN。
- graph-rerank。
- config file。
- registry。

### 7.2 retrieve stage runner

`stages/retrieve.py` 负责把 stage config 转成 domain execution：

```python
def run_retrieve_stage(
    config: RetrieveStageConfig,
    *,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph],
) -> RetrieveStageResult:
    deps = RetrievalDependencies.from_loaded_artifacts(graphs=graphs)
    method = Registry.retrieval.build(config.job, deps)
    predictions = retrieval_execution.run_retrieval(
        retrieval_method=method,
        task_inputs=task_inputs,
        top_k=config.job.top_k,
    )
    return RetrieveStageResult(predictions=predictions)
```

注意这里的 `Registry.retrieval.build(config.job, deps)` 是唯一分发点。

### 7.3 train stage runner

`stages/train.py` 负责把 train stage config 转成 trainer execution：

```python
def run_train_stage(
    config: TrainStageConfig,
    *,
    train_inputs: TrainInputs,
    dev_inputs: DevInputs,
) -> TrainStageResult:
    deps = TrainDependencies.default()
    trainer = Registry.training.build(config.job, deps)
    return trainer.train(train_inputs=train_inputs, dev_inputs=dev_inputs)
```

`stages/train.py` 不应知道：

- R-GCN。
- Dense-FT。
- graph encoder type。
- dense encoder fine-tuning loss。
- checkpoint 内部 record schema。

这些细节分别属于 method settings、method trainer implementation 和 registry builder。

## 8. Config 文件结构

### 8.1 新 schema 去掉 `defaults`

旧格式：

```json
{
  "defaults": {
    "encoder": {},
    "model": {},
    "optimization": {}
  },
  "profiles": {}
}
```

目标格式：

```json
{
  "schema_version": 2,
  "method": "dense_rgcn_graph_retriever",
  "default_profile": "quick",
  "encoder": {
    "model_name": "models/intfloat-e5-base-v2",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "batch_size": 64
  },
  "model": {
    "ablation": "full_rgcn",
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1
  },
  "trainer": {
    "optimizer_name": "AdamW",
    "learning_rate": 0.0001,
    "batch_size": 8,
    "max_grad_norm": 1.0,
    "random_seed": 13,
    "pos_weight_enabled": true,
    "epochs": 5,
    "device": "cuda"
  },
  "pairs": {
    "random_seed": 13,
    "easy_random_per_positive": 2,
    "hard_bm25_per_positive": 2,
    "hard_dense_per_positive": 0,
    "hard_graph_neighbor_per_positive": 1,
    "hard_pool_size": 30
  },
  "profiles": {
    "quick": {
      "trainer": {
        "epochs": 5,
        "batch_size": 8
      }
    },
    "cloud-full": {
      "model": {
        "hidden_dim": 256
      },
      "trainer": {
        "epochs": 10,
        "batch_size": 128
      },
      "pairs": {
        "hard_dense_per_positive": 2
      }
    }
  }
}
```

root 本身就是 base config。`profiles` 只作为 patch。

### 8.2 文件目录保持浅层

继续支持旧路径：

```text
configs/training/dense_rgcn_graph_retriever/base.json
```

目标路径：

```text
configs/methods/dense_rgcn_graph_retriever.json
```

不按 dataclass 子结构拆文件。`encoder`、`model`、`trainer`、`pairs` 同属一个 method settings 文件。

### 8.3 YAML 迁移边界

JSON -> YAML 只影响：

- `ConfigCodec`。
- registry 中 config path 或 alias 后缀。
- docs/config examples。

不影响：

- scripts。
- stage runner。
- retrieval registry factory。
- model training code。

## 9. 命名规则

统一命名：

| 后缀 | 含义 |
|---|---|
| `Settings` | 可序列化、可从 config 文件/CLI 得到的配置 |
| `StageConfig` | 一个 script/workflow stage 的 root config |
| `Record` | artifact/checkpoint/run summary 中保存的记录 |
| `Dependencies` | 已构造的运行时依赖对象，例如 encoder、provider、artifact store |
| `Result` | stage/domain execution 的返回结果 |
| `Spec` | registry 声明 |

避免：

- 对可序列化配置使用 `RuntimeConfig`。
- 同时出现 `RgcnTrainingConfig` 和 `TrainableTrainingConfig`。
- 用 `ConfigSource` 表达多种状态。
- 在同一层 config 中使用大量 `Optional` 表达 method family 差异。

## 10. 目标模块结构

```text
graph_memory/
  config/
    __init__.py
    codec.py
    converter.py
    loader.py
    patches.py

  registry/
    __init__.py
    app.py
    specs.py
    ids.py
    stage_configs.py
    retrieval.py
    training.py
    ablations.py
    projections.py

  stages/
    retrieve.py
    pairs.py
    train.py
    evaluate.py
```

说明：

- `config/converter.py` 使用 `cattrs` 或兼容封装。
- `config/patches.py` 只负责 deep merge 和 CLI/profile/variant patch 数据结构。
- `registry/stage_configs.py` 注册 `Registry.configs.RETRIEVE` 等 config spec。
- `registry/retrieval.py` 注册 method settings type -> builder。
- `registry/training.py` 注册 train settings type -> trainer builder。
- `stages/*.py` 是 scripts 和 domain service 之间的 application layer。
- `registry/projections.py` 提供旧 `retrieval_registry.py`、`retrieval/catalog.py` 和 `scripts/workflow/registry.py` 需要的 compatibility view。

### 10.1 current registry migration

当前文件迁移规则：

```text
graph_memory/retrieval/catalog.py
  -> graph_memory/registry/retrieval.py owns source of truth
  -> retrieval/catalog.py becomes compatibility projection or is deleted

graph_memory/retrieval_registry.py
  -> re-export registry projection for workflow/docs compatibility
  -> no independent METHOD_REGISTRY

scripts/workflow/registry.py
  -> workflow stage/variant projection
  -> no independent R-GCN ablation semantics
```

目标是让 method/capability/ablation 信息只有一个 owner：`graph_memory.registry`。

## 11. 分批迁移计划

### Task 0: 冻结当前 CLI 与 workflow 契约

**Files:**

- Create: `tests/test_cli_contracts.py`
- Modify: no production file.

- [ ] **Step 0.1: 添加 parser contract tests**

覆盖：

- `scripts/run_retrieval.py`
- `scripts/build_train_pairs.py`
- `scripts/train_graph_retriever.py`
- `scripts/evaluate_retrieval.py`
- `scripts/experiment.py`

验收：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli_contracts.py -q
```

- [ ] **Step 0.2: 记录 full baseline**

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-config-registry-baseline -p no:cacheprovider
uv run basedpyright --outputjson --level error
openspec validate --all --strict
```

### Task 1: 建立 config loader 单入口

**Files:**

- Create: `graph_memory/config/__init__.py`
- Create: `graph_memory/config/codec.py`
- Create: `graph_memory/config/converter.py`
- Create: `graph_memory/config/patches.py`
- Create: `graph_memory/config/loader.py`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1.1: 添加 `ConfigLoader.load(spec, argv)`**

目标接口：

```python
class ConfigLoader:
    def load(self, spec: StageConfigSpec[T], argv: Sequence[str] | None) -> T:
        namespace = spec.parser_factory().parse_args(argv)
        raw = self._load_raw_config(spec, namespace)
        merged = self._resolve_layers(spec, namespace, raw)
        return self._converter.structure(merged, spec.config_type)
```

验收：

- 没有 `load_cli_config`。
- 没有 `ConfigSource`。
- CLI override 最后覆盖。

- [ ] **Step 1.2: 使用 `cattrs` 封装 converter**

目标：

- `Path` hook。
- `StrEnum` hook。
- tuple/list hook。
- dataclass unstructure。
- method union discriminator 由 registry 提供。

如果 implementation 阶段决定不引入依赖，必须保留相同 `ConfigConverter` 接口，后续可替换。

### Task 2: 建立 registry stage config spec

**Files:**

- Create: `graph_memory/registry/__init__.py`
- Create: `graph_memory/registry/specs.py`
- Create: `graph_memory/registry/stage_configs.py`
- Create: `graph_memory/registry/app.py`
- Test: `tests/test_registry_stage_configs.py`

- [ ] **Step 2.1: 定义 `StageConfigSpec`**

核心字段：

```python
@dataclass(frozen=True)
class StageConfigSpec(Generic[ConfigT]):
    stage: StageId
    config_type: type[ConfigT]
    parser_factory: Callable[[], argparse.ArgumentParser]
    config_path: Callable[[argparse.Namespace], Path | None]
    profile_name: Callable[[argparse.Namespace, Mapping[str, JsonValue]], str | None]
    cli_patch: Callable[[argparse.Namespace], ConfigPatch]
    registry_patch: Callable[[argparse.Namespace, Mapping[str, JsonValue]], ConfigPatch]
```

验收：

- `ConfigLoader` 不知道任何具体 CLI 参数名。
- `profile_key/defaults_key` 不存在。

- [ ] **Step 2.2: 注册 root configs**

注册：

```text
Registry.configs.PREPARE
Registry.configs.GRAPHS
Registry.configs.PAIRS
Registry.configs.TUNE
Registry.configs.TRAIN
Registry.configs.RETRIEVE
Registry.configs.EVALUATE
Registry.configs.AGGREGATE
Registry.configs.EXPERIMENT_INIT
```

### Task 3: 建立 retrieval settings union 与 builder registry

**Files:**

- Create: `graph_memory/stages/retrieve.py`
- Modify: `graph_memory/retrieval/factory.py`
- Modify: `graph_memory/retrieval/requests.py`
- Create: `graph_memory/registry/retrieval.py`
- Test: `tests/test_retrieval_registry_builders.py`

- [ ] **Step 3.1: 定义 method-specific retrieval settings**

至少包含：

- `Bm25RetrievalSettings`
- `DenseRetrievalSettings`
- `GraphRerankRetrievalSettings`
- `CheckpointGraphRetrievalSettings`

验收：

- BM25 settings 无 dense 字段。
- checkpoint graph settings 无 graph-rerank 字段。
- 不使用 `Optional` 表达 method family 差异。

- [ ] **Step 3.2: registry 按 settings type 构造 method**

目标：

```python
method = Registry.retrieval.build(config.job, deps)
```

验收：

```powershell
rg -n "method ==|method in|builder_id" graph_memory/retrieval graph_memory/stages scripts
```

Expected:

- public method string 分发只允许出现在 registry compatibility projection 或 tests。

- [ ] **Step 3.3: 迁移 `retrieval/catalog.py` 的 source of truth**

处理：

- `graph_memory/registry/retrieval.py` 持有 method id、capability、settings union 和 builder map。
- `graph_memory/retrieval/catalog.py` 只从 registry projection 生成旧 `RetrievalMethodSpec` view。
- `graph_memory/retrieval_registry.py` 只 re-export registry projection。

验收：

```powershell
rg -n "METHOD_REGISTRY|builder_id" graph_memory/retrieval graph_memory/retrieval_registry.py scripts
```

Expected:

- `METHOD_REGISTRY` 只允许出现在 compatibility projection 或迁移测试中。
- `builder_id` 不再作为 runtime dispatch 输入。

### Task 4: 迁移 `scripts/run_retrieval.py`

**Files:**

- Modify: `scripts/run_retrieval.py`
- Create or modify: `graph_memory/stages/retrieve.py`
- Test: `tests/test_config_run_retrieval.py`
- Test: `tests/test_phase1_real_cli_smoke.py`

- [ ] **Step 4.1: script 只选择 stage config**

目标：

```python
config = CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)
```

脚本仍负责：

- `read_json(config.io.tasks)`
- `read_json(config.io.graphs)`
- `validate_memory_task_inputs`
- `validate_ranked_results`
- `write_json(config.io.output, predictions)`
- `write_run_summary(config.io.summary, summary)`

验收：

- `scripts/run_retrieval.py` 不构造 dense/checkpoint runtime。
- CLI contract 不变。
- run summary schema 不变。

### Task 5: 迁移 pairs/train/evaluate stage config

**Files:**

- Modify: `scripts/build_train_pairs.py`
- Modify: `scripts/train_graph_retriever.py`
- Modify: `scripts/evaluate_retrieval.py`
- Create: `graph_memory/stages/pairs.py`
- Create: `graph_memory/stages/train.py`
- Create: `graph_memory/stages/evaluate.py`
- Test: `tests/test_phase2_rgcn_pairs.py`
- Test: `tests/test_phase2_rgcn_training.py`
- Test: `tests/test_experiment_runner.py`

- [ ] **Step 5.1: `build_train_pairs.py` 使用 `Registry.configs.PAIRS`**

验收：

- sampling settings 已结构化。
- hard dense encoder 只在 pair build settings 内部出现。
- direct CLI 中 CLI override 高于 file config。

- [ ] **Step 5.2: `train_graph_retriever.py` 使用 `Registry.configs.TRAIN`**

验收：

- 不再调用多个 `*_from_training_config(dict)` helper。
- train config 命名改为 `RgcnMethodSettings`、`RgcnTrainerSettings`。
- checkpoint record 命名和 schema 迁移策略明确。
- `stages/train.py` 不 import R-GCN trainer implementation；只调用 `Registry.training.build(config.job, deps)`。
- 新增 Dense-FT train settings 和 trainer 时，不修改 `stages/train.py`。

- [ ] **Step 5.3: `evaluate_retrieval.py` 使用 `Registry.configs.EVALUATE`**

验收：

- evaluate config 不依赖 retrieval method settings。
- evaluation 继续作为 shared hub。

- [ ] **Step 5.4: 建立 training builder registry**

目标：

```python
trainer = Registry.training.build(config.job, deps)
result = trainer.train(train_inputs=train_inputs, dev_inputs=dev_inputs)
```

验收：

```powershell
rg -n "dense_rgcn_graph_retriever|Rgcn|DenseFinetune|dense_ft" graph_memory/stages scripts
```

Expected:

- `graph_memory/stages/train.py` 和 `scripts/train_graph_retriever.py` 不出现具体 train method 分支。
- 具体 method 名称只允许出现在 registry、method implementation、config docs 和 tests。

### Task 6: workflow manifest 使用 typed stage root config

**Files:**

- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `scripts/workflow/registry.py`
- Test: `tests/test_experiment_runner.py`
- Test: `tests/test_workflow_orchestration.py`

- [ ] **Step 6.1: manifest 写入 resolved stage configs**

目标：

- manifest 仍保持旧 JSON schema 可读。
- 内部生成 commands 时优先使用 typed stage config projection。
- downstream scripts 继续通过 argv 进入 `CONFIG_LOADER.load(Registry.configs.<STAGE>, argv)`。

- [ ] **Step 6.2: ablation patch 由 registry 产生**

验收：

- `scripts/workflow/registry.py` 不再拥有 R-GCN variant 语义。
- `wo_bridge` 等 variant 的 typed patch 来自 `graph_memory.registry.ablations`。

### Task 7: config 文件 schema 与目录整理

**Files:**

- Create: `configs/methods/dense_rgcn_graph_retriever.json`
- Modify: `configs/training/dense_rgcn_graph_retriever/base.json` only if compatibility note is needed.
- Modify: `docs/configs/README.md`
- Test: `tests/test_config_schema_migration.py`

- [ ] **Step 7.1: 新 schema 去掉 `defaults` wrapper**

验收：

- root base config + `profiles` patch。
- 旧 schema 通过 migration adapter 读取。
- 新代码只产生 schema v2 resolved config。

- [ ] **Step 7.2: 新旧路径并存**

验收：

- `configs/training/dense_rgcn_graph_retriever/base.json` 继续可用。
- `configs/methods/dense_rgcn_graph_retriever.json` 可用。
- scripts 和 stage runner 不感知路径变化。

### Task 8: 删除旧分发和旧 helper

**Files:**

- Modify: `graph_memory/retrieval/factory.py`
- Modify: `graph_memory/retrieval/resolver.py`
- Modify: `graph_memory/models/graph_retriever/config/loading.py`
- Modify: `graph_memory/training_config.py`
- Create: `tests/test_config_registry_architecture.py`

- [ ] **Step 8.1: 删除 method string 分发**

验收：

```powershell
rg -n "method ==|method in|builder_id" graph_memory scripts
```

Expected:

- 只允许 registry compatibility projection 或 tests。

- [ ] **Step 8.2: 删除旧 dict slicing helper**

验收：

```powershell
rg -n "encoder_config_from_training_config|model_config_values_from_training_config|negative_sampling_config_from_training_config|trainable_training_config_from_training_config" graph_memory scripts tests
```

Expected:

- 无 production 调用。

## 12. 测试矩阵

### 12.1 focused tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config_loader.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_registry_stage_configs.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval_registry_builders.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_config_run_retrieval.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_runner.py -q
```

### 12.2 full validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-config-registry-refactor -p no:cacheprovider
uv run basedpyright --outputjson --level error
openspec validate --all --strict
```

### 12.3 必须新增的断言

- `CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)` 能直接返回完整 `RetrieveStageConfig`。
- scripts 不解析 argparse。
- scripts 不做 method 分发。
- scripts 仍负责 artifact IO、validation 和 run summary。
- scripts 调用固定 stage runner 函数，例如 `run_retrieve_stage()`，不调用 `Registry.stages.*.run()`。
- `retrieval/catalog.py` 不是 source of truth；source of truth 在 `graph_memory.registry`。
- profile patch 按固定 `profiles` 约定应用。
- CLI override 最后应用。
- ablation patch 来自 registry，且优先级低于 CLI。
- BM25 retrieval settings 不包含 dense 字段。
- Graph-rerank settings 中 dense encoder 只存在于 dense seed settings。
- 新增 method 只需要新增 settings、builder、registry entry 和 tests。
- 新增 Dense-FT 训练方法不需要修改 train stage runner。
- JSON -> YAML 只影响 codec 和路径 alias。

## 13. 实施顺序建议

### Change A: `add-stage-config-loader-and-registry-specs`

包含：

- Task 0
- Task 1
- Task 2

验收：

- `ConfigLoader.load(spec, argv)` 单入口可用。
- root stage config specs 注册完成。
- 不迁移主要脚本。

### Change B: `migrate-retrieve-stage-to-registry-dispatch`

包含：

- Task 3
- Task 4

验收：

- `scripts/run_retrieval.py` 只选择 stage config 并负责 artifact IO。
- retrieval method 分发只在 registry。
- BM25/dense/graph-rerank/trainable retrieval smoke tests 通过。

### Change C: `migrate-train-pairs-evaluate-stage-configs`

包含：

- Task 5

验收：

- pair/train/evaluate scripts 使用 stage root config。
- direct CLI override 优先级被正式测试。
- train config 不再靠 dict slicing helper。

### Change D: `typed-workflow-config-and-schema-cleanup`

包含：

- Task 6
- Task 7
- Task 8

验收：

- workflow manifest 使用 typed stage config projection。
- ablation patch 归 registry。
- `retrieval/catalog.py` 与 `retrieval_registry.py` 降级为 projection 或删除。
- 新 schema 去掉 `defaults` wrapper。
- 旧 method string 分发和旧 dict slicing helper 删除。

## 14. 审阅检查表

- [ ] scripts 是否只选择 `Registry.configs.<STAGE>`。
- [ ] scripts 是否没有 argparse parser 构造。
- [ ] scripts 是否没有 method 分发。
- [ ] scripts 是否仍保留 artifact IO、validation 和 summary。
- [ ] `ConfigLoader` 是否只有一个主要 `load(spec, argv)` 入口。
- [ ] 是否删除 `ConfigSource` optional bag。
- [ ] 是否删除 `profile_key/defaults_key`。
- [ ] retrieval settings 是否是 method-specific union。
- [ ] BM25 config 是否不携带 dense 字段。
- [ ] graph-rerank dense seed 是否只在 seed settings 中出现。
- [ ] registry 是否是唯一分发点。
- [ ] registry 是否没有 stage execution API。
- [ ] scripts 是否调用固定 stage runner，而不是 `Registry.stages.*.run()`。
- [ ] `retrieval/catalog.py` 是否不再持有 source of truth。
- [ ] train stage 是否不感知 R-GCN 或 Dense-FT。
- [ ] 新增 Dense-FT 是否只需要 method implementation、settings、registry/config、少量 scripts 适配。
- [ ] config 文件是否保持浅层。
- [ ] 新 schema 是否去掉 `defaults` wrapper。
- [ ] YAML 未来迁移是否只影响 codec/path alias。

## 15. 与核心包分层计划的关系

本文不取代 `graph-memory-core-package-refactor-design.md`。

关系如下：

- 核心包分层计划解决 package/domain 边界。
- 本文解决 stage root config、registry dispatch 和 script boundary。
- `RetrievalBuildContext` 删除和 typed build request 与两份计划一致。
- 本文应插入 retrieval/trainable/workflow 迁移批次之间，优先解决配置入口和分发边界。

推荐执行顺序：

```text
core foundations baseline
  -> stage config loader + registry specs
  -> retrieve stage registry dispatch
  -> pairs/train/evaluate stage config
  -> typed workflow manifest
  -> config schema/path cleanup
```
