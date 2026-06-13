# Memory Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个可复现的 `memory_stream` Phase 2 baseline：本地 Qwen2.5-7B-Instruct 离线生成并缓存 query-independent importance，在线检索按 normalized relevance + pseudo-recency + importance 排序。

**Architecture:** importance 作为独立 sidecar artifact，由新增 `importance` workflow stage 在检索前生成；检索阶段绝不调用 LLM，只读取已验证 artifact，并复用当前 DenseTaskRetriever 计算 relevance。annotation stage 直接使用已验证可运行的 `AutoTokenizer`/`AutoModelForCausalLM` 路径，先扫描 cache，再按需加载一次模型，并让同一个模型实例常驻到全部 cache miss 处理完成。

**Tech Stack:** Python dataclass/TypedDict/Protocol、SHA-256、PyTorch、Transformers `AutoTokenizer`/`AutoModelForCausalLM`、现有 SentenceTransformers 2.7.0 dense path、pytest、Ruff、BasedPyright、OpenSpec。

---

日期：2026-06-12

状态：方案已确认，OpenSpec change 为 `add-memory-stream-retrieval`，本文只定义实施范围和顺序，不代表代码已经实现。

## 1. 固定范围

必须实现：

1. public method id：`memory_stream`。
2. 公式：`relevance + pseudo-recency + importance`，默认权重均为 `1.0`。
3. relevance：复用现有 DenseTaskRetriever/DenseEncodingService。
4. pseudo-recency：由 `MemoryItem.position` 推导，不声称是真实时间。
5. importance：Qwen2.5-7B-Instruct 输出 1-10 整数，query-independent。
6. annotation、cache、retrieval 分离，正式 retrieval latency 不包含 LLM。
7. 一任务一次 deterministic generation；一个 annotation 进程只加载一个模型实例，逐任务原子 cache，失败可续跑。
8. 现有 `MemoryItem`、`MemoryTaskInput`、`RankedResult` schema 不改变。
9. 不读取 `*_memory_tasks.labels.json`，不向 prompt 暴露 query、answer、gold nodes 或 graph。
10. workflow、manifest、status/resume、delivery、文档全部接入。

明确不实现：

- 完整 Generative Agents simulator、reflection、planning、动态访问历史。
- importance 训练、蒸馏、权重调参或 ablation suite。
- train split importance。
- HTTP/cloud API、vLLM server 或 OpenAI-compatible server。
- Tensor Parallel、多进程生成或首版多卡切分。
- 由项目自动替换服务器上已经验证可用的 vendor-compatible Torch/Transformers 环境。
- LLM 失败后的静默默认分 4。

## 2. 目标依赖方向

```text
configs/experiments/hotpotqa_evidence_retrieval.json
  -> scripts/workflow/manifest.py
  -> typed ImportanceStageConfig + RetrieveStageConfig
  -> runs/<name>/config/stages/{importance,retrieve}/memory_stream.json

scripts/annotate_importance.py --config ...
  -> graph_memory/stages/importance.py
  -> memory_stream prompt/runtime/cache/validation
  -> one resident AutoTokenizer + AutoModelForCausalLM
  -> runs/<name>/importance/test.memory_stream.importance.json

scripts/run_retrieval.py --config ...
  -> graph_memory/stages/retrieve.py
  -> Registry.retrieval.build(...)
  -> MemoryStreamMethod
       -> DenseTaskRetriever relevance
       -> position pseudo-recency
       -> cached importance
  -> standard RankedResult
```

约束：

- registry 只描述 method、依赖来源和 builder dispatch，不加载 LLM 或执行 workflow。
- workflow 只编译 stage config、artifact path 和命令，不实现打分。
- annotation stage 只处理 input-visible tasks，不知道 labels。
- retrieval method 不知道本地模型、GPU 或 cache 目录。
- annotation stage 在创建 runtime 前完成全部 cache 扫描；全命中时不 import/load Transformers。
- 存在 miss 时 tokenizer/model 只加载一次，所有 miss 串行复用同一实例。
- cache 不进入 timed retrieval，也不写入 run 目录。

## 3. 文件职责图

### 新增文件

| 文件 | 单一职责 |
|---|---|
| `graph_memory/retrieval/methods/memory_stream/__init__.py` | method family 的窄导出面 |
| `graph_memory/retrieval/methods/memory_stream/contracts.py` | importance artifact/cache/result 的 JSON-shaped 类型 |
| `graph_memory/retrieval/methods/memory_stream/prompt.py` | prompt 构造、canonical semantic payload、digest |
| `graph_memory/retrieval/methods/memory_stream/runtime.py` | 本地 Transformers 模型加载、常驻生命周期和 deterministic generate |
| `graph_memory/retrieval/methods/memory_stream/cache.py` | content-addressed cache 读取、验证、原子写入 |
| `graph_memory/retrieval/methods/memory_stream/annotation.py` | cache hit/miss、单次 runtime 创建、串行 generation、最终 artifact assembly |
| `graph_memory/retrieval/methods/memory_stream/normalization.py` | deterministic min-max normalization |
| `graph_memory/retrieval/methods/memory_stream/method.py` | 三信号打分和 RetrievalMethod 实现 |
| `graph_memory/validation/importance.py` | importance artifact 与 task/node/digest 校验 |
| `graph_memory/stages/importance.py` | typed importance stage use case |
| `scripts/annotate_importance.py` | stage config CLI adapter 和 run summary |
| `tests/test_memory_stream_importance.py` | prompt、parser、digest、cache、本地 runtime 生命周期单元测试 |
| `tests/test_memory_stream_annotation_stage.py` | stage/CLI/artifact/run-summary 测试 |
| `tests/test_memory_stream_retrieval.py` | normalization、ranking、alignment、output 测试 |
| `tests/test_memory_stream_workflow.py` | registry/workflow/manifest/status/resume 测试 |
| `docs/configs/methods/memory_stream.md` | 配置字段说明 |

### 修改文件

| 文件 | 改动责任 |
|---|---|
| `graph_memory/infrastructure/io.py`, `graph_memory/io.py` | 增加并导出 `write_json_atomic()` |
| `graph_memory/validation/__init__.py` | 导出 importance validators |
| `graph_memory/registry/retrieval.py` | method id、settings、payload、provenance |
| `graph_memory/registry/retrieval_builders.py` | 构造 MemoryStreamMethod |
| `graph_memory/registry/methods.py` | lifecycle 和 importance dependency source |
| `graph_memory/registry/ids.py` | stage-config registry 的 IMPORTANCE stage id |
| `graph_memory/registry/stage_configs.py` | ImportanceStageConfig 与 RetrieveIO sidecar path |
| `graph_memory/stages/retrieve.py` | 读取 MemoryStream payload 分支 |
| `scripts/run_retrieval.py` | 加载 importance JSON，记录 provenance |
| `scripts/workflow/types.py` | IMPORTANCE stage/artifact/workflow id |
| `scripts/workflow/workflows.py` | MEMORY_STREAM_WORKFLOW 和 command builder |
| `scripts/workflow/registry.py` | lifecycle -> workflow |
| `scripts/workflow/artifacts.py` | importance artifact path |
| `scripts/workflow/manifest.py` | experiment config validation 和 artifact projection |
| `scripts/workflow/stage_configs.py` | 编译 importance/retrieve stage configs |
| `scripts/workflow/contracts.py` | 动态校验 importance stage config mapping |
| `scripts/workflow/planner.py` | stage dispatch 和 retrieve-only dependency |
| `scripts/workflow/status.py` | importance complete/stale/missing |
| `scripts/deliver/collect_run_artifacts.py` | 收集 sidecar 和 summary，不收集 global cache |
| `configs/experiments/hotpotqa_evidence_retrieval.json` | method 与固定 memory_stream config |
| `docs/20-contracts/retrieval-contracts.md` | 三信号和 sidecar contract |
| `docs/20-contracts/data-contracts.md` | importance artifact 数据边界 |
| `docs/30-design/architecture.md` | importance stage 所在层级 |
| `docs/40-operations/commands.md` | MetaX/Qwen/annotation/retrieve 命令 |
| `docs/40-operations/reproducibility.md` | cache、prompt/model provenance、latency 口径 |

## 4. 核心 contract

### 4.1 Importance artifact

`contracts.py` 使用 JSON-shaped `TypedDict`，不要把 runtime client 或 Path 塞入 persisted contract：

```python
class ImportanceGenerationRecord(TypedDict):
    do_sample: bool
    use_cache: bool
    max_new_tokens: int


class TaskImportanceRecord(TypedDict):
    task_id: TaskId
    content_digest: str
    scores: dict[NodeId, int]


class ImportanceArtifact(TypedDict):
    method: Literal["memory_stream"]
    model: str
    prompt_version: str
    generation: ImportanceGenerationRecord
    tasks: list[TaskImportanceRecord]
```

不增加 `schema_version`。仓库使用 current-only strict contract，旧 artifact 直接重跑。

### 4.2 Stage config

```python
@dataclass(frozen=True)
class ImportanceIO:
    tasks: Path
    output: Path
    summary: Path
    cache_dir: Path


@dataclass(frozen=True)
class ImportanceAnnotationSettings:
    model_id: str
    model_path: Path
    prompt_version: str
    device: Literal["auto", "cuda", "cpu"]
    trust_remote_code: bool
    torch_dtype: str
    low_cpu_mem_usage: bool
    tp_plan: None
    do_sample: Literal[False]
    use_cache: Literal[True]
    max_new_tokens: int


@dataclass(frozen=True)
class ImportanceStageConfig:
    io: ImportanceIO
    job: ImportanceAnnotationSettings

    io_type: ClassVar[type[ImportanceIO]] = ImportanceIO
```

首版固定 `do_sample=False`、`use_cache=True`、`torch_dtype="auto"`、`low_cpu_mem_usage=True`、`tp_plan=None`。物理 GPU 不写入 config，由启动前的 `CUDA_VISIBLE_DEVICES` 选择；进程内只看 local device 0。

### 4.3 Retrieval settings

```python
@dataclass(frozen=True)
class MemoryStreamRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    relevance_weight: float = 1.0
    recency_weight: float = 1.0
    importance_weight: float = 1.0
    recency_decay: float = 0.99
    method: Literal[RetrievalMethodId.MEMORY_STREAM] = RetrievalMethodId.MEMORY_STREAM


@dataclass(frozen=True)
class MemoryStreamBuildPayload:
    task_inputs: list[MemoryTaskInput]
    importance_artifact: ImportanceArtifact
    importance_path: Path
    dense_encoder: SentenceEncoder | None = None
```

`RetrieveIO` 新增精确字段：

```python
importance: Path | None = None
```

只有 `MemoryStreamRetrievalSettings` 允许该字段非空；其他 method 若携带 importance path 必须失败，避免 optional-bag 漂移。

## 5. Prompt、cache 和模型生命周期语义

### Prompt

system prompt 固定说明：

- 评估每条 memory sentence 的长期重要性/poignancy；
- 1 表示日常、低信息量，10 表示关键、显著、长期应保留；
- 每条独立按绝对尺度评分；
- 只输出 JSON。

user payload 只包含：

```json
{
  "items": [
    {
      "node_id": "n1",
      "source": "Article Title",
      "text": "Sentence text.",
      "position": 0
    }
  ],
  "output_format": {
    "scores": {
      "<node_id>": "<integer 1-10>"
    }
  }
}
```

### Cache key

```python
semantic_payload = {
    "model_id": settings.model_id,
    "prompt_version": settings.prompt_version,
    "generation": {
        "do_sample": settings.do_sample,
        "use_cache": settings.use_cache,
        "max_new_tokens": settings.max_new_tokens,
    },
    "items": [
        {
            "id": item["id"],
            "source": item["source"],
            "text": item["text"],
            "position": item["position"],
        }
        for item in task["memory_items"]
    ],
}
```

使用：

```python
json.dumps(
    semantic_payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
```

做 SHA-256。不得加入 query、task labels、`model_path`、`CUDA_VISIBLE_DEVICES`、物理卡号或运行目录。移动同一模型目录不应重算；替换权重时必须同步修改 `model_id`。

### 进程环境

- `scripts/annotate_importance.py` 必须在导入可能触发 Torch/Transformers 的 repository module 前清理 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
- 设置 `ACCELERATE_USE_DEEPSPEED=false`。
- `CUDA_VISIBLE_DEVICES` 由 shell 命令设置，脚本不覆盖用户选择。
- CUDA 可用时固定 `device_map={"": 0}`；CPU fallback 使用 `{"": "cpu"}`。
- 显式传递 `tp_plan=None`，禁止 Transformers 自动 Tensor Parallel。

### 模型生命周期

```python
cached, misses = resolve_all_cache_entries(tasks, settings)

if misses:
    runtime = runtime_factory(settings)
    runtime.load()  # 整个 stage 严格一次
    for task in misses:
        generated = runtime.generate(build_importance_messages(task))
        validated = parse_and_validate(generated, task)
        cache.write(validated)

artifact = assemble_in_original_task_order(cached, generated_results)
```

- 全 cache hit：runtime factory 调用次数为 0，`transformers` 不被导入。
- 一个或多个 miss：runtime factory 调用次数为 1，`from_pretrained()` 各调用一次。
- 同一进程内不得按 task 创建、销毁或重新加载模型。
- `model.eval()` 在加载后调用一次。
- generation 使用 chat template、`torch.inference_mode()`、`do_sample=False`、`use_cache=True` 和 pad-token fallback。
- CUDA timing 前后执行 `torch.cuda.synchronize()`。
- 默认串行生成，不对一个模型实例发起并发 `generate()`。
- 每个成功 task 立即原子写 cache。
- 所有 task 成功后才原子写 final sidecar。
- stage 结束或异常退出时才允许释放 runtime；正常进程退出负责最终显存释放。

## 6. 三信号数学定义

对一个 task 的每个 item：

```python
age_steps = max_position - item["position"]
recency_raw = recency_decay ** age_steps
importance_raw = float(task_importance["scores"][item["id"]])
relevance_raw = dense_score_by_node_id[item["id"]]
```

归一化：

```python
def minmax(values: Mapping[NodeId, float]) -> dict[NodeId, float]:
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {node_id: 0.0 for node_id in values}
    scale = maximum - minimum
    return {
        node_id: (value - minimum) / scale
        for node_id, value in values.items()
    }
```

总分：

```python
score = (
    config.relevance_weight * relevance[node_id]
    + config.recency_weight * recency[node_id]
    + config.importance_weight * importance[node_id]
)
```

排序：

```python
sorted(nodes, key=lambda node: (-node.score, node.node_id))
```

所有权重必须非负，且至少一个权重大于 0；`0 < recency_decay <= 1`。

## 7. 实施任务

### Task 1: 冻结 importance contract

**Files:**

- Create: `graph_memory/retrieval/methods/memory_stream/contracts.py`
- Create: `graph_memory/validation/importance.py`
- Modify: `graph_memory/validation/__init__.py`
- Create: `tests/test_memory_stream_importance.py`

- [ ] 先写 exact task/node coverage 和 1-10 integer 测试。
- [ ] 写 duplicate node、boolean、float、missing/extra node 失败测试。
- [ ] 写 content digest 与 item order 变化测试。
- [ ] 实现 TypedDict 和 validator。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_importance.py -q
```

预期：全部 PASS。

### Task 2: 实现 prompt 和 leakage guard

**Files:**

- Create: `graph_memory/retrieval/methods/memory_stream/prompt.py`
- Modify: `tests/test_memory_stream_importance.py`

- [ ] fixture 中放入 query/answer/gold/graph sentinel。
- [ ] 断言 prompt 和 cache canonical JSON 不含 sentinel。
- [ ] 实现 `build_importance_messages(task, prompt_version)`。
- [ ] 实现 `importance_content_digest(task, settings)`。
- [ ] 实现 fenced/plain JSON response parser。
- [ ] 运行 Task 1 命令。

### Task 3: 实现原子 JSON 和 cache store

**Files:**

- Modify: `graph_memory/infrastructure/io.py`
- Modify: `graph_memory/io.py`
- Create: `graph_memory/retrieval/methods/memory_stream/cache.py`
- Modify: `tests/test_memory_stream_importance.py`

- [ ] 写 `write_json_atomic()` 测试，临时文件必须与目标同目录。
- [ ] 实现 flush、close、`Path.replace()`。
- [ ] 实现 digest prefix 目录。
- [ ] cache hit 必须重新跑 strict validator。
- [ ] corrupted cache 视为 miss，并由有效新结果原子覆盖。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_importance.py tests/test_phase1_real_io_observability.py -q
```

### Task 4: 实现常驻本地 Transformers runtime

**Files:**

- Create: `graph_memory/retrieval/methods/memory_stream/runtime.py`
- Modify: `tests/test_memory_stream_importance.py`

- [ ] 定义可注入的 `ImportanceRuntime` Protocol 和 runtime factory。
- [ ] 写全 cache hit 时 factory 调用次数为 0 的测试。
- [ ] 写多个 cache miss 时 factory/load 调用次数均为 1、所有 generate 使用同一实例的测试。
- [ ] lazy import `torch`、`AutoTokenizer`、`AutoModelForCausalLM`。
- [ ] tokenizer 使用 `from_pretrained(model_path, trust_remote_code=True)`。
- [ ] model 使用 `torch_dtype="auto"`、`device_map={"": 0}` 或 CPU fallback、`low_cpu_mem_usage=True`、`tp_plan=None`。
- [ ] 加载后调用 `model.eval()`，记录 `model_load_seconds`。
- [ ] generation 使用 chat template、model parameter device、`torch.inference_mode()`、`do_sample=False`、`use_cache=True`、`max_new_tokens` 和 pad-token fallback。
- [ ] 记录 generated token 数、generation seconds 和 token/s。
- [ ] 架构测试禁止 HTTP、OpenAI SDK、vLLM server、ThreadPoolExecutor、Tensor Parallel 和 task loop 内的 `from_pretrained()`。
- [ ] 运行 Task 1 命令。

### Task 5: 实现 annotation orchestrator

**Files:**

- Create: `graph_memory/retrieval/methods/memory_stream/annotation.py`
- Modify: `tests/test_memory_stream_importance.py`

- [ ] 写全 cache hit 零 runtime/零 generation 测试。
- [ ] 写 mixed hit/miss 计数测试。
- [ ] 写多个 miss 串行复用同一 runtime 且最终仍按 input order 输出测试。
- [ ] 写中途失败后成功 cache 保留测试。
- [ ] 实现 `annotate_importance_tasks(..., runtime_factory=...) -> ImportanceAnnotationResult`。
- [ ] 确认 final artifact 只在全部成功后构造。

### Task 6: 建立 IMPORTANCE stage

**Files:**

- Modify: `graph_memory/registry/ids.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Create: `graph_memory/stages/importance.py`
- Create: `scripts/annotate_importance.py`
- Create: `tests/test_memory_stream_annotation_stage.py`
- Modify: `tests/test_registry_stage_configs.py`
- Modify: `tests/test_cli_contracts.py`

- [ ] 先在 registry `StageId` 增加 `IMPORTANCE`，再写 `Registry.configs.IMPORTANCE` config round-trip。
- [ ] 写 fake runtime 成功 CLI 测试。
- [ ] 写 rank/master 环境变量清理发生在 Torch/Transformers import 前的测试。
- [ ] 写 failed summary 测试。
- [ ] stage runner 不读取 labels。
- [ ] summary 写入 cache_hits、model_load_count、model_load_seconds、generation_calls、generated_tokens、generation_seconds、tasks、items、model_id、model_path、device。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_annotation_stage.py tests/test_registry_stage_configs.py tests/test_cli_contracts.py -q
```

### Task 7: 实现 MemoryStreamMethod

**Files:**

- Create: `graph_memory/retrieval/methods/memory_stream/normalization.py`
- Create: `graph_memory/retrieval/methods/memory_stream/method.py`
- Create: `graph_memory/retrieval/methods/memory_stream/__init__.py`
- Create: `tests/test_memory_stream_retrieval.py`

- [ ] 写 min-max normal/constant 测试。
- [ ] 写 recency position 测试。
- [ ] 写等权、单信号权重和 tie-break 测试。
- [ ] 写 relevance 与 DenseTaskRetriever raw score 一致测试。
- [ ] 写 importance task/node/digest mismatch 失败测试。
- [ ] 实现 method，trace edges 保持空数组。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_retrieval.py tests/test_batched_dense_encoding.py -q
```

### Task 8: 接入 retrieval registry

**Files:**

- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/stages/retrieve.py`
- Modify: `scripts/run_retrieval.py`
- Modify: `tests/test_retrieval_registry_builders.py`
- Modify: `tests/test_config_run_retrieval.py`
- Modify: `tests/test_retrieval_provenance.py`

- [ ] 增加 method id/settings/payload/lifecycle/ImportanceSource。
- [ ] builder 复用 `_build_seed_retriever()`，不创建第二套 dense scoring。
- [ ] `RetrieveIO.importance` 只允许 Memory Stream 使用。
- [ ] script 加载 sidecar 后先 strict validate，再调用 stage。
- [ ] provenance 包含 artifact path/model/prompt/weights/decay/encoder。
- [ ] Transformers 不可 import 或 model path 不存在时，已有 sidecar 的 retrieval 测试仍通过。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_retrieval.py tests/test_retrieval_registry_builders.py tests/test_config_run_retrieval.py tests/test_retrieval_provenance.py -q
```

### Task 9: 接入 workflow 和 manifest

**Files:**

- Modify: `scripts/workflow/types.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `scripts/workflow/registry.py`
- Modify: `scripts/workflow/artifacts.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `scripts/workflow/contracts.py`
- Create: `tests/test_memory_stream_workflow.py`
- Modify: `tests/test_workflow_orchestration.py`
- Modify: `tests/test_experiment_runner.py`

- [ ] `StageId` 顺序变为 `prepare, graphs, importance, pairs, tune, train, retrieve, evaluate, aggregate`。
- [ ] 新 workflow 不含 pairs/train。
- [ ] artifact path 固定：

```text
runs/<name>/importance/test.memory_stream.importance.json
runs/<name>/importance/test.memory_stream.importance.run_summary.json
```

- [ ] `stage_configs.importance` 必须存在，内容恰好为拥有 IMPORTANCE stage 的 selected methods。
- [ ] planner command 固定为 `scripts/annotate_importance.py --config ...`。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_workflow.py tests/test_workflow_orchestration.py tests/test_experiment_runner.py -q
```

### Task 10: status、resume、delivery

**Files:**

- Modify: `scripts/workflow/planner.py`
- Modify: `scripts/workflow/status.py`
- Modify: `scripts/workflow/resume.py`
- Modify: `scripts/deliver/collect_run_artifacts.py`
- Modify: `tests/test_memory_stream_workflow.py`
- Modify: `tests/test_deliver_run_artifacts.py`

- [ ] retrieve-only + missing importance 必须在 planning 阶段失败。
- [ ] sidecar 有文件但无 summary 时状态为 stale。
- [ ] summary model/prompt/generation/tasks/output 不匹配时状态为 stale。
- [ ] 同一路径下 memory item id/source/text/position/order 改变时状态也必须为 stale；query 单独变化不使 importance 失效。
- [ ] complete importance 可被 cache-aware plan 跳过。
- [ ] delivery 收集 final sidecar 和 summary，不遍历外部 cache 目录。
- [ ] 运行：

```powershell
uv run pytest tests/test_memory_stream_workflow.py tests/test_deliver_run_artifacts.py -q
```

### Task 11: experiment config 与文档

**Files:**

- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Create: `docs/configs/methods/memory_stream.md`
- Modify: `docs/20-contracts/data-contracts.md`
- Modify: `docs/20-contracts/retrieval-contracts.md`
- Modify: `docs/30-design/architecture.md`
- Modify: `docs/40-operations/commands.md`
- Modify: `docs/40-operations/reproducibility.md`

- [ ] 增加 `memory_stream` method。
- [ ] 固定配置示例：

```json
{
  "memory_stream": {
    "annotation": {
      "model_id": "Qwen/Qwen2.5-7B-Instruct",
      "model_path": "models/Qwen2.5-7B-Instruct",
      "prompt_version": "memory-stream-importance-v1",
      "device": "auto",
      "trust_remote_code": true,
      "torch_dtype": "auto",
      "low_cpu_mem_usage": true,
      "tp_plan": null,
      "do_sample": false,
      "use_cache": true,
      "max_new_tokens": 2048,
      "cache_dir": "data/cache/memory_stream_importance"
    },
    "retrieval": {
      "relevance_weight": 1.0,
      "recency_weight": 1.0,
      "importance_weight": 1.0,
      "recency_decay": 0.99
    }
  }
}
```

- [ ] 文档明确不启动 vLLM/HTTP server，直接使用服务器当前已验证的 vendor-compatible Torch/Transformers 环境。
- [ ] 文档给出 ModelScope/Hugging Face 下载、环境变量清理、`CUDA_VISIBLE_DEVICES=0`、direct Transformers preflight、annotation-only、restart、retrieve-only、full run 命令。
- [ ] 文档明确模型约 80 秒的加载成本只在存在 cache miss 时发生一次，绝不能为每个 task 启动 Python 或重新 `from_pretrained()`。
- [ ] 文档明确只处理 test split；全 cache hit 时模型不加载，部分命中时仅缺失 task 调用 Qwen。

运维文档中的 preflight 必须保留以下已验证模式，不能改写成 server/HTTP 调用：

```bash
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
export ACCELERATE_USE_DEEPSPEED=false
export CUDA_VISIBLE_DEVICES=0

MODEL="${MODEL:-models/Qwen2.5-7B-Instruct}" MAX_TOKENS=128 python - <<'PY'
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = os.environ["MODEL"]
max_tokens = int(os.environ["MAX_TOKENS"])

tokenizer = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True,
)
device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map=device_map,
    low_cpu_mem_usage=True,
    tp_plan=None,
)
model.eval()
print(f"load_seconds={time.perf_counter() - started:.2f}")

messages = [{"role": "user", "content": "请输出一个简短的 JSON 对象。"}]
prompt = (
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if getattr(tokenizer, "chat_template", None)
    else messages[0]["content"]
)
inputs = tokenizer(prompt, return_tensors="pt")
device = next(model.parameters()).device
inputs = {key: value.to(device) for key, value in inputs.items()}
input_len = inputs["input_ids"].shape[1]

if torch.cuda.is_available():
    torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    result = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()

generated = result[0, input_len:]
elapsed = time.perf_counter() - started
print(tokenizer.decode(generated, skip_special_tokens=True))
print(f"generated_tokens={generated.numel()}")
print(f"generation_seconds={elapsed:.3f}")
PY
```

该 preflight 只用于验证环境。正式 annotation 必须在单个 Python 进程内复用同一 `tokenizer` 和 `model` 处理全部 cache miss，禁止为每个 task 重复执行该脚本。

### Task 12: 全量验收

Windows 上所有 `uv` 命令必须从第一次就在 Codex filesystem sandbox 外执行。

```powershell
uv run pytest tests/test_memory_stream_importance.py tests/test_memory_stream_annotation_stage.py tests/test_memory_stream_retrieval.py tests/test_memory_stream_workflow.py -q
uv run pytest -q
uv run ruff check .
uv run basedpyright graph_memory scripts tests --level error
uv run python scripts/experiment.py methods list
uv run python scripts/experiment.py plan memory_stream_smoke --profile smoke --methods memory_stream --force --no-cache
openspec validate add-memory-stream-retrieval --strict
git diff --check
```

MetaX live smoke：

1. 执行：

```bash
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
export ACCELERATE_USE_DEEPSPEED=false
export CUDA_VISIBLE_DEVICES=0
```

2. 用 direct Transformers preflight 验证 tokenizer/model 可加载，且参数包含 `tp_plan=None`。
3. 在一个 annotation 进程中跑多个 HotpotQA task，确认 `model_load_count=1`，全部 generation 共用同一模型实例。
4. 原命令重跑，确认全部命中 cache、`model_load_count=0`、`generation_calls=0`。
5. 暂时令 Transformers 或 model path 不可用，再跑 retrieve/evaluate，确认已有 sidecar 足以完成。
6. 检查 annotation summary 分开记录 `model_load_seconds` 与 `generation_seconds`。
7. 检查 retrieval summary 的 latency 不包含 annotation/model load 时间。

## 8. 完成标准

- [ ] `memory_stream` 出现在 method list。
- [ ] prompt/cache key 中没有 query 或 label 数据。
- [ ] importance artifact 对 task/node/content 严格对齐。
- [ ] annotation 失败不会产生伪 complete final artifact。
- [ ] 成功 cache 可跨失败和跨 run 复用。
- [ ] 全 cache hit 不 import/load Transformers；有 cache miss 时每个 annotation 进程模型只加载一次。
- [ ] retrieval 不 import 或调用任何本地 causal LLM runtime。
- [ ] relevance 复用现有 dense path。
- [ ] pseudo-recency 明确来自 position。
- [ ] 三信号归一化、权重和 tie-break 可由单元测试精确证明。
- [ ] workflow plan 顺序正确且不包含 pairs/train。
- [ ] status/resume 能区分 missing/stale/complete importance。
- [ ] delivery 包含最终 sidecar 和运行证据，不包含 global cache。
- [ ] 全量 pytest、Ruff、BasedPyright、OpenSpec strict 和 diff check 通过。
