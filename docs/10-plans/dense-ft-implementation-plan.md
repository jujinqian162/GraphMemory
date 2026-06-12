# Dense-FT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `dense_ft` 公开检索方法，用现有 `train_pairs.json` 微调 SentenceTransformer，并让微调后的模型作为独立 dense baseline 进入 experiment workflow、manifest、retrieval、evaluation 和最终表格。

**Architecture:** `dense_ft` 训练侧新增独立 `graph_memory.models.dense_finetune` 包，负责把项目 artifact 转成 SentenceTransformers 训练/评估输入并保存可复用模型目录；推理侧不复制 dense 打分逻辑，继续复用 `DenseTaskRetriever -> DenseEncodingService`。训练 stage 需要从当前 R-GCN 专用形状改为 method-specific train config/payload，避免 dense-ft 被迫携带 graph feature provider、seed signal provider 等无意义依赖。

**Tech Stack:** Python dataclass、现有 `ConfigLoader`/`Registry`/workflow manifest、SentenceTransformers 2.7.0 `InputExample` / `SentenceTransformer.fit()`、PyTorch `DataLoader`、`MultipleNegativesRankingLoss`、`InformationRetrievalEvaluator`、pytest、basedpyright。

---

日期：2026-06-11

状态：OpenSpec 实施中。Dense-FT 训练后端统一以 SentenceTransformers 2.7.0 API 为准。

## 1. 审查结论

推荐方案：`dense_ft` 是一个独立公开方法，但它的推理实现复用现有 frozen dense 路径。

也就是说：

- 公开方法名新增 `dense_ft`，实验表里与 `dense` 并列。
- 训练产物是一个 SentenceTransformer model directory，而不是 `.pt` 图模型 checkpoint。
- 检索阶段用 `DenseTaskRetriever(config=DenseConfig(model_name=<model_dir>, ...))` 加载微调后的目录。
- dense 文本拼接、prefix、normalize_embeddings、batch encode 语义继续由 dense-owned 代码维护。
- dense-ft 训练只接触 task、label、pair artifact；不读取 graph 作为模型输入。
- pair 构建阶段仍可复用现有 `build_train_pairs.py`，因为当前 `train_pairs.json` 已经承载 positive、easy negative、BM25 hard negative、dense hard negative、graph-neighbor hard negative 的监督样本来源。

不推荐方案：

- 不在 `DenseTaskRetriever` 里增加 `if finetuned` 分支。
- 不新增第二套 dense 相似度排序实现。
- 不把 dense-ft 训练塞进 `graph_memory.models.graph_retriever`。
- 不让 `TrainStageConfig` 变成大量 optional 字段组成的通用 bag。
- 不让 workflow 继续用 `requires_checkpoint=True` 隐式等价 R-GCN 生命周期。

## 2. 当前代码边界

### 2.1 可以直接复用的部分

`graph_memory/embeddings/dense.py`

- `DenseEncodingService` 已经是 frozen dense 推理、R-GCN graph feature、seed signal 和 hard dense negative 的共享下层编码服务。
- dense-ft 推理应复用它，而不是重新实现 batch encode。

`graph_memory/retrieval/methods/flat/dense.py`

- `DenseTaskRetriever` 已经负责从 query/passage embeddings 计算 dot-product 排序。
- dense-ft 推理只需要把 `DenseConfig.model_name` 指向微调后的 SentenceTransformer 目录。

`graph_memory/training_pairs/*`

- `TrainPairRecord` 已经有 `task_id`、`node_id`、`label`、`sample_type`。
- dense-ft 训练可以按 task 分组，把 positive node 和 negative node 组装成 SentenceTransformers row。

`scripts/workflow/*`

- manifest、artifact namespace、stage projection 已经能按方法生成 learned artifact 路径。
- 需要补的是 dense-ft 的 workflow 类型和 checkpoint/model-dir 语义。

### 2.2 必须修改的部分

`graph_memory/registry/training.py`

- 当前 `TrainJobSettings = RgcnMethodSettings`。
- 当前 `TrainDependencies` 强制包含 R-GCN 需要的 `TextEmbeddingProvider` 和 `SeedSignalProvider`。
- dense-ft 接入前必须让 training registry 按 settings type 分发到不同 trainer，且 trainer 接收 method-specific payload。

`graph_memory/registry/stage_configs.py`

- 当前 `TrainIO` 固定包含 `train_graphs`、`dev_graphs`、`checkpoint_dir` 等 R-GCN 字段。
- dense-ft 训练不需要 train/dev graphs，因此 TRAIN config 要么变成根级 discriminated union，要么新增精确的 dense-ft train IO 类型。
- 推荐根级 discriminated union：`RgcnTrainStageConfig | DenseFinetuneTrainStageConfig`。

`scripts/train_graph_retriever.py`

- 当前脚本名称、provider 构造和保存逻辑都偏 R-GCN。
- 这个旧入口不继续保留兼容性；本轮直接统一到 `scripts/train_method.py`。
- 统一入口负责所有 train method 的 artifact IO、run summary 和 stage config 加载；具体训练逻辑仍由 registry 分发到 method trainer。

`scripts/workflow/workflows.py`

- 当前 `RGCN_WORKFLOW` 把 `pairs -> train -> retrieve` 与 checkpoint-backed graph retriever 绑定。
- dense-ft 需要自己的 workflow：`prepare -> graphs -> pairs -> train -> retrieve -> evaluate -> aggregate`。
- 注意：`graphs` 仍保留在 workflow 中，因为 pair 构建和 evaluation 现在都需要 graph artifact；dense-ft 模型训练本身不消费 graph。

## 3. 目标文件结构

新增文件：

```text
graph_memory/models/dense_finetune/__init__.py
graph_memory/models/dense_finetune/contracts.py
graph_memory/models/dense_finetune/data.py
graph_memory/models/dense_finetune/training.py
scripts/train_method.py
configs/training/dense_ft/base.json
docs/configs/training/dense_ft/base.md
tests/test_dense_finetune_data.py
tests/test_dense_finetune_training.py
tests/test_dense_ft_retrieval_registry.py
tests/test_dense_ft_workflow.py
```

修改文件：

```text
pyproject.toml
uv.lock
graph_memory/embeddings/dense.py
graph_memory/retrieval/methods/flat/dense.py
graph_memory/registry/retrieval.py
graph_memory/registry/retrieval_builders.py
graph_memory/registry/training.py
graph_memory/registry/stage_configs.py
graph_memory/stages/train.py
scripts/workflow/artifacts.py
scripts/workflow/manifest.py
scripts/workflow/registry.py
scripts/workflow/stage_configs.py
scripts/workflow/workflows.py
configs/experiments/hotpotqa_evidence_retrieval.json
docs/40-operations/commands.md
```

删除文件：

```text
scripts/train_graph_retriever.py
```

## 4. 设计细节

### 4.1 Dense 文本格式化

新增两个 dense-owned helper，供推理和 dense-ft 训练共用：

```python
def format_dense_query(task_input: MemoryTaskInput, *, query_prefix: str) -> str:
    return query_prefix + task_input["query"]


def format_dense_passage(memory_item: MemoryItem, *, passage_prefix: str) -> str:
    return passage_prefix + f'{memory_item["source"]}. {memory_item["text"]}'
```

实现落点：`graph_memory/embeddings/dense.py`。

验收：

- `DenseEncodingService._texts_for_request()` 使用这两个 helper。
- dense-ft data builder 使用同一 helper。
- 不在 dense-ft 包里重新拼接 `source/text`。

### 4.2 Dense-FT 训练样本构造

新增数据类型：

```python
@dataclass(frozen=True)
class DenseFinetuneExample:
    task_id: TaskId
    positive_node_id: NodeId
    negative_node_id: NodeId | None
    anchor: str
    positive: str
    negative: str | None
    negative_sample_type: TrainPairSampleType | None
```

构造规则：

- 输入：`train_task_inputs`、`train_labels`、`train_pairs`、`DenseFinetuneDataSettings`。
- 对每个 task：
  - positive 来自 `train_pairs` 中 `label=1` 的 node。
  - negative 来自同 task 下 `label=0` 的 node。
  - positive 必须能在 task memory item 中找到。
  - negative 必须能在 task memory item 中找到。
- 每个 positive 生成至少一条训练 row。
- 如果同 task 有 negative，则每个 positive 绑定最多 `hard_negatives_per_positive` 条 negative，默认 `1`。
- 如果同 task 没有 negative，则生成 `(anchor, positive)` row。
- 输出给 SentenceTransformers 的 train dataset 字段固定为：
  - 有 negative：`anchor`、`positive`、`negative`
  - 无 negative：`anchor`、`positive`

排序规则：

- positives 按 `train_pairs` 原始顺序保留。
- negatives 按 `sample_type` 优先级和原始顺序选择：
  - `hard_dense`
  - `hard_bm25`
  - `hard_graph_neighbor`
  - `easy_random`
- 同一 `(task_id, positive_node_id, negative_node_id)` 不重复输出。

### 4.3 Dev IR evaluator 构造

`InformationRetrievalEvaluator` 需要三组数据：

```text
queries: dict[str, str]
corpus: dict[str, str]
relevant_docs: dict[str, set[str]]
```

构造规则：

- query id 使用 `task_id`。
- corpus id 使用 `<task_id>::<node_id>`，避免不同 task 的 `m0` 冲突。
- relevant doc id 来自 `gold_evidence_nodes` 映射后的 `<task_id>::<node_id>`。
- corpus 文本使用 `format_dense_passage()`。
- query 文本使用 `format_dense_query()`。

### 4.4 Dense-FT model metadata

训练产物目录中保存额外 metadata：

```text
<model_dir>/
  config_sentence_transformers.json
  modules.json
  ...
  dense_ft_model_config.json
```

`dense_ft_model_config.json` 内容：

```json
{
  "schema_version": 1,
  "method": "dense_ft",
  "base_model": "models/intfloat-e5-base-v2",
  "query_prefix": "query: ",
  "passage_prefix": "passage: ",
  "batch_size": 64,
  "selection": {
    "selected_metric": "eval_dev_cos_sim_map@100",
    "higher_is_better": true
  }
}
```

检索 builder 从该 metadata 读取 prefix 和 batch size，再构造 `DenseConfig`。这样 retrieve 命令只需要 `--checkpoint <model_dir>`，不会要求用户重复传训练时的 prefix。

### 4.5 Training config shape

新增 `DenseFinetuneMethodSettings`：

```python
@dataclass(frozen=True)
class DenseFinetuneDataSettings:
    hard_negatives_per_positive: int = 1


@dataclass(frozen=True)
class DenseFinetuneTrainerSettings:
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 1
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    random_seed: int = 13
    device: str = "cuda"
    use_amp: bool = False


@dataclass(frozen=True)
class DenseFinetuneMethodSettings:
    encoder: DenseEncoderSettings
    data: DenseFinetuneDataSettings
    trainer: DenseFinetuneTrainerSettings
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT
```

新增 config：

```text
configs/training/dense_ft/base.json
```

建议默认：

约定：`profiles` 是对 `defaults` 的 override，只写和默认值不同的字段。不要在 profile 里重复默认值；例如 `defaults.trainer.device` 已经是 `"cuda"`，则 `quick`、`cloud-full` 等 profile 不再写 `"device": "cuda"`，只有需要改成 CPU 的 `smoke` profile 才写 `"device": "cpu"`。

```json
{
  "schema_version": 1,
  "method": "dense_ft",
  "default_profile": "quick",
  "defaults": {
    "encoder": {
      "model": "models/intfloat-e5-base-v2",
      "query_prefix": "query: ",
      "passage_prefix": "passage: ",
      "batch_size": 64
    },
    "data": {
      "hard_negatives_per_positive": 1
    },
    "trainer": {
      "learning_rate": 0.00002,
      "train_batch_size": 16,
      "eval_batch_size": 64,
      "epochs": 1,
      "warmup_steps": 0,
      "max_grad_norm": 1.0,
      "random_seed": 13,
      "device": "cuda",
      "use_amp": false
    },
    "selection": {
      "best_metric": "eval_dev_cos_sim_map@100",
      "higher_is_better": true
    }
  },
  "profiles": {
    "smoke": {
      "trainer": {
        "train_batch_size": 1,
        "eval_batch_size": 4,
        "epochs": 1,
        "device": "cpu"
      }
    },
    "quick": {},
    "cloud-full": {
      "trainer": {
        "train_batch_size": 64,
        "eval_batch_size": 128,
        "epochs": 2
      }
    }
  }
}
```

### 4.6 Train stage config shape

推荐把 TRAIN stage 改成根级 discriminated union：

```python
@dataclass(frozen=True)
class RgcnTrainIO:
    train_tasks: Path
    train_labels: Path | None
    train_graphs: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    dev_graphs: Path
    output_dir: Path
    checkpoint_dir: Path
    metrics: Path
    run_summary: Path
    config: Path | None = None


@dataclass(frozen=True)
class DenseFinetuneTrainIO:
    train_tasks: Path
    train_labels: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    output_dir: Path
    model_dir: Path
    metrics: Path
    run_summary: Path
    config: Path | None = None


@dataclass(frozen=True)
class RgcnTrainStageConfig:
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER]
    io: RgcnTrainIO
    job: RgcnMethodSettings


@dataclass(frozen=True)
class DenseFinetuneTrainStageConfig:
    method: Literal[RetrievalMethodId.DENSE_FT]
    io: DenseFinetuneTrainIO
    job: DenseFinetuneMethodSettings


TrainStageConfig = RgcnTrainStageConfig | DenseFinetuneTrainStageConfig
```

`ConfigConverter` 已支持根据根对象的 `method` 字段匹配 union dataclass；实现时只需要保证 `_train_cli_patch()` 和 `_normalize_train_raw_config()` 都把 `method` 写到 stage root。

统一入口要求：

- 所有 train workflow 只生成 `scripts/train_method.py --method <method>`。
- `scripts/train_graph_retriever.py` 不保留兼容 shim，避免两个 train adapter 长期分叉。
- 低层直接训练命令也必须显式传 `--method dense_rgcn_graph_retriever` 或 `--method dense_ft`。
- `CONFIG_LOADER.load(Registry.configs.TRAIN, argv)` 仍是唯一 train config 入口，`ConfigConverter` 按 root `method` 字段选择具体 stage config。

## 5. 任务分解

### Task 1: 抽出 dense 文本格式化 helper

**Files:**

- Modify: `graph_memory/embeddings/dense.py`
- Modify: `graph_memory/retrieval/methods/flat/dense.py`
- Test: `tests/test_batched_dense_encoding.py`
- Test: `tests/test_dense_finetune_data.py`

- [ ] Step 1.1 添加 failing test：同一个 task 下，dense 推理和 dense-ft 数据构造得到完全相同的 query/passage 文本。
- [ ] Step 1.2 在 `graph_memory/embeddings/dense.py` 新增 `format_dense_query()` 和 `format_dense_passage()`。
- [ ] Step 1.3 改造 `DenseEncodingService._texts_for_request()` 使用 helper。
- [ ] Step 1.4 运行 `uv run pytest tests/test_batched_dense_encoding.py tests/test_dense_finetune_data.py -q`。

验收：

- dense 原有排序测试不变。
- 不新增任何 dense-ft 专属文本拼接代码。

### Task 2: 新增 dense-ft data builder

**Files:**

- Create: `graph_memory/models/dense_finetune/__init__.py`
- Create: `graph_memory/models/dense_finetune/contracts.py`
- Create: `graph_memory/models/dense_finetune/data.py`
- Test: `tests/test_dense_finetune_data.py`

- [ ] Step 2.1 写测试覆盖 positive-only、positive+hard negatives、跨 task node id 冲突、unknown node id 报错。
- [ ] Step 2.2 实现 `DenseFinetuneExample`、`DenseFinetuneDatasetBuildResult`、`DenseFinetuneDataSettings`。
- [ ] Step 2.3 实现 `build_dense_finetune_examples()`。
- [ ] Step 2.4 实现 `build_ir_evaluator_payload()`。
- [ ] Step 2.5 运行 `uv run pytest tests/test_dense_finetune_data.py -q`。

验收：

- 输出 row 数量由 train_pairs 决定，可预测。
- corpus id 使用 `<task_id>::<node_id>`。
- 负样本优先级固定且有测试保护。

### Task 3: 补依赖与训练包骨架

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `graph_memory/models/dense_finetune/training.py`
- Test: `tests/test_dense_finetune_training.py`

- [ ] Step 3.1 在 `pyproject.toml` 固定 `sentence-transformers==2.7.0`，不引入 `datasets` 和 `accelerate`。
- [ ] Step 3.2 运行 `uv lock` 更新 `uv.lock`。
- [ ] Step 3.3 写 fake-model/fake-trainer 测试，先验证 dense-ft training result、metadata 写出路径和 run metrics 记录格式。
- [ ] Step 3.4 实现训练包骨架，不加载真实模型。
- [ ] Step 3.5 运行 `uv run pytest tests/test_dense_finetune_training.py -q`。

验收：

- `uv run python -c "import sentence_transformers; assert sentence_transformers.__version__ == '2.7.0'"` 成功。
- dense-ft training result 不依赖 R-GCN model config。

### Task 4: 改造 training registry 与 TRAIN stage config

**Files:**

- Modify: `graph_memory/registry/training.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: `graph_memory/stages/train.py`
- Test: `tests/test_registry_stage_configs.py`
- Test: `tests/test_dense_finetune_training.py`

- [ ] Step 4.1 写测试：`Registry.configs.TRAIN` 能解析 R-GCN config，也能解析 dense-ft config。
- [ ] Step 4.2 写测试：`TrainJobSettings` union 能按 `method` 分发到 `RgcnMethodSettings` 或 `DenseFinetuneMethodSettings`。
- [ ] Step 4.3 新增 `DenseFinetuneMethodSettings`、data/trainer/selection settings。
- [ ] Step 4.4 把 `TrainStageConfig` 改为根级 discriminated union。
- [ ] Step 4.5 改造 `TrainingRegistry.build()`，让 builder 不接收全局 `TrainDependencies`。
- [ ] Step 4.6 把 R-GCN provider 构造移动到 R-GCN trainer/builder 内部，`scripts` 不再构造 `DenseGraphFeatureProvider`。
- [ ] Step 4.7 运行 `uv run pytest tests/test_registry_stage_configs.py tests/test_dense_finetune_training.py -q`。

验收：

```powershell
rg -n "DenseGraphFeatureProvider|RetrieverSeedSignalProvider|DenseFinetune|dense_ft" scripts graph_memory/stages
```

Expected:

- `scripts/train_method.py` 不构造 R-GCN provider。
- `graph_memory/stages/train.py` 不出现 dense-ft 或 R-GCN method string 分支。
- 具体 method 分支收敛到 registry/trainer 实现。

### Task 5: 实现真实 SentenceTransformers 微调

**Files:**

- Modify: `graph_memory/models/dense_finetune/training.py`
- Test: `tests/test_dense_finetune_training.py`

- [ ] Step 5.1 用 `InputExample` 和 PyTorch `DataLoader` 构造训练输入。
- [ ] Step 5.2 用 `SentenceTransformer(config.encoder.model_name, device=trainer.device)` 加载 base model。
- [ ] Step 5.3 设备行为由 `SentenceTransformer(..., device=...)` 直接控制。
- [ ] Step 5.4 使用 `MultipleNegativesRankingLoss(model)`。
- [ ] Step 5.5 使用 `InformationRetrievalEvaluator(name="dev", main_score_function="cos_sim", ndcg_at_k=[10], accuracy_at_k=[1,3,5,10], precision_recall_at_k=[1,3,5,10])`；2.7.0 返回该 score function 的 MAP@100。
- [ ] Step 5.6 调用 `SentenceTransformer.fit()`，传入 train objectives、evaluator、epochs、learning rate、warmup steps、max grad norm 和 AMP。
- [ ] Step 5.7 训练完成后保存 selected model directory，并写 `dense_ft_model_config.json`。
- [ ] Step 5.8 运行 `uv run pytest tests/test_dense_finetune_training.py -q`。

验收：

- smoke profile 能在 CPU 上用 fake/minimal 数据跑完。
- 训练输出包含 SentenceTransformer 可加载目录。
- `dense_ft_model_config.json` 包含 prefix、base model、batch size、selection 信息。

### Task 6: 接入 dense-ft retrieval registry

**Files:**

- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Test: `tests/test_dense_ft_retrieval_registry.py`
- Test: `tests/test_retrieval_registry_builders.py`

- [ ] Step 6.1 新增 `RetrievalMethodId.DENSE_FT = "dense_ft"`。
- [ ] Step 6.2 新增 `DenseFinetunedRetrievalSettings`，字段为 `top_k`、`checkpoint`、`device`。
- [ ] Step 6.3 新增 metadata：`requires_graphs=False`、`requires_graph_config=False`、`requires_checkpoint=True`、`requires_dense_encoder=True`、`seed_method=RetrievalMethodId.DENSE`。
- [ ] Step 6.4 新增 retrieval builder：读取 `<checkpoint>/dense_ft_model_config.json`，构造 `DenseTaskRetriever(config=DenseConfig(model_name=str(checkpoint), ...))`。
- [ ] Step 6.5 运行 `uv run pytest tests/test_dense_ft_retrieval_registry.py tests/test_retrieval_registry_builders.py -q`。

验收：

- `run_retrieval --method dense_ft --checkpoint <model_dir>` 不要求 `--graphs`。
- dense-ft 排序输出 schema 与 `dense` 完全一致。
- metadata 缺失时错误信息包含 `dense_ft_model_config.json` 和 checkpoint 路径。

### Task 7: 统一 train script 并删除旧入口

**Files:**

- Create: `scripts/train_method.py`
- Delete: `scripts/train_graph_retriever.py`
- Test: `tests/test_phase2_rgcn_training.py`
- Test: `tests/test_dense_finetune_training.py`
- Test: `tests/test_cli_contracts.py`

- [ ] Step 7.1 `scripts/train_method.py` 使用 `CONFIG_LOADER.load(Registry.configs.TRAIN, argv)`。
- [ ] Step 7.2 根据 stage config 的 IO 类型读取对应 artifact。
- [ ] Step 7.3 调用 `run_train_stage(config, payload=...)`。
- [ ] Step 7.4 统一写 metrics 和 run summary。
- [ ] Step 7.5 删除 `scripts/train_graph_retriever.py`，同步更新所有测试、workflow 命令和文档引用。
- [ ] Step 7.6 R-GCN 训练命令改为显式 `scripts/train_method.py --method dense_rgcn_graph_retriever`。
- [ ] Step 7.7 dense-ft 训练命令使用 `scripts/train_method.py --method dense_ft`。
- [ ] Step 7.8 运行 `uv run pytest tests/test_phase2_rgcn_training.py tests/test_dense_finetune_training.py tests/test_cli_contracts.py -q`。

验收：

- 仓库内不再引用 `scripts/train_graph_retriever.py`，历史文档除外。
- R-GCN train workflow 通过统一入口继续可用。
- 新 dense-ft train CLI 产出 model directory。
- run summary 的 `script` 字段统一为 `train_method.py`，method 由 resolved config 记录。

### Task 8: 接入 workflow、manifest 和 artifact paths

**Files:**

- Modify: `scripts/workflow/types.py`
- Modify: `scripts/workflow/artifacts.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `scripts/workflow/registry.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `scripts/workflow/manifest.py`
- Test: `tests/test_dense_ft_workflow.py`
- Test: `tests/test_workflow_orchestration.py`
- Test: `tests/test_experiment_runner.py`

- [ ] Step 8.1 新增 `WorkflowId.DENSE_FINETUNE_RETRIEVAL`。
- [ ] Step 8.2 新增 `DENSE_FT_WORKFLOW`：`prepare -> graphs -> pairs -> train -> retrieve -> evaluate -> aggregate`。
- [ ] Step 8.3 在 `METHOD_WORKFLOW_REGISTRY` 注册 `dense_ft`。
- [ ] Step 8.4 `build_main_method_artifacts()` 对 dense-ft 使用 `learned/dense_ft/checkpoints/best_model` 作为 checkpoint/model-dir。
- [ ] Step 8.5 `build_train_commands()` 对所有 train method 使用 `scripts/train_method.py --method <method>`。
- [ ] Step 8.6 `build_retrieve_commands()` 对 dense-ft 传 `--checkpoint <model_dir>`，不传全局 `--encoder_model`。
- [ ] Step 8.7 `attach_stage_config_projections()` 能生成 dense-ft train/retrieve projection。
- [ ] Step 8.8 运行 `uv run pytest tests/test_dense_ft_workflow.py tests/test_workflow_orchestration.py tests/test_experiment_runner.py -q`。

验收：

- `scripts/experiment.py methods list` 展示 `dense_ft`。
- `scripts/experiment.py plan <name> --methods dense_ft --profile smoke` 生成 pairs/train/retrieve/evaluate/aggregate 命令。
- 从 retrieve stage resume 时，如果 model directory 不存在，错误指向 dense-ft checkpoint/model-dir。

### Task 9: 更新实验配置和文档

**Files:**

- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Create: `configs/training/dense_ft/base.json`
- Create: `docs/configs/training/dense_ft/base.md`
- Modify: `docs/40-operations/commands.md`
- Test: `tests/test_experiment_runner.py`

- [ ] Step 9.1 在 experiment config 的 `methods` 中加入 `dense_ft`。
- [ ] Step 9.2 在 `training_configs` 中加入 `"dense_ft": "configs/training/dense_ft/base.json"`。
- [ ] Step 9.3 写 dense-ft training config 字段说明。
- [ ] Step 9.4 在 commands 文档新增 dense-ft smoke/quick/full 运行命令。
- [ ] Step 9.5 运行 `uv run pytest tests/test_experiment_runner.py -q`。

验收：

- config list 能发现 `dense_ft/base`。
- repository profile 能解析 dense-ft training config。
- 文档明确 `checkpoint` 对 dense-ft 是 model directory。

### Task 10: 全量验证

在 Windows 主机上执行 `uv` 命令时遵守仓库 AGENTS.md：从第一次尝试就使用非 Codex filesystem sandbox 的正常用户环境。

验收命令：

```powershell
uv run pytest tests/test_dense_finetune_data.py tests/test_dense_finetune_training.py tests/test_dense_ft_retrieval_registry.py tests/test_dense_ft_workflow.py -q
uv run pytest tests/test_batched_dense_encoding.py tests/test_registry_stage_configs.py tests/test_retrieval_registry_builders.py -q
uv run pytest tests/test_experiment_runner.py tests/test_workflow_orchestration.py tests/test_phase2_rgcn_training.py -q
uv run basedpyright graph_memory scripts tests --level error
uv run python scripts/experiment.py methods list
uv run python scripts/experiment.py plan dense_ft_smoke --profile smoke --methods dense_ft --force
```

真实训练 smoke：

```powershell
uv run python scripts/experiment.py run dense_ft_smoke --profile smoke --methods dense_ft --force
```

完成标准：

- `dense` 与 `dense_ft` 同时出现在 method list 和 main table 输入中。
- `dense_ft` 的 retrieval output 使用统一 `RankedResult` schema。
- dense-ft 训练产物可被 `SentenceTransformer(<model_dir>)` 加载。
- R-GCN 训练、检索、workflow 旧测试仍通过。
- `graph_memory/stages/train.py` 不直接感知 R-GCN 或 dense-ft 细节。

## 6. 风险与处理

### 风险 1: TRAIN stage 改造范围大于 dense-ft 本身

原因：当前 TRAIN stage 是 R-GCN 形状，dense-ft 接入会暴露这个历史问题。

处理：先做 registry/stage config 的最小精确拆分，只拆 TRAIN root config 和 trainer payload，不重构其他 stage。

### 风险 2: SentenceTransformers evaluator metric name 与预期不一致

原因：SentenceTransformers 2.7.0 `InformationRetrievalEvaluator` 返回主指标标量。

处理：训练实现把 evaluator 返回的标量记录到配置指定的 `selection.best_metric`，不引入版本分支。

### 风险 3: 本地依赖缺失

原因：部署环境固定使用国产卡适配的 SentenceTransformers 2.7.0。

处理：项目固定 `sentence-transformers==2.7.0`，训练只使用该版本已有的 `InputExample`、`DataLoader` 和 `fit()` API。

### 风险 4: dense-ft checkpoint 是目录，现有 checkpoint 文案偏 `.pt`

原因：workflow 的 artifact role 叫 `CHECKPOINT`，R-GCN 使用 `best.pt`。

处理：保留 artifact role 名称以减少 workflow 改动，但在 dense-ft docs、run summary 和错误信息中明确 checkpoint 是 SentenceTransformer model directory。

### 风险 5: dense-ft 训练数据过大

原因：full profile 下 train pairs 和 hard negatives 可能很多。

处理：第一版用 `hard_negatives_per_positive` 限制每个 positive 使用的 negative 数量，默认 `1`；需要扩大时改 config，不改代码。

## 7. 审查重点

请重点审查这四个决策：

1. `dense_ft` 是否应作为独立公开方法，而不是 `dense` 的 config variant。
2. dense-ft checkpoint 是否接受 model directory，并继续复用 `--checkpoint` 这个 CLI 名称。
3. TRAIN stage 是否按本文改成 root-level method-specific config union。
4. 第一版 loss 是否采用 `MultipleNegativesRankingLoss`，并只使用每个 positive 的有限 hard negatives。

只要这四点确认，后面的实现基本是局部工程落地。
