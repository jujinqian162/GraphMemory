# Trainable Stack 零兼容重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 彻底删除 R-GCN 与 Dense-FT trainable stack 中的版本迁移、旧格式读取、兼容 facade、兼容 projection、legacy manifest fallback 和 method 特判补丁，只保留一套当前配置、一套 typed registry、一套严格 workflow manifest 和一套可追溯的训练/检索 artifact contract。

**Architecture:** `ConfigLoader.load(spec, argv)` 继续作为唯一配置加载入口，但配置只接受当前结构，未知字段、缺失字段和旧字段立即失败。`graph_memory.registry` 直接暴露当前 method definition、method config、runtime builder 与 artifact semantics，不再通过 `training_config.py`、`retrieval_registry.py`、`retrieval/catalog.py` 或 `registry/projections.py` 提供兼容视图。workflow 初始化阶段把 experiment config 和 method config 编译为完整 typed stage config，写入当前 run 目录；planner 只消费这些 stage config，不再保留旧 manifest argv assembly。

**Tech Stack:** Python 3.10、dataclass、`typing_extensions.assert_never`、标准库 `argparse` / JSON、PyTorch checkpoint、SentenceTransformers 2.7.0、pytest、Ruff、basedpyright、OpenSpec。

---

日期：2026-06-12

状态：待用户审核。本文只定义重构范围和实施顺序，不代表代码已经修改，也不创建 OpenSpec change。

## 1. 已确认的硬性原则

本计划以下列约束为最高优先级：

1. 这是可重跑的实验程序，不为旧 config、旧 manifest、旧 checkpoint、旧模型目录或旧中间产物保留读取能力。
2. 删除所有 trainable stack 中的版本字段：
   - method config 的 `schema_version`
   - workflow manifest 的 `schema_version`
   - ablation metrics index 的 `schema_version`
   - Dense-FT metadata 的 `schema_version`
   - R-GCN checkpoint 的 `checkpoint_version`
3. 不提供 migration、adapter、fallback、alias、双路径 parser 或“先尝试新格式，再尝试旧格式”的逻辑。
4. 结构校验必须保留并加强：
   - JSON root 类型必须正确
   - dataclass 必填字段必须存在
   - unknown field 必须失败
   - union 必须按 `method` 穷尽匹配
   - checkpoint/metadata 必须通过当前 contract 校验
5. 旧 artifact 的处理方式是删除并重跑，不是转换。
6. 本轮覆盖整个 trainable stack：
   - `dense_rgcn_graph_retriever`
   - `dense_ft`
   - 二者共用的 pair、train、retrieve、workflow、manifest、artifact、registry 和 observability 边界
7. 本轮明确不处理：
   - HotpotQA combined compatibility output
   - `--gold` 等与 trainable stack 无关的 CLI alias
   - Python 3.10 的 `StrEnum` / `NotRequired` 适配
   - SentenceTransformers 2.7 forward hook loss observer
   - 非 trainable 数据 artifact 的格式重构

## 2. 当前问题不是单点，而是一条兼容链

当前 trainable stack 存在以下并行表示：

```text
configs/training/.../base.json
  -> graph_memory/config/training_compat.py
  -> graph_memory/training_config.py
  -> scripts/workflow/manifest.py 中的 raw dict
  -> effective_training_config.json
  -> graph_memory/registry/stage_configs.py legacy normalizer
  -> typed stage config
```

同时 retrieval/workflow 又存在第二条兼容链：

```text
graph_memory/registry/retrieval.py
  -> graph_memory/registry/projections.py
  -> graph_memory/retrieval_registry.py
  -> graph_memory/retrieval/catalog.py
  -> scripts/workflow/*
```

workflow command 还存在两条路径：

```text
新 manifest
  -> manifest["stage_configs"]
  -> typed projection -> argv

旧 manifest
  -> manifest["artifacts"] + method capability bool
  -> legacy argv assembly
```

这三条链导致：

- config 字段存在 `model` / `model_name`、`optimization` / `trainer`、`pair_sampling` / `pairs` 等 alias。
- Dense-FT 作为新方法仍通过旧 `defaults` normalizer。
- workflow 同时维护 typed projection 和 legacy command assembly。
- `builder_id`、`requires_checkpoint`、`requires_dense_encoder` 继续作为兼容 metadata。
- artifact basename、文件/目录类型和 encoder 来源只能通过 `method == "dense_ft"` 补足。
- run summary 无法从统一接口得到实际运行时 provenance。

本轮目标不是继续限制这些 compatibility surface，而是删除整条链。

## 3. 目标依赖方向

```text
configs/experiments/*.json
  -> scripts/workflow/manifest.py
  -> CONFIG_LOADER.load(Registry.configs.TRAINABLE_METHOD, argv)
  -> typed TrainableMethodConfig
  -> workflow stage config compiler
  -> typed Pair/Train/Retrieve/Evaluate stage configs
  -> runs/<name>/config/stages/*.json
  -> planner generic command
  -> scripts/*.py --config <stage-config>
  -> CONFIG_LOADER.load(Registry.configs.<STAGE>, argv)
  -> stage runner
  -> registry runtime builder
  -> method implementation
```

约束：

- method config 只由 workflow compiler 读取。
- low-level script 的 `--config` 永远表示完整 stage-root config，不再有时表示 method config、有时表示 resolved config。
- workflow command 不再重新拼 method-specific 参数。
- stage config 是 workflow 和 low-level script 之间唯一的执行 contract。
- method config 是 experiment compiler 的输入，不是 runtime fallback。

## 4. 唯一 method config 结构

### 4.1 文件位置

只保留：

```text
configs/methods/dense_rgcn_graph_retriever.json
configs/methods/dense_ft.json
```

删除：

```text
configs/training/dense_rgcn_graph_retriever/base.json
configs/training/dense_rgcn_graph_retriever/ablations.json
configs/training/dense_ft/base.json
```

experiment config 中的映射改名为：

```json
"method_configs": {
  "dense_rgcn_graph_retriever": "configs/methods/dense_rgcn_graph_retriever.json",
  "dense_ft": "configs/methods/dense_ft.json"
}
```

不保留 `training_configs` 输入别名。

### 4.2 R-GCN 当前结构

```json
{
  "method": "dense_rgcn_graph_retriever",
  "default_profile": "quick",
  "encoder": {
    "model_name": "models/intfloat-e5-base-v2",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "batch_size": 64
  },
  "pairs": {
    "random_seed": 13,
    "easy_random_per_positive": 2,
    "hard_bm25_per_positive": 2,
    "hard_dense_per_positive": 0,
    "hard_graph_neighbor_per_positive": 1,
    "hard_pool_size": 30
  },
  "train": {
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
    "reporting": {
      "render_training_curves": true
    },
    "selection": {
      "best_metric": "dev_composite",
      "higher_is_better": true
    }
  },
  "profiles": {
    "quick": {},
    "smoke": {
      "train": {
        "model": {
          "hidden_dim": 32,
          "num_layers": 1
        },
        "trainer": {
          "batch_size": 1,
          "epochs": 1
        }
      },
      "pairs": {
        "easy_random_per_positive": 1,
        "hard_bm25_per_positive": 1,
        "hard_graph_neighbor_per_positive": 1
      }
    }
  }
}
```

### 4.3 Dense-FT 当前结构

```json
{
  "method": "dense_ft",
  "default_profile": "quick",
  "encoder": {
    "model_name": "models/intfloat-e5-base-v2",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "batch_size": 64
  },
  "pairs": {
    "random_seed": 13,
    "easy_random_per_positive": 2,
    "hard_bm25_per_positive": 2,
    "hard_dense_per_positive": 0,
    "hard_graph_neighbor_per_positive": 1,
    "hard_pool_size": 30
  },
  "train": {
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
    "quick": {},
    "smoke": {
      "train": {
        "trainer": {
          "train_batch_size": 1,
          "eval_batch_size": 4,
          "device": "cpu"
        }
      }
    }
  }
}
```

### 4.4 结构规则

- root 本身就是 base，不存在 `defaults` wrapper。
- `profiles` 只做 deep patch。
- `default_profile` 是当前固定字段，不提供其他名字。
- `model_name`、`trainer`、`pairs` 是唯一字段名。
- 不接受 `model` 作为 encoder model alias。
- 不接受 `optimization`。
- 不接受 `pair_sampling`。
- 不接受 `schema_version`。
- resolved method config 删除 `default_profile` 与 `profiles` 后写入：

```text
runs/<experiment>/config/methods/<method>.json
```

artifact role 从 `EFFECTIVE_TRAINING_CONFIG` 改为 `EFFECTIVE_METHOD_CONFIG`，文件名从 `effective_training_config.json` 改为 `effective_method_config.json`。

## 5. Typed method config

新增：

```python
@dataclass(frozen=True)
class RgcnTrainSettings:
    model: RgcnModelSettings
    trainer: RgcnTrainerSettings
    reporting: TrainingReportingSettings
    selection: ModelSelectionSettings


@dataclass(frozen=True)
class DenseFinetuneTrainSettings:
    data: DenseFinetuneDataSettings
    trainer: DenseFinetuneTrainerSettings
    selection: DenseFinetuneSelectionSettings


@dataclass(frozen=True)
class RgcnMethodConfig:
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER]
    encoder: DenseEncoderSettings
    pairs: RgcnPairSamplingSettings
    train: RgcnTrainSettings


@dataclass(frozen=True)
class DenseFinetuneMethodConfig:
    method: Literal[RetrievalMethodId.DENSE_FT]
    encoder: DenseEncoderSettings
    pairs: RgcnPairSamplingSettings
    train: DenseFinetuneTrainSettings


TrainableMethodConfig = RgcnMethodConfig | DenseFinetuneMethodConfig
```

说明：

- `TrainableMethodConfig` 表示一个方法完整的 pairs + train 生命周期配置。
- `TrainStageConfig.job` 仍然只保存训练需要的精确 settings。
- pair stage compiler 从 method config 提取 `encoder` 与 `pairs`。
- train stage compiler 从 method config 提取 `encoder` 与 `train`。
- 不把 pair sampling 塞进 Dense-FT trainer。
- 不把 Dense-FT data settings 塞进 R-GCN config。

## 6. ConfigLoader 收口

### 6.1 保留的公开入口

```python
CONFIG_LOADER.load(spec, argv)
CONFIG_LOADER.to_json(config)
```

新增 registry config spec：

```python
Registry.configs.TRAINABLE_METHOD
```

它使用只包含 `--config` 和 `--profile` 的 parser，将当前 method config 结构解析为 `TrainableMethodConfig`。

### 6.2 删除的机制

删除：

```text
graph_memory/config/training_compat.py
graph_memory/training_config.py
StageConfigSpec.normalize_raw_config
_resolve_legacy_training_config()
_legacy_profile_name()
_string_config_alias()
_json_bool_alias()
load_trainable_training_config()
resolve_trainable_training_config()
device_from_training_config()
training_config_required_sections()
```

`ConfigLoader` 不再调用任何 legacy normalizer。method config 和 stage config 都必须直接匹配 target dataclass。

### 6.3 严格失败要求

以下输入必须失败：

```json
{"schema_version": 2, "...": "..."}
{"defaults": {}, "profiles": {}}
{"encoder": {"model": "..."}}
{"optimization": {}}
{"pair_sampling": {}}
```

测试必须断言错误包含 unsupported field 或 missing required field，而不是被静默转换。

## 7. Method registry 重新建模

### 7.1 删除 compatibility projection

删除：

```text
graph_memory/registry/projections.py
graph_memory/retrieval_registry.py
graph_memory/retrieval/catalog.py
```

所有调用方直接使用 `graph_memory.registry.methods` 或 `Registry.methods`。

### 7.2 当前 method definition

新增 `graph_memory/registry/methods.py`：

```python
class RetrievalLifecycle(StrEnum):
    STATELESS = "stateless"
    GRAPH_RERANK = "graph_rerank"
    RGCN_TRAINABLE = "rgcn_trainable"
    DENSE_FINETUNE = "dense_finetune"


class GraphInputSource(StrEnum):
    NONE = "none"
    GRAPH_ARTIFACT = "graph_artifact"


class GraphConfigSource(StrEnum):
    NONE = "none"
    TUNED_ARTIFACT = "tuned_artifact"


class ModelSource(StrEnum):
    NONE = "none"
    CHECKPOINT_FILE = "checkpoint_file"
    MODEL_DIRECTORY = "model_directory"


class EncoderSource(StrEnum):
    NONE = "none"
    EXPERIMENT_CONFIG = "experiment_config"
    CHECKPOINT_METADATA = "checkpoint_metadata"


class ArtifactKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class TrainArtifactSpec:
    basename: str
    kind: ArtifactKind


@dataclass(frozen=True)
class RetrievalDependencySpec:
    graphs: GraphInputSource
    graph_config: GraphConfigSource
    model: ModelSource
    encoder: EncoderSource


@dataclass(frozen=True)
class MethodDefinition:
    identifier: RetrievalMethodId
    lifecycle: RetrievalLifecycle
    retrieval_settings_type: type[object]
    dependencies: RetrievalDependencySpec
    method_config_type: type[object] | None
    train_artifact: TrainArtifactSpec | None
    seed_method: RetrievalMethodId | None = None
```

R-GCN：

```python
MethodDefinition(
    identifier=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
    lifecycle=RetrievalLifecycle.RGCN_TRAINABLE,
    retrieval_settings_type=CheckpointGraphRetrievalSettings,
    dependencies=RetrievalDependencySpec(
        graphs=GraphInputSource.GRAPH_ARTIFACT,
        graph_config=GraphConfigSource.NONE,
        model=ModelSource.CHECKPOINT_FILE,
        encoder=EncoderSource.CHECKPOINT_METADATA,
    ),
    method_config_type=RgcnMethodConfig,
    train_artifact=TrainArtifactSpec("best.pt", ArtifactKind.FILE),
    seed_method=RetrievalMethodId.DENSE,
)
```

Dense-FT：

```python
MethodDefinition(
    identifier=RetrievalMethodId.DENSE_FT,
    lifecycle=RetrievalLifecycle.DENSE_FINETUNE,
    retrieval_settings_type=DenseFinetunedRetrievalSettings,
    dependencies=RetrievalDependencySpec(
        graphs=GraphInputSource.NONE,
        graph_config=GraphConfigSource.NONE,
        model=ModelSource.MODEL_DIRECTORY,
        encoder=EncoderSource.CHECKPOINT_METADATA,
    ),
    method_config_type=DenseFinetuneMethodConfig,
    train_artifact=TrainArtifactSpec("best_model", ArtifactKind.DIRECTORY),
    seed_method=RetrievalMethodId.DENSE,
)
```

### 7.3 删除的旧能力表达

删除：

```text
builder_id
requires_graphs
requires_graph_config
requires_checkpoint
requires_dense_encoder
is_dense_finetune_method()
checkpoint_artifact_name()
get_methods_requiring_dense_encoder()
```

允许保留的 API 只有当前语义：

```python
Registry.methods.list_ids()
Registry.methods.get(method)
Registry.methods.list_by_lifecycle(lifecycle)
```

## 8. Workflow 与 method registry 的关系

workflow 不再维护 `method -> workflow` 平行表。

改为：

```text
MethodDefinition.lifecycle
  -> WORKFLOW_BY_LIFECYCLE
  -> WorkflowSpec
```

`scripts/workflow/registry.py` 只保留：

- lifecycle -> workflow
- ablation suite 查询
- registry 完整性校验

它不再保存：

- `METHOD_WORKFLOW_REGISTRY`
- Dense-FT 判断函数
- checkpoint basename 判断
- runtime method capability projection

artifact path 由 `MethodDefinition.train_artifact` 决定。

## 9. Workflow manifest 当前唯一结构

### 9.1 删除版本和旧 manifest 支持

删除：

```python
manifest["schema_version"]
existing.get("schema_version", 1)
schema-version-1 manifest branch
stage_configs 缺失 fallback
legacy argv assembly
```

旧 manifest 直接失效。用户必须删除 run 目录或使用 `--force` 重建。

### 9.2 typed manifest

新增 `scripts/workflow/contracts.py`，至少定义：

```python
@dataclass(frozen=True)
class StageConfigRef:
    stage: StageId
    method: str | None
    split: str | None
    path: Path


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_name: str
    recipe: str
    profile: str
    created_at: str
    updated_at: str
    paths: ManifestPaths
    selected_methods: tuple[str, ...]
    selected_stages: tuple[StageId, ...]
    effective_config: ExperimentEffectiveConfig
    method_configs: dict[str, TrainableMethodConfig]
    stage_configs: tuple[StageConfigRef, ...]
    artifacts: ManifestArtifacts
    run_units: tuple[RunUnitRecord, ...]
    stage_status: dict[str, JsonValue]
    ablation_suites: dict[str, tuple[str, ...]]
```

要求：

- `load_manifest()` 必须通过 `ConfigConverter.structure()`。
- unknown field 失败。
- 缺失 `stage_configs` 失败。
- `schema_version` 因 unknown field 失败。
- planner 和 status 不再直接接收任意 `dict[str, Any]`。

如果一次把全部 manifest 子结构改成 dataclass 过大，允许在同一 OpenSpec change 内分两批：

1. 先用显式 `validate_current_manifest()` 严格校验。
2. 再把 planner/status 参数改成 dataclass。

最终状态不允许保留 dict fallback。

## 10. Stage config 编译与执行

### 10.1 当前问题

现在 workflow 先用 argv 调 ConfigLoader 生成 projection，再从 projection 反向拼 argv。这是无意义的往返：

```text
manifest -> argv -> typed config -> JSON projection -> argv -> script -> typed config
```

### 10.2 目标

workflow compiler 直接构造 typed stage config，并写到：

```text
runs/<experiment>/config/stages/pairs/<method>.json
runs/<experiment>/config/stages/train/<method>.json
runs/<experiment>/config/stages/retrieve/<method>.json
runs/<experiment>/config/stages/evaluate/<method>.json
```

variant 使用：

```text
runs/<experiment>/ablations/<method>/<variant>/config/stages/*.json
```

planner command 统一为：

```powershell
python scripts/train_method.py --config runs/.../config/stages/train/dense_ft.json
```

同理：

```powershell
python scripts/build_train_pairs.py --config <pair-stage-config>
python scripts/run_retrieval.py --config <retrieve-stage-config>
python scripts/evaluate_retrieval.py --config <evaluate-stage-config>
```

### 10.3 low-level CLI 规则

- `--config` 只表示完整 stage config。
- 不再接受 method config。
- 不再接受旧 effective training config。
- 可以保留完整 CLI 构造 stage config 的能力，供 smoke test 和临时实验使用。
- 当 `--config` 与 CLI 同时提供时，CLI 只覆盖当前正式字段。
- parser 中的字段可以不声明 `required=True`，最终完整性由 dataclass structure 保证。

### 10.4 删除 legacy command builder

`scripts/workflow/workflows.py` 删除所有：

```python
if projection is not None:
    ...
else:
    legacy assembly
```

command builder 只根据：

- WorkflowStepSpec.command_adapter
- StageConfigRef.path

生成命令，不再知道 train method 的 IO 差异。

## 11. TRAIN union 必须穷尽

`scripts/train_method.py` 中以下函数必须显式覆盖每个 union 分支：

- payload load
- method artifact write
- input summary
- output summary
- effective config
- result counts

结构：

```python
def _load_payload(config: TrainStageConfig, ...) -> TrainPayload:
    if isinstance(config, RgcnTrainStageConfig):
        return ...
    if isinstance(config, DenseFinetuneTrainStageConfig):
        return ...
    assert_never(config)
```

同一规则适用于 result：

```python
if isinstance(result, RgcnTrainingResult):
    ...
elif isinstance(result, DenseFinetuneTrainingResult):
    ...
else:
    raise TypeError(...)
```

禁止：

```python
if isinstance(config, RgcnTrainStageConfig):
    ...
return dense_ft_result
```

Ruff 中未使用的 `DenseFinetuneTrainStageConfig` 必须通过真实的显式分支消失，而不是简单删除 import。

## 12. R-GCN checkpoint 当前 contract

### 12.1 删除版本字段

R-GCN checkpoint 删除：

```python
"checkpoint_version": 1
```

validator 删除版本判断，但继续：

- reject unknown fields
- 验证 method
- 验证 state dict
- 验证 epoch/global step/metric
- 验证 model record
- 验证 trainer record

旧 checkpoint 因多出 `checkpoint_version` 而失败。

### 12.2 精确命名

建议同步改名：

```text
TrainableCheckpoint -> RgcnCheckpoint
save_trainable_checkpoint -> save_rgcn_checkpoint
load_trainable_checkpoint -> load_rgcn_checkpoint
TrainableTrainingResult -> RgcnTrainingResult
TrainableModelConfig -> RgcnCheckpointModelRecord
TrainableTrainingConfig -> RgcnCheckpointTrainerRecord
```

这些类型实际上只属于 R-GCN。继续保留 generic `Trainable*` 命名会让 Dense-FT 被错误地看成同一种 checkpoint backend。

不保留旧名字 re-export。

## 13. Dense-FT metadata 当前 contract

新增 `graph_memory/models/dense_finetune/metadata.py`：

```python
@dataclass(frozen=True)
class DenseFinetuneModelMetadata:
    method: Literal[RetrievalMethodId.DENSE_FT]
    base_model: str
    query_prefix: str
    passage_prefix: str
    batch_size: int
    selection: DenseFinetuneSelectionMetadata
```

writer 和 reader 共同使用该类型。

删除：

```json
"schema_version": 1
```

删除 `_metadata_string()`、`_metadata_int()` 这种逐字段 ad hoc parser，改为统一 structure。

旧模型目录因 metadata 多出 `schema_version` 而失败，必须重训。

## 14. Retrieval build result 提供真实 provenance

当前 `run_retrieval.py` 根据 config 类型猜摘要字段，导致 Dense-FT 记录错误。

改为 registry build 返回：

```python
@dataclass(frozen=True)
class RetrievalProvenance:
    checkpoint: Path | None
    model_source: ModelSource
    device: str | None
    encoder_model: str | None
    encoder_source: EncoderSource
    query_prefix: str | None
    passage_prefix: str | None


@dataclass(frozen=True)
class BuiltRetrievalMethod:
    method: RetrievalMethod
    provenance: RetrievalProvenance
```

`Registry.retrieval.build()` 返回 `BuiltRetrievalMethod`。

来源规则：

- BM25：encoder/checkpoint/device 均为 `None`。
- frozen dense：encoder 来自 current config。
- dense graph rerank：encoder 来自 seed config。
- R-GCN：checkpoint 和 encoder 信息来自当前 checkpoint record。
- Dense-FT：checkpoint 是 model directory，device 来自 retrieve settings，encoder/prefix 来自当前 Dense-FT metadata。

`run_retrieve_stage()` 返回 predictions + provenance。`run_retrieval.py` 不再拥有 `_encoder_settings()`、`_checkpoint_path()`、`_device()`。

## 15. Ablation 配置

删除 `configs/training/dense_rgcn_graph_retriever/ablations.json`。

原因：

- 真实 variant 定义已经在 `graph_memory/registry/ablations.py`。
- 该 JSON 只是文档镜像，形成第二 source of truth。
- experiment config 已经能选择 variant。

修正现有 patch 字段：

```python
{"pair_sampling": {...}}
```

改为当前唯一字段：

```python
{"pairs": {...}}
```

ablation metrics index 只保存：

```json
{
  "metrics": [
    {
      "method": "dense_rgcn_graph_retriever",
      "variant": "wo_graph",
      "metrics_path": "..."
    }
  ]
}
```

不含版本字段。

## 16. 明确删除清单

### 16.1 删除生产代码

```text
graph_memory/config/training_compat.py
graph_memory/training_config.py
graph_memory/registry/projections.py
graph_memory/retrieval_registry.py
graph_memory/retrieval/catalog.py
```

### 16.2 删除配置

```text
configs/training/dense_rgcn_graph_retriever/base.json
configs/training/dense_rgcn_graph_retriever/ablations.json
configs/training/dense_ft/base.json
```

如果 `configs/training/` 删除后为空，删除整个目录。

### 16.3 删除文档

```text
docs/configs/training/dense_rgcn_graph_retriever/base.md
docs/configs/training/dense_rgcn_graph_retriever/ablations.md
docs/configs/training/dense_ft/base.md
```

改为：

```text
docs/configs/methods/dense_rgcn_graph_retriever.md
docs/configs/methods/dense_ft.md
```

### 16.4 删除测试

```text
tests/test_config_schema_migration.py
tests/test_retrieval_registry_projections.py
```

这两个测试的目标就是证明兼容层存在，不应改写成继续保护兼容。

替换为：

```text
tests/test_current_method_configs.py
tests/test_method_registry.py
tests/test_current_manifest_contract.py
tests/test_trainable_compatibility_absence.py
```

## 17. 主要修改文件

### Config / registry

```text
graph_memory/config/loader.py
graph_memory/config/converter.py
graph_memory/registry/app.py
graph_memory/registry/specs.py
graph_memory/registry/stage_configs.py
graph_memory/registry/training.py
graph_memory/registry/retrieval.py
graph_memory/registry/retrieval_builders.py
graph_memory/registry/ablations.py
graph_memory/registry/methods.py                 # new
graph_memory/registry/method_configs.py          # new
```

### R-GCN / Dense-FT

```text
graph_memory/models/graph_retriever/checkpoint.py
graph_memory/models/graph_retriever/config/records.py
graph_memory/models/graph_retriever/training.py
graph_memory/models/dense_finetune/training.py
graph_memory/models/dense_finetune/metadata.py   # new
graph_memory/validation/model.py
```

### Stage / scripts

```text
graph_memory/stages/train.py
graph_memory/stages/retrieve.py
scripts/build_train_pairs.py
scripts/train_method.py
scripts/run_retrieval.py
scripts/evaluate_retrieval.py
```

### Workflow

```text
scripts/workflow/contracts.py                    # new
scripts/workflow/types.py
scripts/workflow/registry.py
scripts/workflow/artifacts.py
scripts/workflow/manifest.py
scripts/workflow/stage_configs.py
scripts/workflow/workflows.py
scripts/workflow/planner.py
scripts/workflow/status.py
scripts/workflow/resume.py
```

### Experiment / docs / delivery

```text
scripts/experiment.py
scripts/deliver/collect_run_artifacts.py
configs/experiments/hotpotqa_evidence_retrieval.json
configs/experiments/hotpoqa_dev_full.json
configs/experiments/hotpotqa_rgcn_ablation_selected.json
docs/configs/README.md
docs/configs/experiments/*.md
docs/20-contracts/model-contracts.md
docs/20-contracts/retrieval-contracts.md
docs/20-contracts/phase2-trainable-retriever-contracts.md
docs/30-design/architecture.md
docs/40-operations/commands.md
docs/40-operations/reproducibility.md
```

## 18. 建议的单一 OpenSpec change

change name：

```text
remove-trainable-stack-compatibility
```

不建议拆成多个并行 change。原因：

- config facade、registry projection、manifest fallback 相互依赖。
- 分开合并会产生新的临时 compatibility adapter。
- 用户已经明确不接受过渡兼容。

该 change 下建立五组 capability spec：

```text
current-trainable-method-config
current-method-registry
current-workflow-manifest
current-trainable-artifacts
trainable-runtime-provenance
```

实施可以分批提交，但只有全部完成后才算 change 完成。

## 19. 实施任务

### Task 1: 冻结零兼容契约

**Files:**

- Create: `tests/test_trainable_compatibility_absence.py`
- Modify: `tests/test_config_registry_architecture.py`
- Modify: `tests/test_core_refactor_final_boundaries.py`

- [ ] 写失败测试，断言待删除 module 当前仍可 import。
- [ ] 写失败测试，扫描 trainable config / manifest / metadata / checkpoint 中的 `schema_version` 与 `checkpoint_version`。
- [ ] 写失败测试，扫描 `defaults`、`builder_id`、legacy fallback 和字段 alias helper。
- [ ] 写失败测试，断言旧 config path 当前仍存在。
- [ ] 运行：

```powershell
uv run pytest tests/test_trainable_compatibility_absence.py -q
```

预期：FAIL，列出当前所有兼容残留。

### Task 2: 建立唯一 method config union

**Files:**

- Create: `graph_memory/registry/method_configs.py`
- Modify: `graph_memory/registry/training.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Create: `tests/test_current_method_configs.py`

- [ ] 先写 R-GCN 和 Dense-FT current config structure 测试。
- [ ] 定义 `TrainableMethodConfig` union。
- [ ] 增加 `Registry.configs.TRAINABLE_METHOD`。
- [ ] 删除 method config alias 和 legacy normalizer。
- [ ] 增加旧字段失败测试。
- [ ] 运行：

```powershell
uv run pytest tests/test_current_method_configs.py tests/test_config_loader.py tests/test_registry_stage_configs.py -q
```

### Task 3: 迁移唯一配置文件并删除旧目录

**Files:**

- Modify: `configs/methods/dense_rgcn_graph_retriever.json`
- Create: `configs/methods/dense_ft.json`
- Delete: `configs/training/dense_rgcn_graph_retriever/base.json`
- Delete: `configs/training/dense_rgcn_graph_retriever/ablations.json`
- Delete: `configs/training/dense_ft/base.json`
- Modify: `configs/experiments/*.json`

- [ ] 删除所有 config version 字段。
- [ ] 把 R-GCN config 改成 `encoder/pairs/train/profiles`。
- [ ] 把 Dense-FT config 改成同一顶层语义。
- [ ] 将 experiment config 的 `training_configs` 改为 `method_configs`。
- [ ] 删除 `configs/training/`。
- [ ] 运行 repository config tests。

### Task 4: 删除 training config compatibility facade

**Files:**

- Delete: `graph_memory/config/training_compat.py`
- Delete: `graph_memory/training_config.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: callers and tests

- [ ] 把 workflow method config 加载改成 `CONFIG_LOADER.load(Registry.configs.TRAINABLE_METHOD, argv)`。
- [ ] 删除 `load_trainable_training_config()`。
- [ ] 删除 `device_from_training_config()`；device 从 typed method/stage config 读取。
- [ ] 删除 compatibility helper architecture tests，替换成 module absence 测试。
- [ ] 运行 import scan，确保没有旧 module import。

### Task 5: 建立当前 method registry

**Files:**

- Create: `graph_memory/registry/methods.py`
- Modify: `graph_memory/registry/app.py`
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Delete: `graph_memory/registry/projections.py`
- Delete: `graph_memory/retrieval_registry.py`
- Delete: `graph_memory/retrieval/catalog.py`
- Create: `tests/test_method_registry.py`

- [ ] 先写 method definition 和 artifact/dependency source 测试。
- [ ] 实现 `MethodDefinition`。
- [ ] 迁移所有 method enumeration 调用。
- [ ] 删除 projection/facade。
- [ ] 删除 `builder_id` 和 capability bool。
- [ ] 更新 tune、ranking validation、workflow、experiment method listing。
- [ ] 运行：

```powershell
uv run pytest tests/test_method_registry.py tests/test_phase1_real_retrieval.py tests/test_phase2_rgcn_retrieval.py tests/test_dense_ft_retrieval_registry.py -q
```

### Task 6: 重建 artifact semantics

**Files:**

- Modify: `scripts/workflow/artifacts.py`
- Modify: `scripts/workflow/registry.py`
- Modify: `scripts/workflow/types.py`
- Modify: `scripts/workflow/status.py`
- Modify: `scripts/deliver/collect_run_artifacts.py`

- [ ] artifact basename 从 `MethodDefinition.train_artifact` 获取。
- [ ] file/directory 类型进入 status validation。
- [ ] 将 `EFFECTIVE_TRAINING_CONFIG` 改为 `EFFECTIVE_METHOD_CONFIG`。
- [ ] 更新 main 与 variant artifact paths。
- [ ] 删除 Dense-FT 特判函数。

### Task 7: 编译 current stage config

**Files:**

- Modify: `scripts/workflow/stage_configs.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: low-level script parsers

- [ ] workflow 直接构造 typed stage configs，不再 synthetic argv round-trip。
- [ ] 写入 stage config JSON。
- [ ] low-level `--config` 只接受完整 stage config。
- [ ] 删除 `normalize_raw_config`。
- [ ] pair/train config 不再读取 method config。
- [ ] variant 生成自己的 stage config。

### Task 8: 删除 legacy workflow command assembly

**Files:**

- Modify: `scripts/workflow/workflows.py`
- Modify: `scripts/workflow/planner.py`
- Modify: `scripts/workflow/types.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_workflow_orchestration.py`

- [ ] command builder 只生成 `script --config <stage-config-path>`。
- [ ] 删除 `_stage_config_projection(...)->None` 语义。
- [ ] 删除所有 `else` legacy argv assembly。
- [ ] 删除缺少 `stage_configs` 仍可 plan 的测试。
- [ ] 新增缺少 stage config 立即失败测试。

### Task 9: strict current manifest

**Files:**

- Create: `scripts/workflow/contracts.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/planner.py`
- Modify: `scripts/workflow/status.py`
- Modify: `scripts/workflow/resume.py`
- Create: `tests/test_current_manifest_contract.py`

- [ ] 删除 manifest `schema_version`。
- [ ] 删除 v1 branch。
- [ ] 加入 strict structure。
- [ ] old manifest fixture 必须失败。
- [ ] `--force` 重建 current manifest 测试通过。

### Task 10: TRAIN union 穷尽化

**Files:**

- Modify: `scripts/train_method.py`
- Modify: `graph_memory/registry/training.py`
- Modify: `graph_memory/stages/train.py`
- Modify: `tests/test_dense_finetune_training.py`
- Modify: `tests/test_phase2_rgcn_training.py`

- [ ] 所有 config 分支显式覆盖 R-GCN 和 Dense-FT。
- [ ] 使用 `assert_never()`。
- [ ] result 类型显式检查。
- [ ] 第三种 fake config 测试必须失败，不能落入 Dense-FT。
- [ ] Ruff 两个 F401 归零。

### Task 11: 当前 checkpoint 与 metadata contract

**Files:**

- Modify: `graph_memory/models/graph_retriever/checkpoint.py`
- Modify: `graph_memory/models/graph_retriever/config/records.py`
- Modify: `graph_memory/models/graph_retriever/training.py`
- Modify: `graph_memory/validation/model.py`
- Create: `graph_memory/models/dense_finetune/metadata.py`
- Modify: `graph_memory/models/dense_finetune/training.py`
- Modify: `graph_memory/registry/retrieval_builders.py`

- [ ] 删除 R-GCN checkpoint version。
- [ ] 执行 R-GCN-specific rename。
- [ ] Dense-FT metadata 改为 dataclass。
- [ ] 删除 Dense-FT metadata version。
- [ ] writer/reader 共用 contract。
- [ ] 旧 artifact 因 unknown version field 失败。

### Task 12: retrieval provenance 与摘要修复

**Files:**

- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `graph_memory/stages/retrieve.py`
- Modify: `scripts/run_retrieval.py`
- Modify: `tests/test_config_run_retrieval.py`
- Modify: `tests/test_dense_ft_retrieval_registry.py`
- Modify: `tests/test_phase2_rgcn_retrieval.py`

- [ ] registry build 返回 method + provenance。
- [ ] Dense-FT summary 使用 model directory、实际 device 和 metadata encoder。
- [ ] R-GCN summary 使用 checkpoint record。
- [ ] BM25 summary 不再伪造 encoder CLI 默认值。
- [ ] 删除 script 中的 config 类型猜测 helper。

### Task 13: ablation current contract

**Files:**

- Modify: `graph_memory/registry/ablations.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/planner.py`
- Modify: ablation tests

- [ ] patch 路径改为 current method config 字段。
- [ ] 删除 ablation index version。
- [ ] variant stage configs 从 patched typed method config 编译。
- [ ] baseline alias 和 invalidation 语义保持。

### Task 14: 文档与架构防回归

**Files:**

- Modify/create: `docs/configs/methods/*.md`
- Modify: `docs/configs/README.md`
- Modify: contracts/design/operations docs
- Modify: architecture tests

- [ ] 删除所有旧 config path 文档。
- [ ] 删除 `schema v1/v2`、migration、compatibility projection 说明。
- [ ] 明确旧 run、旧 checkpoint、旧 model directory 不支持。
- [ ] 增加扫描测试，禁止重新引入：
  - `graph_memory.training_config`
  - `graph_memory.config.training_compat`
  - `graph_memory.retrieval_registry`
  - `graph_memory.retrieval.catalog`
  - `graph_memory.registry.projections`
  - `builder_id`
  - trainable `schema_version`
  - `checkpoint_version`
  - workflow legacy fallback

### Task 15: 全量验证

Windows 上所有 `uv` 命令从第一次尝试就在 Codex filesystem sandbox 外运行。

```powershell
uv run pytest -q
uv run ruff check .
uv run basedpyright graph_memory scripts tests --level error
openspec validate remove-trainable-stack-compatibility --strict
git diff --check
```

额外 smoke：

```powershell
uv run python scripts/experiment.py methods list
uv run python scripts/experiment.py plan rgcn_zero_compat_smoke --profile smoke --methods dense_rgcn_graph_retriever --force
uv run python scripts/experiment.py plan dense_ft_zero_compat_smoke --profile smoke --methods dense_ft --force
```

如本机数据与模型可用，再执行：

```powershell
uv run python scripts/experiment.py run rgcn_zero_compat_smoke --profile smoke --methods dense_rgcn_graph_retriever --force
uv run python scripts/experiment.py run dense_ft_zero_compat_smoke --profile smoke --methods dense_ft --force
```

## 20. 完成标准

只有同时满足以下条件，重构才算完成：

- [ ] `configs/training/` 不存在。
- [ ] 两个 method config 都位于 `configs/methods/`。
- [ ] trainable config、manifest、checkpoint、metadata、ablation index 无版本字段。
- [ ] `training_compat.py` 与 `training_config.py` 不存在。
- [ ] `projections.py`、`retrieval_registry.py`、`retrieval/catalog.py` 不存在。
- [ ] `builder_id` 不存在。
- [ ] trainable runtime 不使用 capability bool 组合推断来源。
- [ ] workflow 不存在 legacy manifest fallback。
- [ ] workflow command 不存在 method-specific argv assembly。
- [ ] `stage_configs` 是必填 current contract。
- [ ] TRAIN union 所有分支穷尽。
- [ ] Dense-FT retrieval summary 记录真实 checkpoint/device/encoder。
- [ ] R-GCN checkpoint 和 Dense-FT metadata 都使用当前严格 contract。
- [ ] 旧 config、旧 manifest、旧 checkpoint、旧 model directory 均明确失败。
- [ ] 全量 pytest、Ruff、basedpyright、OpenSpec strict 和 diff check 通过。

## 21. 审核时需要重点确认的决策

1. 是否同意删除整个 `configs/training/`，只保留 `configs/methods/`。
2. 是否同意 experiment config 将 `training_configs` 直接改名为 `method_configs`，不保留 alias。
3. 是否同意 low-level script 的 `--config` 统一表示完整 stage config，而 method config 只由 workflow compiler 读取。
4. 是否同意删除 `retrieval_registry.py`、`retrieval/catalog.py` 和 `registry/projections.py`，所有调用方直接迁移到 current registry。
5. 是否同意 R-GCN checkpoint 删除版本字段并执行 R-GCN-specific 类型重命名。
6. 是否同意 old run 必须删除或 `--force` 重建，不提供 manifest 恢复读取。
7. 是否同意本次使用一个 OpenSpec change 完成，避免在多个 change 之间创建临时 compatibility adapter。

