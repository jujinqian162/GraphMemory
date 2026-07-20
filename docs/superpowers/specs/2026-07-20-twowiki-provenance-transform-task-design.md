# twowiki_provenance transform task 内置化设计

## 背景与问题

`data/` 不在 git 追踪范围内。`twowiki_provenance` 与 HotpotQA 等固定数据集不同：它的 raw 数据是由一次性转换程序 `scripts/data/convert_2wiki_to_execution_provenance.py` 从本地 2Wiki 源生成的合成执行溯源基准，其内容随转换程序版本（schema、参数、scorer 模型）变化。

当前流程：

1. 手动运行 convert 脚本，把 `data/2wiki/raw/{train,dev}.json` 转成 `data/twowiki_provenance/raw/{train,dev,test}.json`（外加 `manifest.json`、`statistics.json`）。
2. `configs/dataset/twowiki_provenance.yaml` 的 `splits.*.source` 指向这些生成好的文件。
3. flow 里 `prepare_split_task` 通过 `_split_source` 读该固定路径，用 `identify_external_source` 的文件 digest 做缓存输入。

问题：转换产物游离在 Prefect 缓存体系之外。若在某分支更新了 schema（如 v3→v4），再回退到旧分支，`data/twowiki_provenance/raw/` 里仍是新 schema 的文件，与旧代码期望的 schema 不匹配。当前 runbook 描述的"回滚"是**手动**把 config 指回旧 raw 目录并恢复 method config——易错，且 v2/v3 raw/prepared/pair/checkpoint 全部互不兼容、无翻译回退。

## 目标

把转换步骤变成 flow 内一个受 Prefect 缓存管理的 `transform_twowiki_task`，置于 `prepare_split_task` 之前。使得：

- 分支切换 schema/转换参数 → 缓存键变化 → 自动重新转换。
- 回滚代码 → 缓存键复原 → 命中旧版本目录的缓存，不重跑之前的转换。
- 转换产物按版本隔离存放，新旧 schema 互不覆盖。

移除独立的 convert 脚本，逻辑并入 flow task。

## 关键机制：为什么版本必须作为 task 输入

Prefect 缓存策略是 `INPUTS + TASK_SOURCE`。`TASK_SOURCE` 只 hash task **函数体本身**，不会 hash 它调用的 `converter.py` / `scoring.py` 等模块。因此如果只改 converter 逻辑而不改 task 函数体，缓存不会自动失效。

为让"分支切换自动切数据集缓存"可靠，必须把一个**显式 schema 版本号**和**结构化转换参数**作为 task 的 INPUTS 参数。参数变→INPUTS 变→缓存键变→重跑；回滚参数→缓存键回到旧值→命中旧缓存。这与 flow 中现有的 `implementation_version="prepare-v1"` 模式一致。

## 架构与组件

### 1. `TwoWikiProvenanceTransformConfig`（`graph_memory/experiment/config.py`）

新增 pydantic `ClosedModel`，承载原 convert 脚本的全部转换参数：

- `edge_scorer: Literal["bm25", "dense", "hybrid"]`
- `seed: ScientificInt`
- `candidate_cap: PositiveInt`
- `dev_fraction: ScientificFloat`（0 < x < 1）
- `successors_per_output: PositiveInt`
- `hybrid_dense_weight: ScientificFloat`
- `semantic_temperature: PositiveFloat`
- `weight_floor: NonNegativeFloat`
- `scorer_identity: str`
- `query_template_version: str`
- `branch_policy_version: str`
- `near_rank_bucket / mid_rank_bucket / tail_rank_bucket`：以可 JSON 序列化的形式表示（如 `[2, 4]`、`[9, null]`）
- dense 相关：`dense_model: str`、`dense_query_prefix: str`、`dense_passage_prefix: str`、`dense_batch_size: PositiveInt`

提供 `identity() -> dict[str, JsonValue]` 方法，返回全部参数的稳定字典（复用/对齐 `ProvenanceGraphConstructionConfig.identity()` 的结构）。

`TwoWikiProvenanceTransformConfig` 挂在 dataset config 下：`DatasetConfig` 新增可选字段 `transform: TwoWikiProvenanceTransformConfig | None = None`，仅 `twowiki_provenance` 使用；解析进 `ResolvedDatasetConfig`。

### 2. `materialize_transform_twowiki(...)`（新文件 `graph_memory/stages/transform.py`）

把原 `convert_2wiki_to_execution_provenance.py` 的**编排逻辑**搬进来（不含 argparse/CLI）：

- 读源 train/dev（`read_json`）。
- 用 `ProvenanceGraphConstructionConfig`（由 transform config 构造）+ 可选 dense ranker 调 `convert_twowiki_source_records`。
- 用 `deterministic_dev_test_partition` 把 dev 源切成 dev/test。
- 写 3 个 split 文件到版本目录。

纯转换逻辑仍复用现有 `graph_memory/datasets/twowiki_provenance/converter.py`，不改动。

**输出目录**：`data/twowiki_provenance/raw/<version_tag>/{train,dev,test}.json`
其中 `version_tag = f"v{schema_version}-{digest}"`，`digest` = 对 `{"schema_version": ..., "transform": config.identity()}` 做 canonical JSON（`sort_keys`, 紧凑分隔符）后取 sha256 前 16 位。

**不再产 manifest.json / statistics.json**（依决策）。参数与计数隐含在版本目录名和 Prefect artifact origin 中；如需分布审计，由独立分析工具按需重算。

返回值：3 个 `FileSourceRef`（对 3 个 split 文件调 `identify_external_source`），封装为一个小的 frozen dataclass / pydantic 模型（如 `TwoWikiProvenanceTransformResult`，字段 `train/dev/test: FileSourceRef`）。

写文件采用与 `ArtifactPublisher` 一致的原子发布思路：先写临时目录再 `os.replace` 到版本目录；若版本目录已存在且内容完整，跳过重写（幂等）。

### 3. `transform_twowiki_task`（`graph_memory/experiment/tasks.py`）

```python
@task(name="transform-twowiki", persist_result=True, cache_policy=SCIENTIFIC_CACHE_POLICY)
def transform_twowiki_task(
    train_source: FileSourceRef,
    dev_source: FileSourceRef,
    config: TwoWikiProvenanceTransformConfig,
    encoder_source: DirectorySourceRef | FileSourceRef | RevisionSourceRef | None,
    schema_version: int,
    implementation_version: str = "transform-twowiki-v1",
) -> TwoWikiProvenanceTransformResult:
    ...
```

- `train_source` / `dev_source`：源 2Wiki 文件的内容寻址引用（digest 进缓存键）。
- `config`：结构化转换参数（进缓存键）。
- `encoder_source`：当 `edge_scorer ∈ {dense, hybrid}` 时，用 `resolve_encoder_source(dense_model)` 得到内容寻址引用（模型内容变→缓存键变）；bm25 时为 `None`。
- `schema_version`：传入 `TWOWIKI_PROVENANCE_SCHEMA_VERSION`（进缓存键，也进版本目录名）。

### 4. `workflow.py` 按 dataset 分叉 source 解析

新增辅助函数，仅当 `config.dataset.name == "twowiki_provenance"` 时：

1. 解析源 2Wiki train/dev 文件为 `FileSourceRef`。
2. 调 `transform_twowiki_task` 得到 `TwoWikiProvenanceTransformResult`。
3. 把结果里对应 split 的 `FileSourceRef` 作为 `prepare_split_task(source=...)` 的输入。

其它数据集维持现有 `_split_source` → `identify_external_source` 直读路径。

因为 `prepare_split_task` 接收的是 `transform_twowiki_task` 返回值中的引用，Prefect 会自动建立数据依赖，保证「先 transform 后 prepare」的顺序与缓存链，`cache.refresh` 语义也随之正确传播。

### 5. `configs/dataset/twowiki_provenance.yaml`

- `splits.train.source` → 源 2Wiki train 文件（如 `data/2wiki/raw/train.json`）。
- `splits.dev.source` 与 `splits.test.source` → 均指向源 2Wiki dev 文件（因为 dev/test 都由 dev 源经种子切分而来）。
- 新增 `transform:` 块，填入 formal v3 的固定参数（hybrid scorer、seed 13、candidate_cap 32、dev_fraction 0.5、successors 2、各 rank bucket、dense_model 等），对齐 runbook 中记录的正式参数。
- `capacity` 由实际生成计数决定（保持现值或按需更新）。

### 6. 删除与保留

- **删除** `scripts/data/convert_2wiki_to_execution_provenance.py`（CLI 编排逻辑并入 flow）。
- **保留** `scripts/data/generate_twowiki_provenance_smoke.py`：它直接调用库函数 `convert_twowiki_source_records` 生成测试 fixture，不依赖被删的 CLI，无需改动。

## 数据流

```
源 2Wiki train.json ─┐
源 2Wiki dev.json  ──┤
transform config ────┼─→ transform_twowiki_task ─→ TransformResult{train,dev,test: FileSourceRef}
encoder_source ──────┤                                    │
schema_version ──────┘                                    │
                                                          ├─ train ref ─→ prepare_split_task(train) ─→ ...
                                                          ├─ dev ref   ─→ prepare_split_task(dev)   ─→ ...
                                                          └─ test ref  ─→ prepare_split_task(test)  ─→ ...
```

版本目录：`data/twowiki_provenance/raw/v3-<digest>/{train,dev,test}.json`

回滚场景：切回旧分支 → `schema_version` 和/或 `transform config` 恢复旧值 → `version_tag` 回到旧值 → 若旧版本目录仍在磁盘，transform task 命中 Prefect 缓存直接返回旧引用，prepare 也命中旧缓存。无需手动改 config。

## 错误处理

- **源文件缺失**：`identify_external_source` 抛 `FileNotFoundError`，提示用户先运行 `prepare_dataset.py` 下载/放置源 2Wiki 文件。
- **dense/hybrid 但模型不可解析**：`resolve_encoder_source` 抛 `ValueError`（现有行为），信息指向模型路径。
- **转换阶段的丢弃语义**：`convert_twowiki_source_records(strict=...)` 是转换阶段（源 2Wiki → provenance raw）的开关，与 prepare 阶段的 `dataset.strict_invalid_examples`（provenance raw → prepared split）是两个不同阶段。转换阶段固定 `strict=False`（沿用原脚本默认行为：无法恢复有序 gold chain 的记录按理由计数丢弃），使正式数据集完整生成；prepare 阶段的严格性仍由 `dataset.strict_invalid_examples` 独立控制。
- **版本目录并发/半写**：采用临时目录 + `os.replace` 原子发布；发布前若目标已存在视为已完成并跳过。
- **config 非 twowiki_provenance 却填了 transform 块**，或 twowiki_provenance 缺 transform 块：在 `resolve_experiment_config` 阶段校验并抛出清晰错误。

## 测试

- **`TwoWikiProvenanceTransformConfig` 单元测试**：`identity()` 稳定性（同参数同摘要、改任一参数摘要变）、version_tag 生成、非法参数拒绝。
- **`materialize_transform_twowiki` 单元测试**：用 `tests/fixtures/twowiki_provenance_smoke_source.json` 作源，bm25 scorer（无模型依赖）跑通，产出 3 个 split 文件与正确的 `FileSourceRef`；幂等性（重复调用命中已存在版本目录不重写）；dev/test 切分确定性。
- **缓存键回归测试**：改 schema_version 或 transform config 任一参数 → version_tag 与预期不同；回滚 → version_tag 复原。
- **workflow 集成测试**：`twowiki_provenance` 分支正确串起 transform→prepare 数据依赖；其它数据集不走 transform。
- 更新 `docs/40-operations/twowiki-provenance.md` runbook：移除手动 convert 步骤，改述 flow 内自动转换与基于分支/config 的回滚。

## 验证方式

- `uv run pytest` 相关测试全绿。
- smoke 运行：`experiment/run.py ... dataset=twowiki_provenance profile=smoke method=bm25`，确认 transform→prepare→rank→evaluate 全链路跑通，版本目录正确生成。
- 类型检查通过。

## 范围外（YAGNI）

- 不做 v2/v3 数据的自动翻译/迁移（runbook 明确无翻译回退）。
- 不保留 manifest/statistics 副产物。
- 不改动 converter/scoring 的转换算法本身。
- 不为其它数据集（hotpotqa/twowiki/musique）引入 transform task。
