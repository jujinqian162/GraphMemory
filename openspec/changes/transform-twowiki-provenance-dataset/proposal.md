## Why

`twowiki_provenance` 的 raw 数据是由一次性脚本 `scripts/data/convert_2wiki_to_execution_provenance.py` 从本地 2Wiki 源生成的合成执行溯源基准，其内容随转换程序的 schema、参数与 scorer 模型变化。因为 `data/` 不受 git 追踪、转换产物也游离在 Prefect 缓存体系之外，在某分支更新 schema 后再回退到旧分支会导致磁盘上的 raw 数据与旧代码期望的 schema 不匹配。当前 runbook 描述的回滚是手动改 config 指回旧目录，易错且无翻译回退。

## What Changes

- 新增 flow 内的 `transform_twowiki_task`（Prefect `@task`，`cache_policy=INPUTS+TASK_SOURCE`），置于 `prepare_split_task` 之前，把源 2Wiki train/dev 转换为 `twowiki_provenance` 的 train/dev/test raw 数据。
- 新增结构化 `TwoWikiProvenanceTransformConfig`（Hydra/pydantic），承载原脚本全部转换参数，并作为 task 的 INPUTS 参与缓存键。
- 转换产物按版本隔离：输出到 `data/twowiki_provenance/raw/<version_tag>/`，`version_tag = v{schema_version}-{config摘要}`，使新旧 schema 互不覆盖、回滚可命中旧缓存。
- `transform_twowiki_task` 返回 3 个 split 的 `FileSourceRef`，由 `prepare_split_task` 消费，建立显式 Prefect 数据依赖，保证「先 transform 后 prepare」的顺序与缓存链。
- dense/hybrid scorer 依赖的 dense 模型用 `resolve_encoder_source` 做内容寻址进入 INPUTS，模型内容变化触发重转换。
- `configs/dataset/twowiki_provenance.yaml` 的 `splits.*.source` 改指向源 2Wiki 文件，并新增 `transform:` 配置块。
- **BREAKING** 删除 `scripts/data/convert_2wiki_to_execution_provenance.py`（编排逻辑并入 flow）；不再生成 `manifest.json` / `statistics.json` 副产物。

## Capabilities

### New Capabilities
- `twowiki-provenance-transform`: flow 内受 Prefect 缓存管理的 2Wiki→provenance 转换步骤，含结构化转换参数配置、版本化输出目录、内容寻址的模型依赖，以及与 prepare 阶段的数据依赖接线。

### Modified Capabilities
<!-- openspec/specs/ 当前为空，无已存在能力的需求变更 -->

## Impact

- 新增：`graph_memory/stages/transform.py`、`TwoWikiProvenanceTransformConfig`（`graph_memory/experiment/config.py`）、`transform_twowiki_task`（`graph_memory/experiment/tasks.py`）。
- 修改：`graph_memory/experiment/workflow.py`（按 dataset 分叉 source 解析）、`graph_memory/experiment/config.py`（`DatasetConfig`/`ResolvedDatasetConfig` 增 `transform` 字段）、`configs/dataset/twowiki_provenance.yaml`、`docs/40-operations/twowiki-provenance.md`。
- 删除：`scripts/data/convert_2wiki_to_execution_provenance.py`。
- 复用不变：`graph_memory/datasets/twowiki_provenance/converter.py`、`scoring.py`。
- 数据布局：`data/twowiki_provenance/raw/` 由平铺文件改为版本子目录。
