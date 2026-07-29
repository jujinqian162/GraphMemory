## ADDED Requirements

### Requirement: 转换步骤内置为 Prefect 缓存管理的 flow task

系统 SHALL 提供一个 flow 内的 `transform_twowiki_task`（Prefect `@task`，`cache_policy=INPUTS+TASK_SOURCE`，`persist_result=True`），把源 2Wiki train/dev 转换为 `twowiki_provenance` 的 train/dev/test raw 数据，并置于 `prepare_split_task` 之前。系统 SHALL NOT 依赖独立的命令行转换脚本来生成该数据集。

#### Scenario: flow 首次运行生成 provenance raw 数据
- **WHEN** 以 `dataset=twowiki_provenance` 运行 experiment flow 且对应版本目录不存在
- **THEN** `transform_twowiki_task` 读取源 2Wiki train/dev 文件，产出 train/dev/test 三个 split 的 raw 文件

#### Scenario: 转换算法复用现有 converter
- **WHEN** `transform_twowiki_task` 执行转换
- **THEN** 系统调用现有 `convert_twowiki_source_records` 与 `deterministic_dev_test_partition` 完成转换，不改动其算法

### Requirement: 结构化转换配置参与缓存键

系统 SHALL 提供结构化的 `TwoWikiProvenanceTransformConfig`（pydantic `ClosedModel`），承载转换参数（edge_scorer、seed、candidate_cap、dev_fraction、successors_per_output、hybrid_dense_weight、semantic_temperature、weight_floor、scorer_identity、query_template_version、branch_policy_version、三个 rank bucket、dense 模型相关字段）。该配置 SHALL 挂在 `DatasetConfig.transform`（可选，仅 `twowiki_provenance` 使用），并作为 `transform_twowiki_task` 的 INPUTS 参与 Prefect 缓存键。schema 版本号 SHALL 作为显式 task 输入，而非依赖 `TASK_SOURCE`。

#### Scenario: 转换参数变化触发重转换
- **WHEN** 修改 `transform` 配置中任一参数（如 edge_scorer 从 hybrid 改为 bm25）并重新运行 flow
- **THEN** 缓存键变化，`transform_twowiki_task` 重新执行并写入新的版本目录

#### Scenario: schema 版本变化触发重转换
- **WHEN** schema 版本号发生变化并重新运行 flow
- **THEN** 缓存键变化，`transform_twowiki_task` 重新执行

#### Scenario: 缺失或多余的 transform 配置被拒绝
- **WHEN** `twowiki_provenance` 数据集缺少 `transform` 配置块，或非 `twowiki_provenance` 数据集填写了 `transform` 块
- **THEN** 配置解析阶段抛出清晰的错误

### Requirement: 转换产物按版本隔离存放

系统 SHALL 把转换产物写入 `data/twowiki_provenance/raw/<version_tag>/{train,dev,test}.json`，其中 `version_tag = "v{schema_version}-{digest}"`，`digest` 为对 `{"schema_version": ..., "transform": config.identity()}` 做 canonical JSON（键排序、紧凑分隔符）后的 sha256 前 16 位。不同 schema 或参数产生的产物 SHALL NOT 互相覆盖。系统 SHALL NOT 生成 `manifest.json` 或 `statistics.json` 副产物。

#### Scenario: 版本目录名随 schema 与参数唯一确定
- **WHEN** 给定一组 schema 版本与转换参数
- **THEN** 版本目录名可由 `schema_version` 与 `config.identity()` 稳定复现，同参数得同目录名、改任一参数得不同目录名

#### Scenario: 幂等发布已存在的版本目录
- **WHEN** 目标版本目录已存在且内容完整
- **THEN** `transform_twowiki_task` 跳过重写，采用临时目录 + 原子替换避免半写损坏

#### Scenario: 回滚命中旧版本缓存
- **WHEN** 切回旧分支使 schema_version 与转换参数恢复旧值，且旧版本目录仍在磁盘
- **THEN** `transform_twowiki_task` 命中 Prefect 缓存并返回旧引用，无需手动修改配置或重跑转换

### Requirement: transform 与 prepare 建立显式数据依赖

`transform_twowiki_task` SHALL 返回包含 train/dev/test 三个 split 的 `FileSourceRef` 的结果对象，`prepare_split_task` SHALL 接收对应 split 的 `FileSourceRef` 作为 `source` 输入，从而建立 Prefect 数据依赖，保证「先 transform 后 prepare」的执行顺序与缓存链，并正确传播 `cache.refresh`。`configs/dataset/twowiki_provenance.yaml` 的 `splits.*.source` SHALL 指向源 2Wiki 文件（train 指向 train 源，dev 与 test 均指向 dev 源）。非 `twowiki_provenance` 数据集 SHALL 维持现有直读 source 路径。

#### Scenario: prepare 消费 transform 输出
- **WHEN** flow 处理 `twowiki_provenance` 数据集
- **THEN** `prepare_split_task` 使用 `transform_twowiki_task` 返回的对应 split 引用作为 source，且 Prefect 保证 transform 先于 prepare 执行

#### Scenario: 其它数据集不走 transform
- **WHEN** flow 处理 hotpotqa / twowiki / musique 数据集
- **THEN** 不调用 `transform_twowiki_task`，`prepare_split_task` 按现有 `identify_external_source` 直读配置路径

### Requirement: dense/hybrid scorer 的模型依赖内容寻址

当 `edge_scorer` 为 `dense` 或 `hybrid` 时，系统 SHALL 使用 `resolve_encoder_source` 把 dense 模型解析为内容寻址引用，并作为 `transform_twowiki_task` 的 INPUTS 参与缓存键；当 `edge_scorer` 为 `bm25` 时该输入为 None。模型内容变化 SHALL 触发重转换。

#### Scenario: 模型权重变化触发重转换
- **WHEN** dense/hybrid scorer 使用的模型权重内容发生变化（即便名称不变）
- **THEN** 内容寻址引用的 digest 变化，缓存键变化，`transform_twowiki_task` 重新执行

#### Scenario: bm25 scorer 无模型依赖
- **WHEN** `edge_scorer` 为 `bm25`
- **THEN** `transform_twowiki_task` 的 encoder_source 输入为 None，转换不依赖任何 dense 模型

## REMOVED Requirements

### Requirement: 命令行转换脚本
**Reason**: 转换编排逻辑并入 flow 内的 `transform_twowiki_task`，由 Prefect 缓存管理，消除游离在缓存体系外的手动步骤。
**Migration**: 不再运行 `scripts/data/convert_2wiki_to_execution_provenance.py`；改为直接运行 experiment flow（`dataset=twowiki_provenance`），转换自动执行并版本化缓存。回滚通过切换代码分支/配置完成，不再手动修改 config 指向旧目录。
