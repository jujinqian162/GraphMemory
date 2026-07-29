## Context

`twowiki_provenance` 是一次性转换程序 `scripts/data/convert_2wiki_to_execution_provenance.py` 从本地 2Wiki 源生成的合成执行溯源基准。当前流程：手动运行 convert 脚本产出 `data/twowiki_provenance/raw/{train,dev,test}.json`（外加 manifest/statistics），config 的 `splits.*.source` 指向这些文件，flow 里 `prepare_split_task` 用 `identify_external_source` 的文件 digest 做缓存输入。

`data/` 不受 git 追踪，转换产物游离在 Prefect 缓存体系外。分支切换 schema 后回退旧分支时，磁盘上是新 schema 数据，与旧代码不匹配。runbook 描述的回滚是手动改 config，且 v2/v3 raw/prepared/pair/checkpoint 互不兼容、无翻译回退。

关键约束：Prefect 缓存策略为 `INPUTS + TASK_SOURCE`。`TASK_SOURCE` 只 hash task 函数体，不 hash 被调用的 `converter.py`/`scoring.py`。因此仅改 converter 逻辑而不改 task 函数体时缓存不会失效——必须把显式 schema 版本号与结构化转换参数作为 task 的 INPUTS，才能让缓存随 schema/参数变化，与现有 `implementation_version="prepare-v1"` 模式一致。

## Goals / Non-Goals

**Goals:**
- 把 2Wiki→provenance 转换变成 flow 内受 Prefect 缓存管理的 `transform_twowiki_task`，置于 `prepare_split_task` 之前。
- 分支切换 schema/参数 → 缓存键变 → 自动重转换；回滚 → 缓存键复原 → 命中旧版本目录缓存，不重跑。
- 转换产物按版本隔离存放，新旧 schema 互不覆盖。
- transform 与 prepare 之间建立显式 Prefect 数据依赖，保证顺序与缓存链、正确传播 `cache.refresh`。
- 移除独立 convert 脚本，逻辑并入 flow。

**Non-Goals:**
- 不做 v2/v3 数据的自动翻译/迁移（runbook 明确无翻译回退）。
- 不保留 manifest/statistics 副产物。
- 不改动 converter/scoring 的转换算法本身。
- 不为其它数据集（hotpotqa/twowiki/musique）引入 transform task。

## Decisions

### D1：单 transform task 产 3 split，写版本化 raw 目录

`transform_twowiki_task` 一次读源 train/dev，产出 train/dev/test 三个 split（dev 源经 `deterministic_dev_test_partition` 种子切分为 dev/test），写入 `data/twowiki_provenance/raw/<version_tag>/{train,dev,test}.json`。

- **为何单 task**：dev/test 来自同一 dev 源的种子切分，拆成独立 task 会破坏切分一致性。
- **为何仍写 raw 目录（而非 ProcessedAssetStore）**：改动最小；版本子目录已提供隔离与回滚一致性。
- 备选（拆分 split 级 task）已否决：破坏切分确定性。

### D2：结构化 `TwoWikiProvenanceTransformConfig` + 版本目录名含参数摘要

原脚本全部转换参数进 pydantic `ClosedModel`，挂在 `DatasetConfig.transform`（可选，仅 twowiki_provenance 用）。提供 `identity() -> dict[str, JsonValue]`。

`version_tag = f"v{schema_version}-{digest}"`，`digest` = 对 `{"schema_version": ..., "transform": config.identity()}` 做 canonical JSON（sort_keys、紧凑分隔符）后取 sha256 前 16 位。

- **为何摘要要含全部参数**：同 schema 不同参数（如换 edge_scorer）必须落到不同目录，否则互相覆盖或错误命中缓存。
- 备选（仅用 schema_version 命名）已否决：有正确性漏洞。

### D3：transform 返回 3 个 `FileSourceRef`，prepare 消费

`transform_twowiki_task` 返回 `TwoWikiProvenanceTransformResult{train,dev,test: FileSourceRef}`（对 3 个 split 文件调 `identify_external_source`）。workflow 里仅当 `dataset.name == "twowiki_provenance"` 时先调 transform，再把对应 split ref 喂给 `prepare_split_task(source=...)`；其它数据集维持现有 `_split_source` 直读。

- **为何**：Prefect 只有当 prepare 接收 transform 返回值作为输入时才建立数据依赖、保证顺序与缓存链。若 prepare 仅按 config 固定路径读，Prefect 不知依赖关系，可能乱序或读到旧版本文件。
- config 的 `splits.*.source` 因此改指向源 2Wiki 文件（train 指 train 源，dev/test 均指 dev 源）。

### D4：dense 模型内容寻址进 INPUTS

edge_scorer 为 dense/hybrid 时，用 `resolve_encoder_source(dense_model)` 得到内容寻址引用（`DirectorySourceRef`）作为 task INPUTS；bm25 时为 None。

- **为何**：模型内容影响转换输出，换权重而不改名时缓存必须失效。与 flow 其它 task 用 `resolve_encoder_source` 一致。
- 备选（模型仅当字符串参数）已否决：换权重不改名有正确性漏洞。

### D5：只产 raw，舍弃 manifest/statistics

transform 只写 3 个 split raw 文件。参数与计数隐含在版本目录名和 Prefect artifact origin 中；分布审计如需由独立工具按需重算。

### D6：编排逻辑搬入 `graph_memory/stages/transform.py`

新增 `materialize_transform_twowiki(...)` 承载原脚本编排（读源、`convert_twowiki_source_records`、`deterministic_dev_test_partition`、写文件），复用现有 `converter.py`。采用临时目录 + `os.replace` 原子发布；目标版本目录已存在且完整则跳过（幂等）。转换阶段固定 `strict=False`（沿用原默认，丢弃不可恢复记录）；prepare 阶段严格性仍由 `dataset.strict_invalid_examples` 独立控制。

## Risks / Trade-offs

- **版本目录无限增长** → 回滚需要旧目录仍在磁盘；接受磁盘占用，提供文档说明可手动清理不再需要的版本目录（清理后回滚会触发重转换而非直接命中）。
- **`identity()` 摘要不稳定导致缓存漂移** → 用 canonical JSON（sort_keys + 固定分隔符）序列化；加单元测试锁定同参数同摘要、改任一参数摘要变。
- **半写/并发产生损坏目录** → 临时目录 + `os.replace` 原子发布；发布前目标已存在视为完成并跳过。
- **config 分叉遗漏其它数据集** → workflow 仅在 `twowiki_provenance` 分支走 transform，其它维持原路径；加集成测试确认非 provenance 数据集不走 transform。
- **BREAKING：删除 convert 脚本** → runbook 同步更新，改述 flow 内自动转换与基于分支/config 的回滚；`generate_twowiki_provenance_smoke.py` 保留（调库函数，不依赖 CLI）。

## Migration Plan

1. 首次运行 flow 会在 `data/twowiki_provenance/raw/<version_tag>/` 生成 v3 数据；旧的平铺 `data/twowiki_provenance/raw/{train,dev,test}.json` 可保留或手动删除，不再被引用。
2. 回滚：切回旧分支即可，schema_version/transform config 恢复旧值 → version_tag 复原 → 命中旧目录缓存（若在）。无需手动改 config。
3. 无数据库/schema 迁移；纯文件布局与 config 变更。

## Open Questions

无。五项核心决策（D1–D5）已与用户对齐并确认。
