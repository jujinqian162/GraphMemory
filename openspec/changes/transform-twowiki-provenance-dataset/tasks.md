## 1. 转换配置模型

- [x] 1.1 在 `graph_memory/experiment/config.py` 新增 `TwoWikiProvenanceTransformConfig`（pydantic `ClosedModel`），承载全部转换参数（edge_scorer、seed、candidate_cap、dev_fraction、successors_per_output、hybrid_dense_weight、semantic_temperature、weight_floor、scorer_identity、query_template_version、branch_policy_version、near/mid/tail rank bucket、dense_model、dense_query_prefix、dense_passage_prefix、dense_batch_size）
- [x] 1.2 为 `TwoWikiProvenanceTransformConfig` 实现 `identity() -> dict[str, JsonValue]`，产生可 canonical JSON 序列化的稳定参数字典（bm25 时 `dense` 块为 None，dense/hybrid 时含 dense 身份）
- [x] 1.3 在 `DatasetConfig` 增加可选字段 `transform: TwoWikiProvenanceTransformConfig | None = None`，并在 `ResolvedDatasetConfig` 中透传
- [x] 1.4 在 `resolve_experiment_config` 增加校验：`twowiki_provenance` 必须提供 `transform` 块；其它数据集不得提供，违反时抛清晰错误
- [x] 1.5 更新 `config.py` 的 `__all__` 导出新类型（`RankBucketConfig`、`TwoWikiProvenanceTransformConfig`）

## 2. 转换 stage 逻辑

- [x] 2.1 新建 `graph_memory/stages/transform.py`，定义 `TwoWikiProvenanceTransformResult`（含 train/dev/test 的 `FileSourceRef`）
- [x] 2.2 实现 `version_tag` 计算：`v{schema_version}-{sha256(canonical_json({schema_version, transform: config.identity(), encoder_digest}))[:16]}`（encoder 内容摘要并入 tag，模型权重变更即使不改名也强制 re-transform）
- [x] 2.3 实现 `materialize_transform_twowiki(...)`：读源 train/dev，由 transform config 构造 `ProvenanceGraphConstructionConfig` 与可选 dense ranker，调用 `convert_twowiki_source_records`（`strict=False`）与 `deterministic_dev_test_partition`，写 train/dev/test 到版本目录
- [x] 2.4 实现原子发布：临时目录 + `os.replace`；目标版本目录已存在且完整则跳过（幂等），返回 3 个 `FileSourceRef`
- [x] 2.5 确认不生成 manifest.json / statistics.json

## 3. Prefect task 与 flow 接线

- [x] 3.1 在 `graph_memory/experiment/tasks.py` 新增 `transform_twowiki_task`（`@task`, `cache_policy=SCIENTIFIC_CACHE_POLICY`, `persist_result=True`），输入 train_source、dev_source、config、encoder_source、schema_version，调用 `materialize_transform_twowiki`
- [x] 3.2 在 task 内当 edge_scorer 为 dense/hybrid 时从 `encoder_source` 提取 encoder_digest 并入 version_tag，bm25 时为 None；encoder_source 由 workflow 侧经 `resolve_encoder_source(dense_model)` 解析
- [x] 3.3 更新 `tasks.py` 的 `__all__` 导出 `transform_twowiki_task`
- [x] 3.4 在 `graph_memory/experiment/workflow.py` 增加分叉：`dataset.name == "twowiki_provenance"` 时先解析源 2Wiki train/dev 为 `FileSourceRef`，调 `transform_twowiki_task`，再把对应 split 引用喂给 `prepare_split_task`（`_resolve_split_sources` / `_transform_split_sources`）
- [x] 3.5 保持非 provenance 数据集走直读路径（`_split_source` 重命名为 `_direct_split_source` → `identify_external_source`，行为不变）

## 4. 配置与文档

- [x] 4.1 更新 `configs/dataset/twowiki_provenance.yaml`：`splits.train.source` 指向源 2Wiki train，`splits.dev.source` 与 `splits.test.source` 指向源 2Wiki dev；新增 `transform:` 块填入 formal v3 参数（hybrid scorer）
- [x] 4.2 （改为）capacity 设为可选：`all_available` 运行时直接取文件内全部可用记录，无需预跑一次拿 capacity 数值；现有数据集保留 capacity 语义（向后兼容）。见 3.5 用户指令
- [x] 4.3 更新 `docs/40-operations/twowiki-provenance.md` runbook：移除手动 convert 步骤，改述 flow 内自动转换与基于分支/config 的回滚
- [x] 4.4 删除 `scripts/data/convert_2wiki_to_execution_provenance.py`
- [x] 4.5 确认 `scripts/data/generate_twowiki_provenance_smoke.py` 保留且不受影响

## 5. 测试

- [x] 5.1 单元测试：`TwoWikiProvenanceTransformConfig.identity()` 稳定性（同参同摘要、改任一参数摘要变）与 version_tag 生成、非法参数拒绝（`tests/test_twowiki_provenance_transform.py`）
- [x] 5.2 单元测试：`materialize_transform_twowiki` 端到端跑通（合成 source，bm25 scorer），产出 3 个 split 与正确 `FileSourceRef`、字节确定性、幂等性、dev/test 切分确定性、无 manifest/statistics（`tests/test_twowiki_provenance_dataset.py::test_transform_is_byte_deterministic_and_raw_only`）
- [x] 5.3 单元测试：config 校验（provenance 缺 transform 块、其它数据集含 transform 块均被拒绝）
- [x] 5.4 集成测试：workflow 中 `twowiki_provenance` 正确串起 transform→prepare 数据依赖；其它数据集不走 transform（`tests/test_twowiki_provenance_workflow_fork.py`）
- [x] 5.5 运行 `uv run pytest` 全绿（165 passed），ruff + basedpyright（0 errors）通过
- [x] 5.6 smoke 验证：`experiment/run.py ... dataset=twowiki_provenance profile=smoke method=bm25` 全链路跑通（recall@10=1.0），版本目录正确生成，重跑命中 Prefect 缓存且幂等不新增目录
