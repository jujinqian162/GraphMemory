# Memory Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可复现的 `memory_stream` Phase 2 baseline：先用本地 Qwen 一次性生成全局共享、query-independent 的 importance，后续检索按 normalized relevance + pseudo-recency + importance 排序。

**Architecture:** importance preparation 是与 workflow 同级的一次性全局数据预处理，不是 workflow stage。`scripts/annotate_importance.py` 默认读取 canonical HotpotQA dev input，写入 `data/hotpotqa/processed/memory_stream/`；任何 workflow workspace 后续只读该 artifact，并按 task id/content digest/node coverage 选择自身子集。检索阶段绝不加载 causal LLM。

**Tech Stack:** Python dataclass/TypedDict/Protocol、argparse、SHA-256、PyTorch、Transformers、现有 dense retrieval、pytest、Ruff、BasedPyright、OpenSpec。

---

## 1. 固定范围

必须实现：

1. public method id：`memory_stream`，在 retrieval milestone 中注册。
2. 公式：`relevance + pseudo-recency + importance`，默认权重均为 `1.0`。
3. relevance：复用现有 DenseTaskRetriever/DenseEncodingService。
4. pseudo-recency：由 `MemoryItem.position` 推导，不声称是真实时间。
5. importance：Qwen2.5-7B-Instruct 输出 1-10 整数，query-independent。
6. annotation 是全局一次性预处理；retrieval latency 不包含 LLM。
7. 一个 annotation 进程只加载一个模型实例，逐任务原子 cache，失败可续跑。
8. `MemoryItem`、`MemoryTaskInput`、`RankedResult` schema 不改变。
9. prompt/cache key 不读取 query、answer、labels、gold nodes 或 graph。
10. 全局 artifact 可覆盖完整 canonical dev corpus；workflow profile 可安全消费其子集。

明确不实现：

- 完整 Generative Agents simulator、reflection、planning、动态访问历史。
- importance 训练、蒸馏、权重调参或 ablation suite。
- train split importance。
- HTTP/cloud API、vLLM server、OpenAI-compatible server。
- Tensor Parallel、多进程生成或首版多卡切分。
- annotation workflow stage、run-local importance artifact、annotation stage config。
- LLM 失败后的静默默认分。

## 2. Ownership 与数据流

```text
data/hotpotqa/processed/dev_memory_tasks.input.json
  -> python scripts/annotate_importance.py
  -> graph_memory/retrieval/methods/memory_stream/{prompt,cache,runtime,annotation}
  -> data/cache/memory_stream_importance/<prefix>/<digest>.json
  -> data/hotpotqa/processed/memory_stream/dev.importance.json
  -> data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json

runs/<name>/inputs/test.input.json
  + global dev.importance.json
  -> select records by task_id
  -> validate content_digest and exact node ids
  -> later MemoryStream retrieval
```

默认命令：

```powershell
python scripts/annotate_importance.py
```

默认值：

| 参数 | 默认值 |
|---|---|
| `--tasks` | `data/hotpotqa/processed/dev_memory_tasks.input.json` |
| `--output` | `data/hotpotqa/processed/memory_stream/dev.importance.json` |
| `--summary` | output 同目录的 `dev.importance.run_summary.json` |
| `--cache-dir` | `data/cache/memory_stream_importance` |
| `--model-id` | `Qwen/Qwen2.5-7B-Instruct` |
| `--model-path` | `models/Qwen2.5-7B-Instruct` |
| `--prompt-version` | `memory-stream-importance-v2` |
| `--device` | `auto` |
| `--max-new-tokens` | `2048` |

`MEMORY_STREAM_MODEL_PATH` 可覆盖默认模型目录，CLI `--model-path` 优先。

## 3. 文件职责

| 文件 | 单一职责 |
|---|---|
| `contracts.py` | importance artifact/cache/result JSON-shaped 类型 |
| `settings.py` | annotation semantic/runtime settings |
| `prompt.py` | prompt、response parser、content/cache digest |
| `cache.py` | content-addressed cache 读取、验证、原子写入 |
| `runtime.py` | 本地 Transformers 加载与 deterministic generate |
| `annotation.py` | cache hit/miss、单 runtime 生命周期、artifact assembly |
| `validation/importance.py` | producer exact validation 与 consumer subset selection |
| `scripts/annotate_importance.py` | 默认 CLI、IO、环境清理、run summary |
| `tests/test_memory_stream_importance.py` | prompt、parser、digest、cache、subset、runtime 生命周期 |
| `tests/test_memory_stream_importance_prepare.py` | zero-argument CLI 和 ownership smoke |

workflow 代码不得导入或调用 `scripts/annotate_importance.py`，不得声明
`StageId.IMPORTANCE`，不得分配 `runs/<name>/importance/`。

## 4. Artifact contract

```json
{
  "method": "memory_stream",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt_version": "memory-stream-importance-v2",
  "generation": {
    "do_sample": false,
    "use_cache": true,
    "max_new_tokens": 2048
  },
  "tasks": [
    {
      "task_id": "hotpot_x",
      "content_digest": "<sha256>",
      "scores": {
        "m0": 7,
        "m1": 4
      }
    }
  ]
}
```

Producer 对 canonical input 要求 task 数量、顺序、digest 和 node coverage
完全一致。Consumer 允许 artifact 是 workflow task 的超集，但必须：

- artifact task id 唯一；
- 请求 task 必须存在；
- 选中记录的 content digest 必须匹配；
- node ids 必须精确覆盖；
- 返回顺序跟随 workflow task 顺序。

## 5. Prompt、cache 与模型生命周期

Prompt 只包含：

```json
{"node_id": "m0", "source": "title", "text": "sentence", "position": 0}
```

Cache key 包含：

- `model_id`
- prompt version
- semantic generation settings
- ordered item id/source/text/position

Cache key 不包含 query、labels、`model_path`、GPU 编号、运行目录。

进程流程：

```python
scan_all_cache_entries()
if misses:
    runtime = runtime_factory(settings)
    runtime.load()
    for task in misses:
        generated = runtime.generate(build_importance_messages(task), settings)
        validated = parse_and_validate(generated, task)
        cache.write(validated)
artifact = assemble_in_original_task_order()
write_final_artifact_atomically()
```

脚本在模型相关 import 前清理 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、
`MASTER_ADDR`、`MASTER_PORT`，并设置
`ACCELERATE_USE_DEEPSPEED=false`。`CUDA_VISIBLE_DEVICES` 由 shell 控制。

## 6. 三信号定义

```python
relevance_raw = dense_score_by_node_id[item["id"]]
age_steps = max_position - item["position"]
recency_raw = recency_decay ** age_steps
importance_raw = float(task_importance["scores"][item["id"]])
```

每个 signal 在 task 内独立 min-max normalization。常量 signal 全部映射为
`0.0`。最终分数：

```python
score = (
    relevance_weight * relevance[node_id]
    + recency_weight * recency[node_id]
    + importance_weight * importance[node_id]
)
```

按 `(-score, node_id)` 排序。权重非负且至少一个大于零；
`0 < recency_decay <= 1`。

## 7. 实施任务

### Task 1: Global importance prepare

- [x] importance contracts、strict validator、prompt 和 parser。
- [x] content-addressed cache 与 atomic JSON。
- [x] 单进程单模型 direct Transformers runtime。
- [x] zero-argument standalone CLI 和完整默认值。
- [x] success/failure run summary。
- [x] global artifact subset selection。
- [x] 删除 stage-config/workflow ownership。

### Task 2: MemoryStreamMethod

- [ ] 实现 normalization。
- [ ] 实现 relevance/pseudo-recency/importance 加权。
- [ ] 实现 deterministic ties 和标准 RankedResult。
- [ ] 测试 raw relevance 与现有 dense path 一致。

### Task 3: Retrieval registry

- [ ] 注册 `memory_stream` method/settings/build payload。
- [ ] RetrieveIO 增加只读 global importance path。
- [ ] builder 加载 artifact 并选择 workflow task 子集。
- [ ] provenance 记录 artifact path/model/prompt/digests/weights/encoder。
- [ ] 证明 retrieval 不 import/load Qwen runtime。

### Task 4: Workflow external dependency

- [ ] workflow 为 `prepare -> graphs -> retrieve -> evaluate -> aggregate`。
- [ ] 无 importance stage、无 annotation command。
- [ ] manifest 无 run-local importance artifact/config。
- [ ] retrieval 默认读取 global importance path。
- [ ] missing/stale selected records 在 retrieval 前失败。
- [ ] delivery 只记录 external provenance，不复制 global cache。

### Task 5: Docs 与最终验收

- [ ] 更新 data/retrieval/architecture/config/operations 文档。
- [ ] 记录 ModelScope/Hugging Face 下载和 MetaX preflight。
- [ ] 完整 pytest、Ruff、BasedPyright、OpenSpec strict、diff check。
- [ ] MetaX 零参数实跑：首次 `model_load_count=1`，重跑
  `model_load_count=0`。

## 8. 当前 prepare 验收

```powershell
uv run pytest tests/test_memory_stream_importance.py tests/test_memory_stream_importance_prepare.py -q
uv run pytest -q
uv run ruff check .
uv run basedpyright graph_memory scripts tests --level error
uv run python scripts/annotate_importance.py --help
openspec validate add-memory-stream-retrieval --strict
git diff --check
```

MetaX：

```bash
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
export ACCELERATE_USE_DEEPSPEED=false
export CUDA_VISIBLE_DEVICES=0
python scripts/annotate_importance.py
```

成功后保留：

```text
data/hotpotqa/processed/memory_stream/dev.importance.json
data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json
data/cache/memory_stream_importance/
```
