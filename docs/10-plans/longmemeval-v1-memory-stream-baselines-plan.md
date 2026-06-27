# LongMemEval V1 Memory Stream and Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 LongMemEval V1 接入当前 request-first 检索体系，让 `memory_stream` 只面向 LongMemEval V1，并让 BM25、Dense、Graph Rerank、Dense-FT 等 baseline 按能力分层适配该数据集。

**Architecture:** 数据集只负责 raw parsing、record conversion 和 consumer request projection；retriever 继续只消费 `TextRankingRequest`、`TemporalMemoryRankingRequest` 或 `GraphRankingRequest`。Memory Stream 的 LongMemEval 版本使用官方 haystack 顺序/时间信息导出的 recency，不再依赖 HotpotQA/2Wiki 的伪位置或 HotpotQA 专用 importance sidecar。

**Tech Stack:** Python dataclasses / TypedDict, current `graph_memory.datasets.*` pattern, `GraphBuildRequest`, `TextRankingRequest`, `TemporalMemoryRankingRequest`, workflow manifest/stage config compiler, pytest, basedpyright, ruff, `uv`.

---

Date: 2026-06-26

Status: Implemented through LongMemEval first-stage dataset/workflow, request-owned Memory Stream, active configs, and LongMemEval turn/session metrics. Real end-to-end smoke run is pending local raw file availability.

## 1. 背景和决策

当前仓库已经有 request-first 检索边界：

```text
dataset-specific record
  -> dataset-specific projector
  -> consumer-owned request
  -> method registry / stage adapter
  -> retriever
  -> RankedResult
```

这个边界适合 LongMemEval V1。LongMemEval V1 的价值不在于多跳 path label，而在于长历史、session、turn、time、recency。它比 HotpotQA 和 2WikiMultiHopQA 更适合作为 Memory Stream 的主论据。

新的实验定位如下：

```text
HotpotQA
  -> evidence retrieval / graph retrieval / trainable graph baseline
  -> Memory Stream 不再作为主要适配目标

2WikiMultiHopQA
  -> evidence + path metric benchmark
  -> Memory Stream 不再作为主要适配目标

LongMemEval V1
  -> long-term memory retrieval benchmark
  -> Memory Stream 主适配目标
  -> Dense/BM25/Graph Rerank/Dense-FT 等 baseline 按能力分层适配
```

## 2. 已确认工程事实

### 2.1 现有 consumer request 足够承接 LongMemEval

`graph_memory/retrieval/requests.py` 中已有三类请求：

```python
TextRankingRequest
GraphRankingRequest
TemporalMemoryRankingRequest
```

LongMemEval 至少应投影到：

- `TextRankingRequest`：BM25、Dense、Dense-FT。
- `TemporalMemoryRankingRequest`：Memory Stream。
- `GraphBuildRequest` + `GraphRankingRequest`：Graph Rerank、R-GCN。

不需要新增一个 `LongMemEvalRetrievalRequest`。如果要新增 LongMemEval 专属结构，应放在 dataset record / eval request 层，而不是 retriever 方法层。

### 2.2 当前 dataset selector 仍写死在 HotpotQA / 2Wiki

`graph_memory/datasets/selection.py` 目前只有：

```python
DatasetId = Literal["hotpotqa", "twowiki"]
```

需要扩展为：

```python
DatasetId = Literal["hotpotqa", "twowiki", "longmemeval"]
```

并补齐：

- ranking record validation
- label record validation
- text request projection
- temporal memory request projection
- graph build request projection
- evaluation request projection
- evidence label extraction

### 2.3 当前 workflow 层也写死 dataset 枚举

需要同步修改：

- `scripts/build_graphs.py`
- `scripts/tune_graph_rerank.py`
- `scripts/workflow/workflows.py`
- `scripts/workflow/status.py`
- `scripts/workflow/stage_configs.py`
- `graph_memory/registry/stage_configs.py`

这些位置不应继续只接受 `hotpotqa|twowiki`。

### 2.4 当前 Memory Stream 仍有 HotpotQA sidecar 残留

当前 Memory Stream 设计把外部 importance artifact 作为硬依赖，默认路径仍指向：

```text
data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json
```

这与新方向冲突。新方向下：

- Memory Stream 不再为了 HotpotQA 维护 `split_sources = "importance"`。
- LongMemEval 的 recency 来自真实时间或顺序，而不是 HotpotQA candidate position。
- LongMemEval 的 `has_answer` / answer-session label 不能作为 importance 输入，因为那是 gold label。
- 第一版可以把 importance 设为中性信号，或只接受 LongMemEval 专用的非 gold 外部 importance artifact。

## 3. LongMemEval V1 数据建模策略

### 3.0 Official raw schema and phase-1 file choice

第一版以官方 cleaned retrieval file 为输入，优先使用：

```text
data/longmemeval/raw/longmemeval_s_cleaned.json
```

官方还提供：

```text
longmemeval_m_cleaned.json
longmemeval_oracle.json
```

`longmemeval_s_cleaned` / `longmemeval_m_cleaned` 每个文件约 500 个 evaluation instances，其中包含约 30 个 abstention instances。第一版 retrieval pipeline 应跳过 abstention 样本，或在 run summary 中明确记录跳过数量；不要把 abstention 样本混入 item-level support retrieval 指标。

官方 raw schema 不是 session object list，而是并列数组：

```text
question_id
question_type
question
answer
question_date
haystack_session_ids
haystack_dates
haystack_sessions
answer_session_ids
```

其中：

- `haystack_session_ids[i]`、`haystack_dates[i]`、`haystack_sessions[i]` 共同描述一个 session。
- `haystack_sessions[i]` 是 turn 列表；每个 turn 至少有 `role` 和 `content`。
- evidence turn 可能带有 `has_answer: true`；无该字段时按 `False` 处理。
- `longmemeval_s_cleaned` / `longmemeval_m_cleaned` 的 haystack sessions 已按 timestamp 排序；`oracle` 文件不是 phase-1 主输入，不能默认复用同一顺序假设。

### 3.1 Candidate 粒度

第一版 candidate 粒度使用 message / turn，不使用整 session。

原因：

- BM25/Dense 对较短文本更稳定。
- 官方 cleaned retrieval 数据的 evidence turn 使用 `has_answer` 标注，可以直接映射到 gold support item。
- Session-level candidate 太粗，会让 Full Support/Recall 指标失真。
- Graph 可以通过 `group_key=session:<session_id>` 保留 session 结构。

目标 prepared ranking record：

```python
class LongMemEvalMemoryItem(TypedDict):
    item_id: str
    session_id: str
    session_order: int
    turn_index: int
    global_position: int
    role: str
    datetime: str
    text: str


class LongMemEvalRankingRecord(TypedDict):
    task_id: str
    question: str
    question_datetime: str
    candidate_items: list[LongMemEvalMemoryItem]
    metadata: dict[str, JsonValue]
```

目标 prepared label record：

```python
class LongMemEvalLabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_support_item_ids: list[str]
    gold_support_session_ids: list[str]
    gold_dependency_edges: list[list[str]]
    metadata: dict[str, JsonValue]
```

`gold_dependency_edges` 第一版保持空列表。LongMemEval V1 的重点不是 reasoning path；不要为了 path metric 人工制造 gold path。

### 3.2 Raw 字段和可见性

LongMemEval raw adapter 应只让 input-visible 字段进入 ranking record：

| Raw concept | Ranking record usage |
|---|---|
| `question_id` | `task_id = "longmem_" + raw_id` |
| `question_type` | non-label metadata |
| question text | `question` / `TextRankingRequest.query_text` |
| `question_date` | recency anchor metadata |
| `haystack_session_ids` | session id metadata / graph group key |
| `haystack_dates` | session datetime metadata |
| `haystack_sessions` | candidate turn source |
| turn `role` | candidate metadata |
| turn `content` | candidate text |

Label-only 字段只能进入 label record：

| Raw concept | Label usage |
|---|---|
| answer | `gold_answer` |
| answer session ids | `gold_support_session_ids` |
| message / turn `has_answer` | `gold_support_item_ids` |

禁止把下面信息写入 ranking record、graph nodes、visible graph edges 或 retrieval request metadata：

```text
has_answer
answer
answer_session_ids
gold_support_item_ids
gold_support_session_ids
```

### 3.3 Item id 和排序规则

Prepared artifact 中的 `item_id` 使用稳定 position id：

```text
m0, m1, m2, ...
```

`global_position` 按官方 cleaned haystack 顺序 flatten：

```text
for session_order, (session_id, session_date, turns) in enumerate(
    zip(haystack_session_ids, haystack_dates, haystack_sessions)
):
    for turn_index, turn in enumerate(turns):
        ...
```

`longmemeval_s_cleaned` / `longmemeval_m_cleaned` 已按时间排序，因此该 flatten 顺序就是 phase-1 的时间顺序。若后续支持未排序或 oracle 输入，converter 必须显式声明排序策略；不要默默把 oracle 当作 chronological input。

转换后要求：

```text
candidate_items[i].item_id == f"m{i}"
candidate_items[i].global_position == i
```

这与 HotpotQA/2Wiki 当前 `m{position}` contract 保持一致，降低 validator、retriever 和 graph builder 的变更面。Graph projection 不能把 `global_position` 当作 session 内 sequential index；session 内顺序应使用 `turn_index`。

### 3.4 Recency 定义

Memory Stream 的 LongMemEval recency 第一版使用 candidate 在 cleaned haystack 中的相对顺序：

```python
recency_raw = recency_decay ** (max_position - global_position)
```

如果 `question_date` 和 `haystack_dates` 可稳定解析，metadata 同时保留 session-level 时间差：

```python
seconds_before_question = question_timestamp - session_timestamp
```

但第一版 scoring 仍用 position-based recency，原因是它与现有 `pseudo_recency_scores()` 兼容，且不会被 timestamp timezone/format 问题阻塞。正式报告中应称为 order-based recency，不要声称已经实现 real-time decay。

后续如果要用 real-time decay，应单独扩展：

```python
recency_raw = exp(-seconds_before_question / half_life_seconds)
```

这不是第一阶段目标。

## 4. Baseline 可行性分层

| Method | 第一阶段适配 | 工程量 | 可行性判断 | 说明 |
|---|---:|---:|---|---|
| `bm25` | 是 | 低 | 高 | 只需要 `TextRankingRequest`。 |
| `dense` | 是 | 低 | 高 | 只需要 `TextRankingRequest` 和现有 encoder config。 |
| `memory_stream` | 是 | 中 | 高 | 需要 LongMemEval temporal projector 和去 HotpotQA sidecar。 |
| `bm25_graph_rerank` | 第二阶段 | 中 | 中高 | 需要 LongMemEval graph projection 和 graph-rerank tuning dataset 化。 |
| `dense_graph_rerank` | 第二阶段 | 中 | 中高 | 推荐作为 graph baseline。 |
| `dense_ft` | 第三阶段 | 中偏高 | 中 | 取决于训练 split 是否足够可信。 |
| `dense_rgcn_graph_retriever` | 第三阶段之后 | 高 | 中低 | 要先证明 graph semantics 和 train pairs 合理。 |
| `dense_ft_rgcn_graph_retriever` | 最后 | 高 | 低到中 | 依赖 Dense-FT 和 R-GCN 两条链稳定。 |

第一轮不建议同时承诺所有 trainable baseline。先把 LongMemEval dataset、BM25、Dense、Memory Stream 跑通，确认指标和 workflow contract，再扩展 graph / trainable 方法。

## 5. 目标调用流

### 5.1 Stateless / Memory Stream 第一阶段

```text
prepare_longmemeval
  -> train/dev/test .input.json
  -> train/dev/test .labels.json

build_graphs
  -> LongMemEval visible graph
  -> 第一阶段也运行，因为当前 workflow/evaluate 需要 graph artifact

retrieve: bm25
  -> TextRankingRequest

retrieve: dense
  -> TextRankingRequest

retrieve: memory_stream
  -> TemporalMemoryRankingRequest
  -> relevance from dense seed ranker
  -> recency from position_by_item_id
  -> importance from request-owned neutral or non-gold scores

evaluate
  -> LongMemEval support metrics

aggregate
  -> main / efficiency tables
```

### 5.2 Graph Rerank 第二阶段

```text
prepare_longmemeval
  |
build_graphs
  -> q node
  -> message/turn nodes
  -> sequential edges ordered by time
  -> entity/query/bridge edges from existing rules
  |
tune_graph_rerank
  -> dataset=longmemeval
  |
retrieve: bm25_graph_rerank / dense_graph_rerank
  |
evaluate
```

第一版不新增 edge type。LongMemEval 的 session/time 顺序先复用现有 `sequential` edge type，session 信息放在 node metadata / group key 中。新增 `temporal`、`same_session` 等 edge type 会牵动 `EdgeType`、graph validation、neighbor weights、R-GCN tensorization 和 checkpoint compatibility，应作为独立后续 change。

### 5.3 Dense-FT / R-GCN 第三阶段

```text
prepare_longmemeval
  |
build_graphs
  |
build_train_pairs
  -> positives from gold_support_item_ids
  -> negatives from random/BM25/dense/graph-neighbor sampling
  |
train: dense_ft
  |
retrieve/evaluate: dense_ft
```

R-GCN 继续排在 Dense-FT 之后：

```text
build_train_pairs
  |
train: dense_rgcn_graph_retriever
  |
retrieve/evaluate
```

如果 LongMemEval V1 没有足够可靠的 official train/dev/test split，则第三阶段必须在报告里明确：

```text
trainable results use repository-defined deterministic split, not an official LongMemEval training protocol.
```

## 6. 文件责任图

### 6.1 LongMemEval dataset package

- Create `graph_memory/datasets/longmemeval/records.py`
  - 定义 raw dataclass、ranking record、label record、conversion result。

- Create `graph_memory/datasets/longmemeval/parser.py`
  - 从 raw JSON/JSONL 解析 LongMemEval V1 样本。
  - 校验 question、history/session/message、timestamp、answer label 的基本结构。
  - 不做 retrieval projection。

- Create `graph_memory/datasets/longmemeval/converter.py`
  - 将 raw example flatten 为 message/turn candidates。
  - 生成 `m{position}` item ids。
  - 生成 label-only `gold_support_item_ids` 和 `gold_support_session_ids`。
  - 严格禁止 gold 字段进入 ranking record。

- Create `graph_memory/datasets/longmemeval/projectors.py`
  - `LongMemEvalToTextRankingRequest`
  - `LongMemEvalToTemporalMemoryRankingRequest`
  - `LongMemEvalToGraphBuildRequest`
  - `LongMemEvalToEvidenceEvaluationRequest` 或后续 `LongMemEvalToEvaluationRequest`

- Create `graph_memory/datasets/longmemeval/compatibility.py`
  - coerce prepared JSON records to typed records。
  - build combined inspection artifact。

- Create `graph_memory/datasets/longmemeval/__init__.py`
  - 导出 public dataset API。

### 6.2 Dataset selector and validation

- Modify `graph_memory/datasets/selection.py`
  - 增加 `longmemeval` 分支。
  - 所有 consumer helper 保持 dataset-owned projector。

- Modify `graph_memory/validation/tasks.py`
  - 增加 LongMemEval ranking/label validator。
  - 保持 HotpotQA/2Wiki validator 行为不变。

- Modify `graph_memory/validation/__init__.py`
  - 导出新 validator。

### 6.3 Prepare CLI

- Create `scripts/prepare_longmemeval.py`
  - 与 `prepare_hotpotqa.py` / `prepare_2wiki.py` 对称。
  - 输入 raw local JSON/JSONL。
  - 输出 `.input.json`、`.labels.json`、`.combined.json` 和 run summary。

- Modify `scripts/workflow/workflows.py`
  - `_prepare_script("longmemeval") -> "scripts/prepare_longmemeval.py"`。
  - 删除或限制 HotpotQA-only `split_sources = "importance"` 对 Memory Stream 的特殊路径。

- Modify `scripts/workflow/status.py`
  - 支持 `longmemeval` prepare status。

### 6.4 Graph and retrieval scripts

- Modify `scripts/build_graphs.py`
  - `choices=("hotpotqa", "twowiki", "longmemeval")`。

- Modify `scripts/run_retrieval.py`
  - 保持通过 `datasets.selection` 构造 text request。
  - Memory Stream 不应再要求 HotpotQA 默认 importance artifact。

- Modify `graph_memory/stages/retrieve.py`
  - 对 Memory Stream 构造 `MemoryStreamBuildPayload` 时，优先使用 request-owned `TemporalMemoryRankingRequest.importance_by_item_id`。
  - 若保留 external importance，则必须是显式配置，且不能默认到 HotpotQA 路径。

- Modify `graph_memory/registry/retrieval.py`
  - 调整 `MemoryStreamBuildPayload`，使 importance artifact 不再是 LongMemEval 第一版硬依赖。

- Modify `graph_memory/registry/retrieval_builders.py`
  - LongMemEval Memory Stream 应能在没有 external sidecar 的情况下构建。
  - 如果 request importance 为空，按 scoring config 的 `importance_weight=0.0` 或全 0 信号处理。

### 6.5 Tuning and metrics

- Modify `scripts/tune_memory_stream.py`
  - 增加 `--dataset` 参数。
  - 移除 HotpotQA projector/validator 直接 import。
  - 改用 `text_ranking_requests_for_dataset()`、`temporal_memory_requests_for_dataset()`、`evidence_labels_for_dataset()`。

- Modify `scripts/workflow/workflows.py`
  - `_memory_stream_tune_argv()` 传入 `--dataset longmemeval`。

- Modify `graph_memory/evaluation/suites.py`
  - 内部 smoke 可以临时复用 evidence suite 计算 item-level support recall。
  - 第一版可报告 baseline 必须增加 LongMemEval-specific suite，输出 turn/session support metrics，避免把 LongMemEval 写成 HotpotQA supporting-fact 任务。

- Modify `scripts/evaluate_retrieval.py` and `graph_memory/stages/evaluate.py`
  - 按 dataset/task 或 stage config 选择 metric suite。

### 6.6 Experiment configs and docs

- Create `configs/experiments/longmemeval_v1_memory_retrieval.json`
  - 第一阶段只包含 `bm25`、`dense`、`memory_stream`。

- Create `configs/experiments/longmemeval_v1_graph_retrieval.json`
  - 第二阶段包含 `bm25`、`dense`、`memory_stream`、`bm25_graph_rerank`、`dense_graph_rerank`。

- Create or later add `configs/experiments/longmemeval_v1_trainable_retrieval.json`
  - 第三阶段再包含 `dense_ft` 和 R-GCN methods。

- Modify or remove `configs/experiments/hotpotqa_memory_stream.json`
  - 新方向下它不应继续作为 active experiment recipe。
  - 推荐删除；如果担心旧结果复现，改名到 docs/report 说明，而不是 active config。

- Modify docs that still imply HotpotQA/2Wiki are Memory Stream targets。
  - `docs/10-plans/memory-stream-implementation-plan.md`
  - `docs/10-plans/generic-grid-search-and-memory-stream-tuning-plan.md`
  - `docs/configs/search_spaces/memory_stream.md` if present

## 7. Prepared artifact contract

### 7.1 Ranking record

LongMemEval prepared input should look like:

```json
{
  "task_id": "longmem_example_001",
  "question": "Where did I say I planned to meet Alex?",
  "question_datetime": "2024-01-10T12:00:00",
  "candidate_items": [
    {
      "item_id": "m0",
      "session_id": "session_001",
      "session_order": 0,
      "turn_index": 0,
      "global_position": 0,
      "role": "user",
      "datetime": "2023-12-01T09:00:00",
      "text": "Let's meet Alex at the library tomorrow."
    }
  ],
  "metadata": {
    "dataset": "longmemeval_v1",
    "raw_id": "example_001",
    "question_type": "single-session-user",
    "candidate_granularity": "turn",
    "raw_file": "longmemeval_s_cleaned.json"
  }
}
```

No gold fields are allowed in this record.

### 7.2 Label record

LongMemEval label should look like:

```json
{
  "task_id": "longmem_example_001",
  "gold_answer": "At the library.",
  "gold_support_item_ids": ["m0"],
  "gold_support_session_ids": ["session_001"],
  "gold_dependency_edges": [],
  "metadata": {
    "dataset": "longmemeval_v1",
    "raw_id": "example_001",
    "support_label_source": "has_answer"
  }
}
```

If a raw sample has answer-session labels but no turn/message support labels, the converter must not invent precise message labels silently. It should either:

1. reject that sample for item-level retrieval evaluation, or
2. write session-level labels only and mark `support_label_source = "session_only"`.

First implementation should choose option 1 for the main evidence-style pipeline. Option 2 belongs with a LongMemEval-specific metric suite.

### 7.3 Text projection

`LongMemEvalToTextRankingRequest` should format candidates as:

```python
TextCandidate(
    item_id=item["item_id"],
    text=f'{item["role"]}: {item["text"]}',
    metadata={
        "session_id": item["session_id"],
        "session_order": item["session_order"],
        "turn_index": item["turn_index"],
        "sequence_index": item["turn_index"],
        "position": item["global_position"],
        "datetime": item["datetime"],
        "source_ref": item["session_id"],
    },
)
```

### 7.4 Temporal projection

`LongMemEvalToTemporalMemoryRankingRequest` should produce:

```python
TemporalMemoryRankingRequest(
    task_id=record["task_id"],
    query_text=record["question"],
    candidates=text_request.candidates,
    importance_by_item_id={
        candidate.item_id: 0.0
        for candidate in text_request.candidates
    },
    metadata={
        "position_by_item_id": {
            item["item_id"]: item["global_position"]
            for item in record["candidate_items"]
        },
        "datetime_by_item_id": {
            item["item_id"]: item["datetime"]
            for item in record["candidate_items"]
        },
        "question_datetime": record["question_datetime"],
        "session_order_by_item_id": {
            item["item_id"]: item["session_order"]
            for item in record["candidate_items"]
        },
    },
)
```

The zero importance map is intentional for phase 1. It avoids leaking `has_answer` while keeping the Memory Stream scoring contract complete. The experiment config should set:

```json
{
  "memory_stream_relevance_weight": 1.0,
  "memory_stream_recency_weight": 0.1,
  "memory_stream_importance_weight": 0.0,
  "memory_stream_recency_decay": 0.99
}
```

Tuning may later select these weights from `configs/search_spaces/memory_stream.json`.

### 7.5 Graph projection

`LongMemEvalToGraphBuildRequest` should create nodes:

```python
GraphBuildNode(
    node_id=item["item_id"],
    text=item["text"],
    node_kind="conversation_turn",
    source_ref=item["session_id"],
    group_key=f'session:{item["session_id"]}',
    sequence_index=item["turn_index"],
    metadata={
        "session_id": item["session_id"],
        "session_order": item["session_order"],
        "turn_index": item["turn_index"],
        "global_position": item["global_position"],
        "role": item["role"],
        "datetime": item["datetime"],
    },
)
```

First version should leave `input_visible_edges=()` and rely on existing graph rules:

- `SequentialEdgeRule`
- `QueryOverlapEdgeRule`
- `EntityOverlapEdgeRule`
- `BridgeEdgeRule`

With `group_key=session:<session_id>` and `sequence_index=turn_index`, current `SequentialEdgeRule` connects adjacent turns within the same session. Cross-session temporal edges are not part of phase 1; if they are needed later, add them as a separate task after checking graph validation, neighbor weights, R-GCN relation vocabulary, and checkpoint compatibility. Do not introduce a new edge type in the first LongMemEval change.

## 8. Implementation Tasks

### Task 0: Supersede the old HotpotQA Memory Stream plan

**Files:**

- Modify: `openspec/changes/add-memory-stream-retrieval/proposal.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/design.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/tasks.md`
- Modify: `docs/10-plans/memory-stream-implementation-plan.md`

- [ ] **Step 0.1: Decide whether to update or replace the active OpenSpec change**

Current `add-memory-stream-retrieval` is HotpotQA-centric and still in progress. Before implementation, choose one of these two concrete paths:

```text
Preferred:
  create a new OpenSpec change named adapt-longmemeval-v1-memory-baselines
  leave add-memory-stream-retrieval unimplemented or archive/supersede it explicitly

Alternative:
  rewrite add-memory-stream-retrieval so it targets LongMemEval V1 only
```

The preferred path is cleaner because the old change contains HotpotQA first-1000 importance sidecar requirements that are no longer desired.

- [ ] **Step 0.2: Remove active docs language that treats HotpotQA as the Memory Stream target**

Search:

```powershell
rg -n "hotpotqa_memory_stream|dev.first_1000.importance|split_sources.*importance|Memory Stream" docs openspec configs scripts graph_memory tests
```

Expected outcome: every remaining HotpotQA/2Wiki Memory Stream mention is either historical/reporting text or explicitly marked as retired.

### Task 1: Add LongMemEval records and parser

**Files:**

- Create: `graph_memory/datasets/longmemeval/records.py`
- Create: `graph_memory/datasets/longmemeval/parser.py`
- Create: `tests/test_longmemeval_parser.py`

- [x] **Step 1.1: Write parser fixture tests**

Test cases:

```python
def test_parse_longmemeval_example_accepts_parallel_session_arrays() -> None:
    raw = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "Where did I plan to meet Alex?",
        "question_date": "2024-01-10",
        "answer": "At the library.",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I will meet Alex at the library.", "has_answer": True},
                {"role": "assistant", "content": "Got it."},
            ]
        ],
    }
    example = parse_longmemeval_example(raw, record_index=0)
    assert example.raw_id == "q1"
    assert example.question_type == "single-session-user"
    assert example.question == "Where did I plan to meet Alex?"
    assert example.answer == "At the library."
    assert example.sessions[0].session_id == "s1"
    assert example.sessions[0].session_order == 0
    assert example.sessions[0].messages[0].has_answer is True
    assert example.sessions[0].messages[1].has_answer is False
```

Also test:

- missing question fails
- empty haystack fails
- mismatched `haystack_session_ids` / `haystack_dates` / `haystack_sessions` lengths fail
- turn without `content` fails
- sample with no `has_answer=True` fails or is marked unusable according to the final parser contract
- abstention question types are skipped or explicitly counted as unsupported for item-level retrieval

- [x] **Step 1.2: Implement records**

Define dataclasses and TypedDicts:

```python
@dataclass(frozen=True)
class LongMemEvalMessage:
    role: str
    text: str
    has_answer: bool


@dataclass(frozen=True)
class LongMemEvalSession:
    session_id: str
    session_order: int
    datetime: str
    messages: tuple[LongMemEvalMessage, ...]


@dataclass(frozen=True)
class LongMemEvalExample:
    raw_id: str
    question_type: str
    question: str
    question_datetime: str
    answer: str
    answer_session_ids: tuple[str, ...]
    sessions: tuple[LongMemEvalSession, ...]
```

Prepared record types follow section 7.

- [x] **Step 1.3: Implement parser with strict field errors**

Parser responsibility:

```text
raw object -> LongMemEvalExample
```

It must not create `TextRankingRequest`, graph nodes, or labels.

- [x] **Step 1.4: Run parser tests**

Run outside the Codex Windows sandbox:

```powershell
uv run pytest tests/test_longmemeval_parser.py -q
```

Expected: parser tests pass.

### Task 2: Add converter and leakage tests

**Files:**

- Create: `graph_memory/datasets/longmemeval/converter.py`
- Create: `tests/test_longmemeval_converter.py`
- Create: `tests/test_longmemeval_leakage_boundaries.py`

- [x] **Step 2.1: Write converter tests**

Core expectations:

```python
def test_convert_longmemeval_example_flattens_messages_to_position_ids() -> None:
    converted = convert_longmemeval_example(_example())
    assert converted.ranking_record["task_id"] == "longmem_q1"
    assert [item["item_id"] for item in converted.ranking_record["candidate_items"]] == ["m0", "m1"]
    assert [item["session_order"] for item in converted.ranking_record["candidate_items"]] == [0, 0]
    assert [item["turn_index"] for item in converted.ranking_record["candidate_items"]] == [0, 1]
    assert [item["global_position"] for item in converted.ranking_record["candidate_items"]] == [0, 1]
```

Gold support expectation:

```python
def test_convert_longmemeval_example_maps_has_answer_to_gold_support_items() -> None:
    converted = convert_longmemeval_example(_example())
    assert converted.label_record["gold_support_item_ids"] == ["m0"]
    assert converted.label_record["gold_support_session_ids"] == ["s1"]
```

Leakage expectation:

```python
def test_longmemeval_ranking_record_contains_no_gold_fields() -> None:
    ranking_record = convert_longmemeval_example(_example()).ranking_record
    rendered = json.dumps(ranking_record, ensure_ascii=False)
    assert "has_answer" not in rendered
    assert "gold" not in rendered
    assert "answer_session_ids" not in rendered
```

- [x] **Step 2.2: Implement converter**

Conversion rules:

```text
task_id = "longmem_" + raw_id
candidate item order = official cleaned haystack order flattened by session_order then turn_index
item_id = "m{global_position}"
session_order = index in haystack_session_ids / haystack_dates / haystack_sessions
turn_index = index inside one haystack session
label positives = turns where has_answer is true
label sessions = answer_session_ids from raw
gold_dependency_edges = []
```

- [x] **Step 2.3: Run converter and leakage tests**

```powershell
uv run pytest tests/test_longmemeval_converter.py tests/test_longmemeval_leakage_boundaries.py -q
```

Expected: all tests pass.

### Task 3: Add projectors and dataset selector integration

**Files:**

- Create: `graph_memory/datasets/longmemeval/projectors.py`
- Create: `graph_memory/datasets/longmemeval/compatibility.py`
- Create: `graph_memory/datasets/longmemeval/__init__.py`
- Modify: `graph_memory/datasets/selection.py`
- Modify: `graph_memory/validation/tasks.py`
- Modify: `graph_memory/validation/__init__.py`
- Test: `tests/test_longmemeval_projectors.py`
- Test: `tests/test_longmemeval_dataset_selector.py`

- [x] **Step 3.1: Test `TextRankingRequest` projection**

Expected:

```python
request = LongMemEvalToTextRankingRequest().project(_ranking_record())
assert request.task_id == "longmem_q1"
assert request.query_text == _ranking_record()["question"]
assert request.candidates[0].item_id == "m0"
assert request.candidates[0].metadata["session_id"] == "s1"
assert request.candidates[0].metadata["session_order"] == 0
assert request.candidates[0].metadata["sequence_index"] == 0
assert request.candidates[0].metadata["position"] == 0
```

- [x] **Step 3.2: Test `TemporalMemoryRankingRequest` projection**

Expected:

```python
request = LongMemEvalToTemporalMemoryRankingRequest().project(_ranking_record())
assert request.importance_by_item_id == {"m0": 0.0, "m1": 0.0}
assert request.metadata["position_by_item_id"] == {"m0": 0, "m1": 1}
assert request.metadata["session_order_by_item_id"] == {"m0": 0, "m1": 0}
```

- [x] **Step 3.3: Test `GraphBuildRequest` projection**

Expected:

```python
request = LongMemEvalToGraphBuildRequest().project(_ranking_record())
assert request.nodes[0].node_kind == "conversation_turn"
assert request.nodes[0].group_key == "session:s1"
assert request.nodes[0].sequence_index == 0
assert request.nodes[0].metadata["global_position"] == 0
assert request.input_visible_edges == ()
```

- [x] **Step 3.4: Wire `selection.py`**

Add `longmemeval` to:

```python
validate_ranking_records_for_dataset()
validate_label_records_for_dataset()
text_ranking_requests_for_dataset()
temporal_memory_requests_for_dataset()
graph_build_requests_for_dataset()
evidence_evaluation_request_for_dataset()
evidence_labels_for_dataset()
```

- [x] **Step 3.5: Run selector tests**

```powershell
uv run pytest tests/test_longmemeval_projectors.py tests/test_longmemeval_dataset_selector.py -q
```

Expected: all tests pass.

### Task 4: Add prepare CLI and workflow dataset support

**Files:**

- Create: `scripts/prepare_longmemeval.py`
- Modify: `scripts/build_graphs.py`
- Modify: `scripts/tune_graph_rerank.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `scripts/workflow/status.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Test: `tests/test_longmemeval_prepare_cli.py`
- Test: `tests/test_longmemeval_workflow.py`

- [x] **Step 4.1: Implement `prepare_longmemeval.py`**

The CLI should mirror existing prepare scripts:

```powershell
uv run python scripts/prepare_longmemeval.py `
  --input data/longmemeval/raw/longmemeval_s_cleaned.json `
  --output_input data/longmemeval/processed/dev.input.json `
  --output_labels data/longmemeval/processed/dev.labels.json `
  --output_combined data/longmemeval/processed/dev.combined.json `
  --max_examples 10 `
  --seed 13 `
  --offset 0
```

The script writes:

```text
*.input.json
*.labels.json
*.combined.json
*.input.run_summary.json
```

- [x] **Step 4.2: Add CLI choices**

Update dataset choices:

```python
choices=("hotpotqa", "twowiki", "longmemeval")
```

Locations:

- `scripts/build_graphs.py`
- `scripts/tune_graph_rerank.py`
- config parser surfaces that use `DatasetId`

- [x] **Step 4.3: Wire workflow prepare script**

Expected mapping:

```python
def _prepare_script(dataset: str) -> str:
    if dataset == "hotpotqa":
        return "scripts/prepare_hotpotqa.py"
    if dataset == "twowiki":
        return "scripts/prepare_2wiki.py"
    if dataset == "longmemeval":
        return "scripts/prepare_longmemeval.py"
    raise ValueError(...)
```

- [x] **Step 4.4: Remove HotpotQA-only importance split from new Memory Stream path**

For LongMemEval experiments, workflow should use normal raw split sources:

```json
"split_sources": {
  "train": "train",
  "dev": "dev",
  "test": "dev"
}
```

Do not route LongMemEval through `split_sources = "importance"`.

- [x] **Step 4.5: Run prepare/workflow tests**

```powershell
uv run pytest tests/test_longmemeval_prepare_cli.py tests/test_longmemeval_workflow.py -q
```

Expected: all tests pass.

### Task 5: Refactor Memory Stream to LongMemEval request-owned signals

**Files:**

- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `graph_memory/stages/retrieve.py`
- Modify: `scripts/run_retrieval.py`
- Modify: `graph_memory/retrieval/methods/memory_stream/method.py`
- Modify: `graph_memory/retrieval/methods/memory_stream/scoring.py`
- Test: `tests/test_memory_stream_method.py`
- Test: `tests/test_retrieval_registry_builders.py`
- Test: `tests/test_config_run_retrieval.py`

- [x] **Step 5.1: Write failing test for Memory Stream without external artifact**

Expected behavior:

```python
def test_memory_stream_builds_from_temporal_requests_without_importance_artifact() -> None:
    settings = MemoryStreamRetrievalSettings(
        top_k=10,
        encoder=DenseEncoderSettings(
            model_name="fake",
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        scoring=MemoryStreamScoringConfig(
            relevance_weight=1.0,
            recency_weight=0.1,
            importance_weight=0.0,
        ),
    )
    payload = MemoryStreamBuildPayload(
        temporal_requests=[_temporal_request()],
        dense_encoder=_fake_encoder(),
    )
    built = build_retrieval_registry().build(settings, payload)
    assert built.provenance.importance is None
```

- [x] **Step 5.2: Adjust `MemoryStreamBuildPayload`**

Target shape:

```python
@dataclass(frozen=True)
class MemoryStreamBuildPayload:
    temporal_requests: list[TemporalMemoryRankingRequest]
    scoring_config: MemoryStreamScoringConfig | None = None
    dense_encoder: SentenceEncoder | None = None
    importance_artifact: ImportanceArtifact | None = None
    importance_path: Path | None = None
    importance_sha256: str | None = None
```

External importance becomes optional, not the default LongMemEval path.

- [x] **Step 5.3: Builder behavior**

Builder rules:

```text
if importance_artifact is supplied:
  validate/select sidecar scores
  overlay scores into TemporalMemoryRankingRequest
else:
  trust request.importance_by_item_id
  require it covers every candidate when importance_weight > 0
```

If `importance_weight == 0.0`, an empty or all-zero map is valid.

- [x] **Step 5.4: Run Memory Stream focused tests**

```powershell
uv run pytest tests/test_memory_stream_method.py tests/test_retrieval_registry_builders.py tests/test_config_run_retrieval.py -q
```

Expected: all tests pass.

### Task 6: Make Memory Stream tuning dataset-aware

**Files:**

- Modify: `scripts/tune_memory_stream.py`
- Modify: `graph_memory/retrieval/tuning/memory_stream.py`
- Modify: `scripts/workflow/workflows.py`
- Test: `tests/test_memory_stream_tuning.py`
- Test: `tests/test_cli_contracts.py`
- Test: `tests/test_longmemeval_workflow.py`

- [x] **Step 6.1: Add `--dataset` to parser**

Parser contract:

```python
parser.add_argument("--dataset", choices=("hotpotqa", "twowiki", "longmemeval"), default="hotpotqa")
```

During final cleanup, default can remain HotpotQA for backward CLI compatibility, but the LongMemEval workflow must pass `--dataset longmemeval`.

- [x] **Step 6.2: Replace HotpotQA direct imports**

Remove direct imports from:

```python
graph_memory.datasets.hotpotqa.projectors
graph_memory.datasets.hotpotqa.records
validate_hotpotqa_ranking_records
validate_hotpotqa_label_records
```

Use dataset selector helpers instead.

- [x] **Step 6.3: Tune with request-owned importance**

If no importance path is supplied, tuning should run with the temporal request importance maps. For LongMemEval phase 1, this means tuning relevance/recency weights while importance weight is fixed to zero unless a non-gold external importance source is explicitly supplied. The LongMemEval phase-1 search space should therefore use `importance_weight: [0.0]`.

- [x] **Step 6.4: Run tuning tests**

```powershell
uv run pytest tests/test_memory_stream_tuning.py tests/test_cli_contracts.py tests/test_longmemeval_workflow.py -q
```

Expected: all tests pass.

### Task 7: Add LongMemEval experiment configs

Do not add or validate the first-stage config until Task 5 and Task 6 have removed the hard requirement for a HotpotQA importance sidecar. With current code, a `memory_stream` config would still generate a selected-config/importance path contract and fail before LongMemEval retrieval can run.

**Files:**

- Create: `configs/experiments/longmemeval_v1_memory_retrieval.json`
- Create: `configs/experiments/longmemeval_v1_graph_retrieval.json`
- Optionally later create: `configs/experiments/longmemeval_v1_trainable_retrieval.json`
- Test: `tests/test_current_manifest_contract.py`
- Test: `tests/test_workflow_orchestration.py`

- [x] **Step 7.1: Add first-stage config**

Target:

```json
{
  "dataset": "longmemeval",
  "default_profile": "quick",
  "enable_ablation": false,
  "defaults": {
    "dense_encoder": "models/intfloat-e5-base-v2",
    "passage_prefix": "passage: ",
    "query_prefix": "query: ",
    "seed": 13,
    "top_k": 10
  },
  "graph": {
    "max_bridge_edges": 50,
    "max_entity_neighbors": 10,
    "max_query_overlap": 20,
    "use_spacy": false
  },
  "methods": [
    "bm25",
    "dense",
    "memory_stream"
  ],
  "profiles": {
    "smoke": {
      "train_examples": 1,
      "dev_examples": 1,
      "test_examples": 1
    },
    "quick": {
      "train_examples": 50,
      "dev_examples": 50,
      "test_examples": 50
    },
    "full": {
      "train_examples": 300,
      "dev_examples": 100,
      "test_examples": 70
    }
  },
  "raw": {
    "cleaned_s": "data/longmemeval/raw/longmemeval_s_cleaned.json"
  },
  "recipe": "longmemeval_v1_memory_retrieval",
  "search_spaces": {
    "memory_stream": "configs/search_spaces/memory_stream.json"
  },
  "split_offsets": {
    "train": 0,
    "dev": 300,
    "test": 400
  },
  "split_sources": {
    "train": "cleaned_s",
    "dev": "cleaned_s",
    "test": "cleaned_s"
  },
  "task": "long_memory_retrieval",
  "memory_stream_relevance_weight": 1.0,
  "memory_stream_recency_weight": 0.1,
  "memory_stream_importance_weight": 0.0,
  "memory_stream_recency_decay": 0.99
}
```

Counts should be adjusted after inspecting the downloaded cleaned file and after applying the abstention skip rule. The important contract is deterministic non-overlapping offsets over repository-defined splits; this is not an official LongMemEval train/dev/test protocol.

- [x] **Step 7.2: Add graph config**

Same as first-stage config, but methods include:

```json
[
  "bm25",
  "dense",
  "memory_stream",
  "bm25_graph_rerank",
  "dense_graph_rerank"
]
```

- [x] **Step 7.3: Validate manifest initialization**

```powershell
uv run python scripts/experiment.py init longmem-smoke --config longmemeval_v1_memory_retrieval --profile smoke --force
uv run python scripts/experiment.py status longmem-smoke
```

Expected:

- prepare commands use `scripts/prepare_longmemeval.py`
- retrieve config for `memory_stream` has dataset `longmemeval`
- no HotpotQA importance path appears in manifest or stage configs

### Task 8: Retire HotpotQA/2Wiki Memory Stream active surfaces

**Files:**

- Delete or rename: `configs/experiments/hotpotqa_memory_stream.json`
- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Modify: `configs/experiments/2wiki_evidence_retrieval.json`
- Modify: docs/tests that expect Memory Stream on HotpotQA/2Wiki

- [x] **Step 8.1: Remove active HotpotQA Memory Stream recipe**

Preferred action:

```text
delete configs/experiments/hotpotqa_memory_stream.json
```

If deletion is too disruptive, rename to a retired docs artifact outside `configs/experiments/`, for example:

```text
docs/configs/retired/hotpotqa_memory_stream.json
```

Do not leave it in active experiment config discovery.

- [x] **Step 8.2: Ensure normal HotpotQA/2Wiki configs do not list Memory Stream**

`methods` arrays for active HotpotQA and 2Wiki evidence configs should not contain:

```text
memory_stream
```

It is acceptable for old reports to mention previous Memory Stream results as historical.

- [x] **Step 8.3: Run active config discovery tests**

```powershell
uv run pytest tests/test_current_manifest_contract.py tests/test_config_loader.py tests/test_workflow_orchestration.py -q
```

Expected: active configs are valid and no HotpotQA Memory Stream recipe is listed.

### Task 9: Add LongMemEval metric suite or explicit evidence-suite mapping

**Files:**

- Modify: `graph_memory/evaluation/requests.py`
- Modify: `graph_memory/evaluation/suites.py`
- Modify: `graph_memory/evaluation/tables.py`
- Modify: `graph_memory/stages/evaluate.py`
- Modify: `scripts/aggregate_tables.py`
- Test: `tests/test_longmemeval_metrics.py`

- [x] **Step 9.1: Temporary smoke-only mapping**

If keeping the existing evidence suite during early smoke, document that:

```text
Recall@k / Full Support@k are turn support metrics for LongMemEval,
not HotpotQA supporting-fact metrics.
```

This is acceptable only for internal wiring validation. Do not use evidence-suite column names for mentor-facing or paper-style LongMemEval baseline results.

- [x] **Step 9.2: Required reportable metric suite**

Add LongMemEval-specific aggregate columns:

```text
Turn Recall@5
Turn Recall@10
Full Turn Support@10
Session Recall@5
Session Recall@10
Full Session Support@10
MRR
Retrieval Latency / Query
Memory Size
```

This avoids reporting LongMemEval as if it were HotpotQA evidence retrieval. First-stage implementation can be considered engineering-complete with smoke-only evidence-suite mapping, but experiment results should not be reported until this suite is available.

- [x] **Step 9.3: Test session metric behavior**

Expected:

```python
ranked_node_ids = ["m2", "m0"]
item_to_session = {"m0": "s1", "m1": "s2", "m2": "s3"}
gold_sessions = {"s1"}
assert session_recall_at(ranked_node_ids, item_to_session, gold_sessions, 2) == 1.0
```

### Task 10: Verify first-stage end-to-end LongMemEval workflow

**Files:**

- Test: `tests/test_longmemeval_workflow_smoke.py`
- Runtime artifacts: `runs/<name>/...`

- [x] **Step 10.1: Run focused tests**

```powershell
uv run pytest `
  tests/test_longmemeval_parser.py `
  tests/test_longmemeval_converter.py `
  tests/test_longmemeval_projectors.py `
  tests/test_longmemeval_prepare_cli.py `
  tests/test_longmemeval_dataset_selector.py `
  tests/test_memory_stream_method.py `
  tests/test_memory_stream_tuning.py `
  -q
```

Expected: all pass.

- [ ] **Step 10.2: Run smoke workflow**

Local raw file note: not run in this implementation pass because `data/longmemeval/raw/longmemeval_s_cleaned.json` is not present on this machine. `experiment.py init` and `experiment.py status` were verified against `longmemeval_v1_memory_retrieval` using a temporary run root under `C:\tmp`.

```powershell
uv run python scripts/experiment.py run longmem-v1-smoke `
  --config longmemeval_v1_memory_retrieval `
  --profile smoke `
  --force
```

Expected artifacts:

```text
runs/longmem-v1-smoke/inputs/test.input.json
runs/longmem-v1-smoke/graphs/test.graphs.json
runs/longmem-v1-smoke/predictions/test.bm25.ranked.json
runs/longmem-v1-smoke/predictions/test.dense.ranked.json
runs/longmem-v1-smoke/predictions/test.memory_stream.ranked.json
runs/longmem-v1-smoke/tables/main_results.csv
runs/longmem-v1-smoke/tables/efficiency_results.csv
```

Expected invariant:

```text
No path under runs/longmem-v1-smoke mentions data/hotpotqa/processed/memory_stream.
```

- [x] **Step 10.3: Run quality gates**

```powershell
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
```

On this Windows host, run every `uv` command outside the Codex filesystem sandbox.

## 9. Validation and leakage checklist

Before implementation is considered complete:

- [x] Parser tests use the official parallel-array raw shape: `haystack_session_ids` / `haystack_dates` / `haystack_sessions`.
- [x] LongMemEval ranking records contain no `answer`, `has_answer`, `answer_session_ids`, `gold_*`, or label-derived booleans.
- [x] LongMemEval graph artifacts contain no gold labels.
- [x] Graph nodes use session-local `turn_index` as `sequence_index`; `global_position` stays in metadata and temporal request metadata.
- [x] Memory Stream no longer requires the HotpotQA first-1000 importance artifact.
- [x] `memory_stream` is absent from active HotpotQA/2Wiki experiment method lists.
- [x] LongMemEval `memory_stream` stage config uses dataset `longmemeval`.
- [ ] LongMemEval `memory_stream` run summary records relevance/recency/importance weights.
- [x] If importance weight is positive, every candidate has non-gold importance coverage.
- [x] Path metrics remain `N/A` for flat methods and Memory Stream.
- [x] Graph-aware path metrics are only reported if LongMemEval label records later define real `gold_dependency_edges`.
- [x] Reportable LongMemEval results use LongMemEval-specific turn/session metrics; evidence-suite columns are smoke-only.
- [ ] Repository-defined split results are not reported as official LongMemEval benchmark results unless split/tuning protocol is explicitly documented.

## 10. Review Questions

These are the main decisions to approve before implementation:

1. Candidate 粒度是否确定为 turn/message，而不是 session？
2. 第一阶段 Memory Stream 是否接受 `importance_weight=0.0`，只比较 dense relevance + order-based recency？
3. 是否删除 `configs/experiments/hotpotqa_memory_stream.json`，还是移到 retired docs？
4. 第一阶段是否允许 smoke 临时复用 evidence suite，但正式结果必须实现 LongMemEval-specific metric suite？
5. Dense-FT/R-GCN 是否明确放到第三阶段，不进入第一阶段验收范围？

我的建议是：

```text
1. turn/message
2. yes, importance_weight=0.0 for phase 1 and describe recency as order-based
3. delete active config or move to retired docs
4. smoke can temporarily reuse evidence suite; reportable results require LongMemEval suite
5. yes, trainable baselines are phase 3
```



