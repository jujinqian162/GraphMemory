# 2WikiMultiHopQA 适配计划

Date: 2026-06-20

Status: Draft plan. 本文档记录 2WikiMultiHopQA 作为下一个 evidence/path
数据集的适配边界和实施顺序，尚未表示代码已实现。

## 目标

把 2WikiMultiHopQA 接入当前 `graph_memory` 的 request-first 检索体系，用它补足
HotpotQA 缺少的 reasoning-path 监督：

- 保留当前 top-k evidence retrieval 主线。
- 继续支持 BM25、dense、Memory Stream、Graph Rerank、R-GCN 等方法输出
  `ranked_nodes`。
- 使用 2Wiki 的 `supporting_facts` 训练和评估 evidence node retrieval。
- 使用 2Wiki 的 `evidences` / `evidences_id` 在 label artifact 中构造
  `gold_dependency_edges`，让 `Edge Recall@10` 和 `Path Recall@10` 从 HotpotQA
  的 `N/A` 变成真实 path-level 指标。
- 保证 `gold_dependency_edges` 只用于训练标签和评估标签，不进入 test-time graph。

2Wiki 在本项目里的定位是 graph/path generalization benchmark，不是 long-term
memory benchmark。它没有 session、timestamp、recency、knowledge update 这类
LongMemEval V1 信息，因此不应被包装成 Memory Stream 的强论据。

## 已确认结论

### 1. 2Wiki 比 LongMemEval V2 更适合当前 top-k 检索评估

公开 LongMemEval V2 主要提供 answer-level label，retriever 只输出 top-k 时无法直接
计算 evidence Recall@k 或 Full Support@k。2Wiki 继承 HotpotQA 风格，提供
`supporting_facts`，可以直接映射到句子级 gold evidence nodes。

### 2. 2Wiki 比 HotpotQA 更适合 path 指标

HotpotQA 当前 artifact 的 `gold_dependency_edges` 为空，因此
`Path Recall@10` 和 `Edge Recall@10` 被固定为 `N/A`。2Wiki 额外提供
`evidences` / `evidences_id`，可以从实体关系三元组推导 evidence sentence 之间的
reasoning dependency label。

### 3. `gold_dependency_edges` 是 label，不是 graph input

这个边界必须保持严格：

```text
方法可见：
question + context sentences
  -> dataset projector
  -> GraphBuildRequest / TextRankingRequest / GraphRankingRequest
  -> visible graph
  -> retriever / reranker / R-GCN
  -> RankedResult

方法不可见：
supporting_facts + evidences + evidences_id + answer_id
  -> label artifact
  -> EvidenceLabel.gold_evidence_item_ids
  -> EvidenceLabel.gold_dependency_edges
  -> evaluator / train split supervision only
```

禁止把 `supporting_facts`、`evidences`、`evidences_id`、`answer_id` 或由它们直接
得到的 gold edge 写入 `GraphBuildRequest.input_visible_edges`、`*_graphs.json`、
`GraphRankingRequest.graph` 或 test-time R-GCN tensor。

### 4. Path Recall 必须基于 retrieved subgraph，不是只看 top-k 节点

弱定义：

```text
top-k 覆盖 gold path 的所有端点
```

这只能叫 dependency endpoint coverage，不能叫严格的 `Path Recall@k`。

本计划采用严格定义：

```text
Path Recall@k 检查 prediction.retrieved_subgraph 是否包含能够连接 gold
dependency path 的可见子图结构。
```

也就是说，evaluator 应同时读取：

- prediction 的 `ranked_nodes`
- prediction 的 `retrieved_subgraph.nodes`
- prediction 的 `retrieved_subgraph.edges`
- label 的 `gold_dependency_edges`

只有节点和结构都覆盖时才记 path 命中。

### 5. 不支持 path metric 的 retriever 继续输出 `N/A`

BM25、dense、Memory Stream 等 flat/top-k 方法如果没有返回 meaningful
`retrieved_subgraph.edges`，`Path Recall@10` 和 `Edge Recall@10` 继续记为
`N/A`。这不是失败，而是能力边界。

Graph Rerank、R-GCN 这类 graph-aware 方法会返回 top-k 在 visible graph 上的
诱导子图，因此可以参与 path metric。若 graph-aware 方法支持 path metric，但
retrieved subgraph 没有覆盖 gold path，则记 `0.0`，不是 `N/A`。

需要在输出契约里区分：

```text
unsupported method -> Path Recall@10 = N/A
supported method but missed path -> Path Recall@10 = 0.0
supported method and recovered path -> Path Recall@10 = 1.0
```

## 数据字段使用策略

2Wiki raw record 中的字段按可见性分为三类。

### Retrieval-visible fields

这些字段可以进入 ranking record、text candidate、graph node 或 visible graph：

| Field | 用途 |
|---|---|
| `_id` | 生成 `task_id = "2wiki_" + _id`。 |
| `question` | `query_text`。 |
| `context` | 构造 candidate sentence nodes。 |
| `type` | 可以进入 metadata，用于指标 breakdown；不得作为 gold answer/evidence 暗示。 |

`type` 本身是问题模板类型，不直接暴露哪个句子是证据；可以作为分析维度使用。

### Label-only fields

这些字段不能进入 test-time input 或 graph：

| Field | 用途 |
|---|---|
| `answer` | `gold_answer`；不进入 retrieval-visible text。 |
| `supporting_facts` | 构造 `gold_evidence_sentence_ids`。 |
| `evidences` | 构造 `gold_dependency_edges` 和 path metric label。 |
| `evidences_id` | 更稳健地对齐实体 relation path；label-only。 |
| `answer_id` | 辅助 path label 分析；label-only。 |

### First-version ignored fields

若 raw record 带有额外字段，第一版 parser 应 fail fast 或显式保存到
label-side/debug-side inspection artifact，不应默默流入 input-visible artifact。

## Artifact 设计

### 2Wiki ranking record

第一版使用与 HotpotQA 接近但 dataset-specific 的 record，而不是把 2Wiki 强行塞进
`HotpotQARankingRecord`。

```json
{
  "task_id": "2wiki_example-id",
  "question": "Which country ...?",
  "question_type": "compositional",
  "candidate_sentences": [
    {
      "sentence_id": "m0",
      "title": "Article A",
      "sentence_index": 0,
      "position": 0,
      "text": "Visible sentence text."
    }
  ],
  "metadata": {
    "dataset": "2wiki",
    "raw_id": "example-id"
  }
}
```

### 2Wiki label record

```json
{
  "task_id": "2wiki_example-id",
  "gold_answer": "France",
  "gold_evidence_sentence_ids": ["m3", "m8"],
  "gold_dependency_edges": [["m3", "m8"]],
  "metadata": {
    "question_type": "compositional",
    "path_label_source": "evidences_id"
  }
}
```

`gold_dependency_edges` 第一版仍保持当前 `EvidenceLabel` 能消费的
`list[list[str]]` 形状。relation/triple 细节不放进运行时 `EvidenceLabel`，只允许放进
label artifact 的 `metadata` 或单独 inspection artifact。

### Combined inspection artifact

可以继续写 combined artifact 方便人工检查，但 retrieval、graph construction、
training 和 evaluation 的正式入口必须使用分离的 input/label artifacts。

## Graph 构造边界

第一版 visible graph 只使用 input-visible 信息：

- `sequential`：同一 article 内相邻 sentence。
- `query_overlap`：query 和 sentence 的 lexical overlap。
- `entity_overlap`：由 text/title 中可见实体或词项构造。
- `bridge`：由当前已有 bridge rule 根据可见 text/title 推断。

第一版不新增 R-GCN relation vocab。原因是当前 R-GCN checkpoint、tensorizer、
ablation 和 config 默认 edge type 集合都是：

```text
sequential
query_overlap
entity_overlap
bridge
```

2Wiki 的 `evidences` 不进入 visible graph。它只生成 label-side
`gold_dependency_edges`，用于训练监督和 path metric。

后续若要加入 relation-aware visible edges，必须满足：

1. edge 来自所有候选文本的无监督/公开可见抽取，而不是 `evidences` gold annotation。
2. train/dev/test 构造规则一致。
3. 新 edge type 同步更新 contract、tensorizer、R-GCN relation vocab、ablation 配置和文档。

## Path metric 语义

### Edge Recall@10

只对 path-metric-supported methods 计算。

定义：

```text
gold_dependency_edges = label 中的 gold edge 集合
retrieved_edges = prediction.retrieved_subgraph.edges

Edge Recall@10 =
  covered_gold_edges / total_gold_edges
```

`covered_gold_edges` 的第一版判定规则：

1. gold edge 两端节点都必须在 `retrieved_subgraph.nodes` 中。
2. `retrieved_subgraph.edges` 中必须存在一条连接这两个节点的 visible edge。
3. 若 visible edge 是 directed，则方向必须与 gold edge 一致。
4. 若 visible edge 是 undirected，则任一方向均可覆盖。

如果某个 task 没有可构造的 `gold_dependency_edges`，该 task 不进入
`Edge Recall@10` 分母；若整个评估文件没有可评 task，则该 method 的
`Edge Recall@10 = N/A`。

### Path Recall@10

只对 path-metric-supported methods 计算。

定义：

```text
Path Recall@10 = 1.0
  if every gold dependency segment is reachable inside retrieved_subgraph
  else 0.0
```

判定规则：

1. gold path 涉及的所有节点必须出现在 `retrieved_subgraph.nodes`。
2. 对每条 gold dependency edge `(source, target)`，在 `retrieved_subgraph.edges`
   形成的图中必须存在从 `source` 到 `target` 的路径。
3. traversal 尊重 visible edge 的 `directed` 字段；undirected edge 可双向走。
4. 多条 gold dependency edge 必须全部满足。

这一定义比 `Edge Recall@10` 稍宽，因为它允许 retrieved subgraph 用多跳 visible
path 覆盖一个 gold dependency segment。

### Comparison 类型

2Wiki 的 `comparison` 样本可能是两个并行事实比较，不一定存在自然的线性 dependency
edge。第一版不要强行造假 path。策略：

- 能从 `evidences` / `evidences_id` 映射出有序 dependency chain 的样本参与 path metric。
- 不能形成 dependency edge 的样本只参与 node-level metrics 和 connectivity metrics。
- 在 run summary 中记录 path-supported task count。

## 文件责任划分

### 新增 dataset package

| File | Responsibility |
|---|---|
| `graph_memory/datasets/twowiki/records.py` | 定义 2Wiki raw dataclass、ranking record、label record、conversion result。 |
| `graph_memory/datasets/twowiki/parser.py` | 解析 raw 2Wiki JSON，严格校验 `_id/question/answer/context/supporting_facts/evidences/type`。 |
| `graph_memory/datasets/twowiki/converter.py` | flatten context sentences，映射 supporting facts 到 node ids，构造 label-side dependency edges。 |
| `graph_memory/datasets/twowiki/projectors.py` | 投影到 `TextRankingRequest`、`GraphBuildRequest`、`GraphRankingRequest`、`TemporalMemoryRankingRequest`、`EvidenceEvaluationRequest`。 |
| `graph_memory/datasets/twowiki/__init__.py` | 导出 package public API。 |

### 新增 / 修改脚本

| File | Responsibility |
|---|---|
| `scripts/prepare_2wiki.py` | 从 raw 2Wiki JSON 写出 `.input.json`、`.labels.json`、`.combined.json` 和 run summary。 |
| `scripts/build_graphs.py` | 增加 `--dataset hotpotqa|twowiki`，默认保持 `hotpotqa`，按 dataset 选择 projector。 |
| `scripts/build_train_pairs.py` | 增加 dataset-aware label/request 投影；不再假设输入一定是 `HotpotQARankingRecord`。 |
| `scripts/run_retrieval.py` | 增加 dataset-aware request assembly；保持 method 运行时只看 request。 |
| `scripts/evaluate_retrieval.py` | 增加 dataset-aware evaluation request projection。 |
| `scripts/train_method.py` | R-GCN 和 Dense-FT 训练 payload 使用 dataset-specific projector，而不是直接 cast HotpotQA record。 |

第一版可以先完成 direct script smoke，再接入 `scripts/experiment.py` workflow。避免一次性同时改
dataset、workflow manifest、config registry 和指标语义。

### 修改 evaluation

| File | Responsibility |
|---|---|
| `graph_memory/evaluation/path_metrics.py` | 新增 `edge_recall_at`、`path_recall_at`，只消费 `retrieved_subgraph` 和 `gold_dependency_edges`。 |
| `graph_memory/evaluation/suites.py` | 当 labels 有 dependency edges 且 method 支持 path metric 时输出真实 path values；否则输出 `N/A`。 |
| `graph_memory/contracts/ranking.py` | 保留现有 `retrieved_subgraph`；如需区分 unsupported/missed，增加 `metadata.path_metrics_supported`。 |
| `graph_memory/validation/ranking.py` | 校验 `retrieved_subgraph` 结构；如新增 metadata flag，校验其为 boolean。 |
| `graph_memory/validation/metrics.py` | 允许 `Path Recall@10` / `Edge Recall@10` 为 number 或 `N/A`。 |

### 测试文件

| File | Responsibility |
|---|---|
| `tests/test_twowiki_parser.py` | raw field parsing、invalid example fail-fast。 |
| `tests/test_twowiki_converter.py` | supporting facts -> gold node ids；evidences -> label-only dependency edges。 |
| `tests/test_twowiki_projectors.py` | 2Wiki records 投影到所有 consumer requests。 |
| `tests/test_twowiki_prepare_cli.py` | tiny raw fixture 生成分离 input/label/combined artifact。 |
| `tests/test_path_metrics.py` | retrieved_subgraph 上的 edge/path recall 严格语义。 |
| `tests/test_twowiki_leakage_boundaries.py` | 禁止 gold 字段流入 input graph。 |

## 实施顺序

### Task 1：建立 2Wiki raw parser 和 record 类型

目标：让 raw 2Wiki JSON 被解析为 dataset-specific dataclass，不复用
HotpotQA parser。

验收：

- 能解析 tiny 2Wiki fixture。
- 缺少 `_id/question/context/supporting_facts/evidences/type` 时 fail fast。
- `context` 中空文档或非字符串 sentence 报错。
- parser 不丢弃 `type`、`evidences`、`evidences_id` 等 2Wiki 字段。

### Task 2：实现 converter，生成分离 input/label artifacts

目标：把 2Wiki raw example 转成 ranking record 和 label record。

验收：

- `task_id = "2wiki_" + raw_id`。
- candidate sentence ids 使用 `m{position}`，position 全局递增。
- `supporting_facts` 按 `(title, sentence_index)` 映射到
  `gold_evidence_sentence_ids`。
- `evidences` / `evidences_id` 只生成 label-side `gold_dependency_edges`。
- ranking record 中不存在 `gold_answer`、`supporting_facts`、`evidences`、
  `gold_dependency_edges`、`is_gold*`。

### Task 3：实现 2Wiki projectors

目标：保持 request-first 结构：

```text
2Wiki ranking/label record
  -> TwoWikiToTextRankingRequest
  -> TwoWikiToGraphBuildRequest
  -> TwoWikiToGraphRankingRequest
  -> TwoWikiToTemporalMemoryRankingRequest
  -> TwoWikiToEvidenceEvaluationRequest
```

验收：

- retriever method 不 import `graph_memory.datasets.twowiki`。
- graph builder 只接收 `GraphBuildRequest`。
- evaluator 只接收 `EvidenceEvaluationRequest`。
- 2Wiki projector 的字段不污染 HotpotQA projector。

### Task 4：实现 `scripts/prepare_2wiki.py`

目标：提供与 `scripts/prepare_hotpotqa.py` 对称的 preparation CLI。

验收：

```powershell
uv run python scripts/prepare_2wiki.py `
  --input data/2wiki/raw/dev.json `
  --output_input runs/2wiki_tiny/inputs/test.input.json `
  --output_labels runs/2wiki_tiny/inputs/test.labels.json `
  --output_combined runs/2wiki_tiny/inputs/test.combined.json `
  --max_examples 5 `
  --seed 13 `
  --offset 0
```

- 输出 input/label/combined 三个文件。
- run summary 记录 raw path、输出 path、valid/invalid count、path-supported count。
- invalid example 默认跳过并计数；`--strict_invalid_examples` 在第一个 invalid
  example 处失败。

### Task 5：让现有 graph/retrieval/train/evaluate scripts 支持 dataset selector

目标：保留 HotpotQA 默认行为，同时允许 `--dataset twowiki`。

验收：

- `scripts/build_graphs.py --dataset hotpotqa` 与旧行为一致。
- `scripts/build_graphs.py --dataset twowiki` 使用 `TwoWikiToGraphBuildRequest`。
- `scripts/run_retrieval.py --dataset twowiki` 能跑 BM25/dense/graph rerank。
- `scripts/build_train_pairs.py --dataset twowiki` 能用 2Wiki gold nodes 生成 train pairs。
- `scripts/train_method.py --dataset twowiki` 能构造 R-GCN/Dense-FT train payload。
- `scripts/evaluate_retrieval.py --dataset twowiki` 能读取 2Wiki labels。

### Task 6：实现严格 path metrics

目标：把 `Path Recall@10` / `Edge Recall@10` 从 HotpotQA-only `N/A` 变为
dependency-aware evaluator。

验收：

- Flat method 若标记不支持 path metric，则 `Path Recall@10 = N/A`，
  `Edge Recall@10 = N/A`。
- Graph-aware method 支持 path metric；若 retrieved subgraph 缺边，输出 `0.0`。
- Graph-aware method 的 retrieved subgraph 中存在可见路径时，`Path Recall@10 = 1.0`。
- `Edge Recall@10` 要求 direct visible edge 覆盖 gold edge。
- `Path Recall@10` 允许多跳 visible path 覆盖 gold dependency segment。
- 只有 label 中有 non-empty `gold_dependency_edges` 的 task 进入 path metric 分母。

### Task 7：tiny end-to-end smoke

目标：在 tiny fixture 上跑通最小闭环。

建议 smoke 方法：

```text
bm25
dense
dense_graph_rerank
dense_rgcn_graph_retriever
```

验收：

- prepare -> graphs -> retrieve -> evaluate -> aggregate 在 tiny fixture 上可运行。
- BM25/dense 的 path metrics 为 `N/A`。
- dense_graph_rerank / R-GCN 的 path metrics 为 number 或 `N/A`，取决于该 method
  是否声明 path support；一旦声明 support，missed path 必须是 `0.0`。
- `main_results.csv`、`path_results.csv`、`efficiency_results.csv` 均能生成。

### Task 8：接入 named experiment workflow

目标：tiny direct scripts 稳定后，再扩展 `scripts/experiment.py` 相关 workflow。

验收：

- experiment config 能声明 dataset 为 `twowiki`。
- manifest 中记录 raw paths、processed input/label paths、dataset id。
- stage configs 把 dataset id 传给 prepare/build/retrieve/evaluate/train scripts。
- HotpotQA 现有 profile 和方法矩阵不变。

## 指标表策略

主表继续保持现有 evidence retrieval 指标：

- `Recall@2`
- `Recall@5`
- `Recall@10`
- `Evidence F1@5`
- `Evidence F1@10`
- `Full Support@5`
- `Full Support@10`
- `MRR`

path 表保持现有列：

- `Connected Evidence Recall@5`
- `Connected Evidence Recall@10`
- `Query-Evidence Connectivity@10`
- `Path Recall@10`
- `Edge Recall@10`

2Wiki 新增 breakdown 建议另写分析表或 failure-case summary，不塞进现有主表：

- by `question_type`
- by path-supported / path-unsupported
- by number of gold evidence nodes
- by number of gold dependency edges

## R-GCN 适配策略

R-GCN 可以用 2Wiki 训练，因为 2Wiki 有 `gold_evidence_sentence_ids`。

第一版训练目标不改变：

```text
gold evidence node -> label 1
sampled negative node -> label 0
BCEWithLogitsLoss
```

`gold_dependency_edges` 第一版只用于 path-level evaluation，不直接改 R-GCN loss。
原因：

1. 当前 R-GCN 是 node scorer，不是 edge/path scorer。
2. 直接把 gold dependency edge 加进 graph 会造成 test-time leakage。
3. path-aware loss 需要新的 pair/path training contract，应作为第二阶段设计。

第一版 R-GCN 对 2Wiki 的价值是：

- 检验 learned graph encoder 是否比 dense / graph rerank 更能找全 multi-hop evidence。
- 在 graph-aware retrieved subgraph 上报告严格 path metric。
- 做 edge-type ablation 时观察 path metric 是否比 node recall 更敏感。

## Memory Stream 和 GraphRAG 边界

### Memory Stream

2Wiki 没有 session/time 信息。Memory Stream 可以作为普通 top-k baseline 跑，但它的
recency 只能来自 synthetic position，论证力弱。

第一版策略：

- 可以跑 `memory_stream`，但不要把 2Wiki 作为 Memory Stream 主论据。
- `Path Recall@10` / `Edge Recall@10` 对 Memory Stream 保持 `N/A`，除非之后让它显式
  返回基于 visible graph 的 retrieved subgraph edges。

### GraphRAG

GraphRAG baseline 若能输出 per-task ranked sentence nodes 和 retrieved subgraph，
可以参与 2Wiki path metrics。若只输出 LLM context 或答案，则不放入本计划第一版。

第一版不要为了 GraphRAG 改变 artifact contract。GraphRAG 适配仍必须满足：

```text
query + candidate sentences -> ranked node ids + retrieved_subgraph
```

## 验证命令

Windows 主机上 `uv` 和 pytest 应按 AGENTS.md 要求在 sandbox 外运行。

计划实施时建议的验证顺序：

```powershell
uv run pytest tests/test_twowiki_parser.py -q
uv run pytest tests/test_twowiki_converter.py -q
uv run pytest tests/test_twowiki_projectors.py -q
uv run pytest tests/test_path_metrics.py -q
uv run pytest tests/test_twowiki_prepare_cli.py -q
uv run pytest tests/test_twowiki_leakage_boundaries.py -q
```

完成 workflow 接入后再跑：

```powershell
uv run python scripts/experiment.py plan 2wiki_tiny --config configs/experiments/2wiki_tiny.json
uv run python scripts/experiment.py run 2wiki_tiny --config configs/experiments/2wiki_tiny.json --method bm25 --method dense_graph_rerank
uv run python scripts/experiment.py status 2wiki_tiny
```

## 接受标准

这个适配可以认为达到第一版完成，当且仅当：

1. 2Wiki raw -> separated input/label artifacts 可复现。
2. input-visible artifacts 和 graph artifacts 不包含 label-only fields。
3. BM25/dense 能在 2Wiki 上产生 node-level retrieval metrics。
4. dense_graph_rerank 至少能在 2Wiki 上产生 graph-aware `retrieved_subgraph`。
5. R-GCN 能用 2Wiki train/dev labels 训练，并在 test split 输出 ranked results。
6. `Path Recall@10` / `Edge Recall@10` 的 `N/A`、`0.0`、`1.0` 语义由测试覆盖。
7. `path_results.csv` 中 graph-aware methods 可以出现真实 path values。
8. HotpotQA 现有 workflow 和指标行为保持不变。

## 风险和决策点

### Dependency edge 构造可能不唯一

`evidences` 是实体关系三元组，不是直接的 sentence-to-sentence edge。需要把 triple
映射回 supporting sentences。若多个 supporting sentence 都包含同一实体或关系，第一版
应采用确定性规则并在 metadata 中记录 mapping ambiguity count。

### Comparison 样本不一定有线性 path

不要为了提高 path-supported count 人工制造 comparison edge。没有自然 dependency edge
的样本继续用于 node metrics，不进入 path metric 分母。

### Path metric 和 connectivity metric 不应混淆

`Connected Evidence Recall@10` 可以继续用统一参考 graph 判断 top-k gold nodes 是否
连通。`Path Recall@10` 应使用 method 返回的 `retrieved_subgraph`，并检查
gold dependency path 的结构覆盖。

### Dataset selector 不应退化成新一轮 generic wrapper

适配方式必须保持当前 request-first 设计：

```text
dataset-specific record -> dataset-specific projector -> consumer request
```

不要新增 `EvidenceRankingView`、`MemoryTaskInput` 风格的跨数据集中间大对象。

### 第一版不要新增 edge type

2Wiki 的主要价值先通过 labels 和 path metrics 体现。relation-aware visible edge 是
后续增强，不应阻塞第一版 adapter。
