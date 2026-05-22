# Phase 1 Real Graph Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real Phase 1 evidence-tracing experiment system for HotpotQA: strong BM25 and frozen dense baselines, typed evidence graph construction, graph-aware reranking, dev-set parameter tuning, and leakage-safe evaluation.

**Architecture:** Convert labeled HotpotQA distractor examples into sentence-level memory tasks, build a typed weighted graph per task using only input-visible text/title/position features, run flat retrievers and graph rerankers under one ranked-result schema, tune graph weights on dev, and report final metrics on test. The system must separate model inputs from label-only evaluation data so answer text and supporting-fact labels cannot leak into retrieval.

**Tech Stack:** Python 3.12, `pytest`, `numpy`, `rank-bm25`, `sentence-transformers`, optional `spacy` for NER, JSONL/JSON artifacts, CSV result tables.

---

## Rethink And Design Review

The current repository is a minimal scaffold, not an implemented graph-memory pipeline. The real Phase 1 should first create the package, script, config, and test structure, then implement the scientific boundary below without pulling in later-phase systems.

The Phase 1 claim should be narrow and testable:

> Given a HotpotQA question and its candidate distractor sentences, graph-aware memory retrieval should recover the complete supporting evidence set and a connected retrieved evidence subgraph better than flat BM25 or frozen dense retrieval.

Design conclusions after review:

- Keep Phase 1 retrieval-only. Do not add answer generation, Dense-FT, GNNs, GraphRAG, Memory Stream, or MemGPT-style memory to this implementation.
- Use real frozen dense encoders, such as E5 or BGE, instead of hashed vectors.
- Tune graph rerank parameters on dev only. Report test results using fixed dev-selected parameters.
- Treat HotpotQA `supporting_facts` as labels only. They map `(title, sentence_id)` to gold evidence node ids but must not enter graph construction or retrieval scoring.
- Evaluate graph connectivity fairly: all methods select nodes; connectivity can be computed on the same constructed graph for each method's selected top-k nodes.
- Emit complete per-task rankings in `ranked_nodes`; use `top_k` only as the metric cutoff and retrieved-subgraph cutoff.
- Keep graph construction lexical/entity-based and deterministic in Phase 1. A trainable graph retriever belongs to a later phase.

## File Structure

The current repository is a minimal root project with only `pyproject.toml`, `main.py`, `README.md`, and `docs/`. Implement Phase 1 from the repository root. Create the Python package, scripts, configs, and tests directly under this root so commands are runnable from `E:\College\AdviserProject\EPGM\graph_memory`.

Preserve script names from the experiment plan. The leakage-safe implementation may write separate input and label artifacts, but it must also provide compatibility outputs or aliases documented below so the original command surface remains usable.

- Modify: `pyproject.toml`
  - Add runtime dependencies: `numpy`, `rank-bm25`, `sentence-transformers`, `tqdm`.
  - Add optional dependency group for NER: `spacy`.
- Create: `graph_memory/__init__.py`
- Create: `graph_memory/types.py`
  - Define shared aliases, `TypedDict`s, lightweight dataclasses, and behavior protocols used across the package.
- Create: `graph_memory/validation.py`
  - Implement fail-fast artifact validators and `ContractValidationError`.
- Create: `graph_memory/io.py`
  - Provide deterministic JSON/CSV/config read-write helpers.
- Create: `graph_memory/hotpotqa.py`
  - Convert HotpotQA examples into input tasks and label records.
  - Preserve `title`, `sentence_id`, and global `position` for every sentence.
- Create: `graph_memory/splits.py`
  - Deterministic train/dev/test sampling from labeled HotpotQA train/dev files.
- Create: `graph_memory/text.py`
  - Normalize text, filter stopwords, compute content tokens, extract title aliases, and support IDF-weighted lexical scoring.
- Create: `graph_memory/entities.py`
  - Extract entities using deterministic heuristics and optional spaCy NER.
- Create: `graph_memory/graphs.py`
  - Build typed weighted graph edges: sequential, query_overlap, entity_overlap, bridge.
- Create: `graph_memory/indexes/__init__.py`
- Create: `graph_memory/indexes/bm25.py`
  - Provide a BM25 retriever over per-task memory sentences.
- Create: `graph_memory/indexes/dense.py`
  - Provide frozen dense encoding and cosine retrieval using Sentence-Transformers.
- Create: `graph_memory/retrieval.py`
  - Dispatch BM25, dense, BM25-seeded graph rerank, and dense-seeded graph rerank.
- Create: `graph_memory/rerank.py`
  - Implement typed multi-hop graph reranking and retrieved subgraph extraction.
- Create: `graph_memory/tuning.py`
  - Grid-search graph rerank parameters on dev and save selected config.
- Create: `graph_memory/evaluation.py`
  - Compute node metrics, graph connectivity metrics, and efficiency metrics.
- Create: `graph_memory/observability.py`
  - Write run summaries and compact graph/debug summaries at script boundaries.
- Create: `scripts/prepare_hotpotqa.py`
  - Emit input and label files, with an optional combined compatibility file and leakage-check report.
- Create: `scripts/build_graphs.py`
  - Build graphs from input-visible task fields only.
- Create: `scripts/run_retrieval.py`
  - Accept method, encoder model, graph config, and output path.
- Create: `scripts/tune_graph_rerank.py`
  - Select graph rerank parameters from dev metrics.
- Create: `scripts/evaluate_retrieval.py`
  - Read predictions, labels, graphs, and produce per-method CSVs.
- Create: `scripts/aggregate_tables.py`
  - Merge per-method metrics into Phase 1 tables.
- Create: `tests/test_phase1_real_data_structures.py`
  - Test HotpotQA conversion, label mapping, and leakage separation.
- Create: `tests/test_phase1_real_graphs.py`
  - Test edge construction and stopword-safe scoring.
- Create: `tests/test_phase1_real_retrieval.py`
  - Test BM25, dense interface, graph rerank, and parameter config behavior.
- Create: `tests/test_phase1_real_evaluation.py`
  - Test Recall, Evidence F1, Full Support, Connected Evidence Recall, and fair graph connectivity.
- Optional create: `tests/test_phase1_real_validation.py`
  - Test validators directly when validation coverage becomes too large for the four main test files.

## Data Contracts

Use input and label files as separate artifacts when running full experiments. This intentionally strengthens the original combined `*_memory_tasks.json` schema from the student experiment plan by preventing labels from being passed to retrieval or graph construction.

Compatibility rule:

- `*_memory_tasks.input.json` is the only file accepted by graph construction and retrieval.
- `*_memory_tasks.labels.json` is the only file accepted by evaluation and dev tuning for gold labels.
- `scripts/prepare_hotpotqa.py` may also write `*_memory_tasks.json` as a compatibility artifact for readers of the original plan, but retrieval code must not consume gold fields from that file.
- `docs/40-operations/commands.md` must show the leakage-safe command path and note the original-plan compatibility artifacts where relevant.

`*_memory_tasks.input.json`:

```json
[
  {
    "task_id": "hotpot_000001",
    "query": "question text",
    "memory_items": [
      {
        "id": "m0",
        "node_type": "document_sentence",
        "text": "sentence text",
        "source": "Document_Title",
        "sentence_id": 0,
        "position": 0
      }
    ]
  }
]
```

`*_memory_tasks.labels.json`:

```json
[
  {
    "task_id": "hotpot_000001",
    "gold_answer": "answer text",
    "gold_evidence_nodes": ["m1", "m7"],
    "gold_dependency_edges": []
  }
]
```

`*_graphs.json`:

```json
[
  {
    "task_id": "hotpot_000001",
    "nodes": [
      {"id": "q", "node_type": "question", "text": "question text"},
      {"id": "m0", "node_type": "document_sentence", "text": "sentence text", "source": "Document_Title", "sentence_id": 0, "position": 0}
    ],
    "edges": [
      {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 2.5, "directed": true},
      {"source": "m0", "target": "m1", "edge_type": "sequential", "weight": 1.0, "directed": false}
    ]
  }
]
```

`ranked_results_{method}.json`:

```json
[
  {
    "task_id": "hotpot_000001",
    "method": "dense_graph_rerank",
    "ranked_nodes": [{"node_id": "m7", "score": 0.913}],
    "retrieved_subgraph": {
      "nodes": ["m7"],
      "edges": []
    },
    "latency_ms": 14.2,
    "input_tokens": 640
  }
]
```

Ranking semantics:

```text
ranked_nodes must contain the complete sorted ranking of all memory nodes for the task.
--top_k controls metric cutoffs and retrieved_subgraph extraction only.
MRR is computed from the complete ranking. If a future run truncates ranked_nodes,
the metric must be reported explicitly as MRR@k instead of MRR.
```

## Core Algorithms

### HotpotQA Conversion

- Require each raw HotpotQA example to contain `_id`.
- Generate stable task IDs from raw IDs, not from sampled position:

```text
task_id = "hotpot_" + raw_example["_id"]
```

- Do not renumber task IDs when the split seed, offset, or count changes.
- If a raw example lacks `_id`, raise a clear validation/conversion error instead of inventing a position-based ID.
- Iterate through `context` in the raw example order.
- For each `(title, sentences)` pair, assign `sentence_id` from the sentence list index.
- Assign `position` from the flattened memory item index.
- Maintain `title_sentence_to_node_id[(title, sentence_id)] = node_id`.
- Convert each `supporting_facts` pair into a `gold_evidence_nodes` entry through that map.
- Write answer and gold labels only to label artifacts.

### Labeled Split Policy

Use only labeled HotpotQA splits. Do not use the official unlabeled test set for metric computation.

Recommended Phase 1 split:

```text
train: 5,000 examples sampled from labeled HotpotQA train
dev:     500 examples sampled from labeled HotpotQA dev, offset 0
test:  1,000 examples sampled from labeled HotpotQA dev, offset 500
```

The dev and test subsets must be disjoint. If the labeled dev file has too few examples for `offset + count`, fail loudly rather than silently overlapping examples or falling back to another file.

### Stopword-Safe Lexical Scoring

Content token processing:

```text
lowercase
remove punctuation
drop stopwords such as the, a, an, and, of, in, on, for, to, with
drop tokens with length <= 2 unless they are part of a title alias
optionally lemmatize through spaCy when spaCy is enabled
```

Lexical score:

```text
lexical_score(query, title_sentence)
= sum(idf[token] for token in shared_content_tokens)
  + 1.5 * title_alias_match_count
  + 2.0 * named_entity_overlap_count
```

### Typed Graph Edges

Sequential:

```text
source == target document title
abs(sentence_id_i - sentence_id_j) == 1
edge_type = sequential
weight = 1.0
```

Query overlap:

```text
source = q
target = memory node
score = lexical_score(query, title + sentence)
keep top 20 positive-scoring targets
edge_type = query_overlap
directed = true
```

Entity overlap:

```text
source = memory node
target = memory node
same or different document
score = shared entity / title alias / content phrase score
keep top 10 neighbors per node
edge_type = entity_overlap
directed = false
```

Bridge:

```text
source = memory node
target = memory node
source documents differ
score rewards title mention across documents, shared named entity, and query entity coverage
keep top 50 per task
edge_type = bridge
directed = false
```

### Flat Retrieval

BM25:

```text
For each task:
  corpus = [item.source + ". " + item.text for item in memory_items]
  query = task.query
  score every memory item with BM25
  return complete ranked_nodes sorted by score descending
```

Frozen dense:

```text
encoder = SentenceTransformer(model_name)
query_embedding = encoder.encode([query_prefix + query], normalize_embeddings=True)
passage_embeddings = encoder.encode([passage_prefix + source + ". " + text for each item], normalize_embeddings=True)
score = query_embedding dot passage_embedding
return complete ranked_nodes sorted by score descending
```

Default dense prefix config is tied to the default E5 encoder:

```text
dense_encoder = intfloat/e5-base-v2
query_prefix = "query: "
passage_prefix = "passage: "
```

If the encoder changes, update the prefixes according to that encoder's retrieval instructions rather than treating E5/BGE prefixes as interchangeable.

### Graph Rerank

Rerank must consume a flat initial score vector and a graph.

```text
S_final(v)
= lambda_init * S_init(v)
 + lambda_query * S_query(v)
 + lambda_neighbor * S_neighbor(v)
 + lambda_bridge * S_bridge(v)
 + lambda_path * S_path(v)
```

Definitions:

```text
S_init(v): normalized BM25 or dense score.
S_query(v): per-task normalized weight of q -> v query_overlap edge.
S_neighbor(v): degree-normalized weighted average over neighbors u using S_init(u), edge_weight(u, v), and type_weight(edge_type).
S_bridge(v): per-task normalized bridge-specific score from high-scoring cross-document neighbors.
S_path(v): bonus when v helps connect two high-scoring seed components through entity_overlap or bridge edges.
```

`S_query`, `S_neighbor`, and `S_bridge` must be normalized per task before weighted combination so raw graph edge magnitudes cannot dominate `S_init`. `S_neighbor` must divide by the total incoming weighted edge mass for the target node so high-degree clusters do not receive a score simply because they have more edges.

For HotpotQA-only Phase 1, keep `lambda_path = 0.0` because HotpotQA does not provide gold dependency paths. `S_path` is reserved for later 2Wiki/tool-trajectory experiments unless it is implemented as a fully unsupervised, leakage-safe structural bonus with its own tests.

Default edge type weights:

```json
{
  "query_overlap": 0.8,
  "sequential": 0.3,
  "entity_overlap": 0.7,
  "bridge": 1.0
}
```

Start with a two-hop expansion budget:

```text
seed_top_s = 30
max_hops = 2
candidate_nodes = seeds plus graph neighbors within max_hops
rank all original memory nodes, but only graph bonuses are nonzero inside candidate_nodes
```

## Parameter Tuning

Use grid search on dev only.

Simplicity-first tuning policy:

- Phase 1 should prioritize debuggability over tuning speed.
- Do not introduce a persistent score cache in the first implementation.
- `graph_rerank(initial_scores, graph, config)` remains the lightweight boundary that makes future score reuse possible without changing the rerank formula.
- It is acceptable for dev grid search to recompute BM25 or dense initial rankings while the pipeline is still being debugged.
- For quick debugging, use a smaller dev artifact produced through the normal split/conversion path; do not report debug-subset tuning as the official Phase 1 config.
- If full-dev tuning becomes a practical blocker after the pipeline is correct, add score-artifact reuse as a later optimization with its own validation and run summary fields.

Candidate values:

```json
{
  "lambda_init": [1.0],
  "lambda_query": [0.0, 0.05, 0.1, 0.2],
  "lambda_neighbor": [0.0, 0.05, 0.1, 0.2, 0.4],
  "lambda_bridge": [0.0, 0.05, 0.1, 0.2],
  "lambda_path": [0.0],
  "seed_top_s": [20, 30],
  "max_hops": [1, 2]
}
```

The all-zero graph-lambda candidate (`lambda_query = lambda_neighbor = lambda_bridge = 0.0`) is an intentional pure initial-score fallback. Tuning may select it when graph features hurt dev-set retrieval.

Selection objective:

```text
0.50 * Full Support@5
+ 0.30 * Recall@5
+ 0.20 * Connected Evidence Recall@10
```

Tie-breakers:

```text
1. higher Full Support@10
2. lower retrieval latency
3. smaller average retrieved subgraph edge count
```

Save selected parameters to:

```text
configs/phase1_graph_rerank_dev_selected.json
```

## Metrics

Node metrics:

```text
Recall@2, Recall@5, Recall@10
Evidence F1@5, Evidence F1@10
Full Support@5, Full Support@10
MRR
```

Graph/path metrics:

```text
Connected Evidence Recall@5
Connected Evidence Recall@10
Query-Evidence Connectivity@10
Path Recall@10 = N/A for HotpotQA-only Phase 1
Edge Recall@10 = N/A for HotpotQA-only Phase 1
```

For HotpotQA Phase 1, do not report `Path Recall@10` as a gold reasoning-path metric because HotpotQA does not provide explicit dependency paths. Report `Query-Evidence Connectivity@10` instead:

```text
Query-Evidence Connectivity@10:
  selected_nodes = top-10 ranked memory node IDs.
  If any gold evidence node is missing from selected_nodes, score 0.
  Otherwise build the induced graph over {"q"} union selected_nodes.
  Directed edges are traversed source -> target; undirected edges are traversed both ways.
  Score 1 only if every gold evidence node is reachable from q in that induced graph.
  Otherwise score 0.
```

`Path Recall@10` and `Edge Recall@10` should remain in the table schema as `N/A` for HotpotQA-only Phase 1, then become real metrics when 2Wiki/tool-trajectory data with gold dependency edges is added.

Efficiency metrics:

```text
Index Build Time
Graph Construction Time
Retrieval Latency / Query
Memory Size
Avg Retrieved Nodes
Avg Retrieved Edges
```

## Tasks

### Task 0: Contract, I/O, And Observability Foundation

**Files:**
- Create: `graph_memory/types.py`
- Create: `graph_memory/validation.py`
- Create: `graph_memory/io.py`
- Create: `graph_memory/observability.py`
- Test: existing Phase 1 test files; split to `tests/test_phase1_real_validation.py` only if needed.

- [ ] **Step 1: Define shared data shapes and configs**

Create aliases, `TypedDict`s, and frozen config dataclasses for the contracts already defined in `docs/20-contracts/phase1-data-contracts.md`.

- [ ] **Step 2: Implement fail-fast validators**

At minimum:

```python
class ContractValidationError(ValueError):
    ...

def validate_memory_task_inputs(records: object) -> None:
    ...

def validate_memory_task_labels(records: object, inputs_by_task_id: Mapping[TaskId, MemoryTaskInput]) -> None:
    ...

def validate_graphs(graphs: object, inputs_by_task_id: Mapping[TaskId, MemoryTaskInput]) -> None:
    ...

def validate_ranked_results(predictions: object, inputs_by_task_id: Mapping[TaskId, MemoryTaskInput]) -> None:
    ...
```

Validators must not repair, sort, drop, or infer data.

- [ ] **Step 3: Implement boring artifact I/O helpers**

Use UTF-8, deterministic JSON formatting, and explicit CSV column order.

- [ ] **Step 4: Implement run summary helpers**

Every CLI script should be able to write a compact run summary with effective config, paths, counts, timings, and notes.

- [ ] **Step 5: Test the critical negative cases**

Cover label leakage in input artifacts, missing graph endpoints, duplicate ranked nodes, task ID mismatches, and non-finite scores.

### Task 1: Data Conversion And Leakage Separation

**Files:**
- Create: `graph_memory/hotpotqa.py`
- Create: `graph_memory/splits.py`
- Create: `scripts/prepare_hotpotqa.py`
- Test: `tests/test_phase1_real_data_structures.py`

- [ ] **Step 1: Write tests for title/sentence label mapping**

Test fixture:

```python
def test_supporting_facts_map_title_sentence_to_node_ids():
    raw = [{
        "_id": "ex1",
        "question": "Where is the Eiffel Tower and what river runs through that city?",
        "answer": "Paris and the Seine",
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is in Paris.", "It opened in 1889."]],
            ["Paris", ["Paris is in France.", "The Seine runs through Paris."]],
        ],
        "supporting_facts": [["Eiffel Tower", 0], ["Paris", 1]],
    }]
    parsed_examples = parse_hotpotqa_examples(raw)
    conversion = convert_hotpotqa_examples(parsed_examples)
    inputs = conversion.task_inputs
    labels = conversion.task_labels
    assert inputs[0]["memory_items"][0]["sentence_id"] == 0
    assert inputs[0]["memory_items"][0]["position"] == 0
    assert inputs[0]["memory_items"][3]["sentence_id"] == 1
    assert inputs[0]["memory_items"][3]["position"] == 3
    assert labels[0]["gold_evidence_nodes"] == ["m0", "m3"]
    assert "gold_answer" not in inputs[0]
    assert "gold_evidence_nodes" not in inputs[0]
```

- [ ] **Step 2: Run the new conversion test and confirm it fails**

Run:

```powershell
uv run pytest tests/test_phase1_real_data_structures.py -q
```

Expected: FAIL because the current repository has no converter yet.

- [ ] **Step 3: Implement parser and converter returning named domain records**

Required public signatures:

```python
@dataclass(frozen=True)
class HotpotQAConversionResult:
    task_inputs: list[MemoryTaskInput]
    task_labels: list[MemoryTaskLabels]


def parse_hotpotqa_examples(raw_records: Sequence[object]) -> list[HotpotQAExample]:
    ...


def convert_hotpotqa_examples(examples: Sequence[HotpotQAExample]) -> HotpotQAConversionResult:
    ...
```

Raw JSON validation belongs in `parse_hotpotqa_examples`. Artifact contract validation still belongs in
`validate_memory_task_inputs` and `validate_memory_task_labels`. Do not expose
`tuple[list[dict], list[dict]]` from the converter.

Required behavior:

```text
input record has task_id, query, memory_items
label record has task_id, gold_answer, gold_evidence_nodes, gold_dependency_edges
sentence_id is the index inside one title's sentence list
position is the flattened index across all memory items in the task
```

- [ ] **Step 4: Implement deterministic split sampling**

Required public signature:

```python
def sample_split(examples: Sequence[T], count: int, seed: int, offset: int = 0) -> list[T]:
    ...
```

Required behavior:

```text
shuffle example indices with the provided seed
return examples at shuffled_indices[offset:offset + count]
raise ValueError when count or offset is negative
raise ValueError when offset + count exceeds available examples
```

Required split usage:

```text
train = sample_split(labeled_train_examples, 5000, seed=13, offset=0)
dev   = sample_split(labeled_dev_examples,    500, seed=13, offset=0)
test  = sample_split(labeled_dev_examples,   1000, seed=13, offset=500)
```

- [ ] **Step 5: Update `prepare_hotpotqa.py` CLI**

Required arguments:

```text
--input
--output_input
--output_labels
--output_combined
--max_examples
--seed
--offset
```

The script must write UTF-8 JSON with indentation and must not write labels into the input file. `--output_combined` is optional and exists only for compatibility with the original project schema; retrieval and graph-building commands must use `--output_input` artifacts, not combined artifacts.

- [ ] **Step 6: Run tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_data_structures.py -q
```

Expected: PASS.

### Task 2: Text Normalization And Entity Extraction

**Files:**
- Create: `graph_memory/text.py`
- Create: `graph_memory/entities.py`
- Test: `tests/test_phase1_real_graphs.py`

- [ ] **Step 1: Write tests for stopword-safe content tokens**

Test behavior:

```python
def test_content_tokens_drop_stopwords_and_keep_entities():
    tokens = content_tokens("Which city hosts the Eiffel Tower and what river runs through it?")
    assert "and" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "eiffel" in tokens
    assert "tower" in tokens
    assert "river" in tokens
```

- [ ] **Step 2: Write tests for lexical score**

Test behavior:

```python
def test_lexical_score_rewards_content_overlap_more_than_stopwords():
    idf = {"eiffel": 3.0, "tower": 3.0, "the": 0.0, "and": 0.0}
    assert lexical_score("the Eiffel Tower", "Eiffel Tower is in Paris", idf) > lexical_score("the and", "the and", idf)
```

- [ ] **Step 3: Implement text functions**

Required public functions:

```python
def content_tokens(text: str, keep_short: set[str] | None = None) -> list[str]:
    ...

def compute_idf(documents: list[str]) -> dict[str, float]:
    ...

def lexical_score(query: str, passage: str, idf: dict[str, float], title_aliases: set[str] | None = None, query_entities: set[str] | None = None, passage_entities: set[str] | None = None) -> float:
    ...
```

- [ ] **Step 4: Implement entity extraction**

Required public functions:

```python
def title_aliases(title: str) -> set[str]:
    ...

def heuristic_entities(text: str) -> set[str]:
    ...

def extract_entities(text: str, use_spacy: bool = False, nlp: object | None = None) -> set[str]:
    ...
```

Heuristic extraction must include capitalized phrase spans and title-like phrases. spaCy usage must be optional so tests pass without model downloads.

- [ ] **Step 5: Run graph text tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_graphs.py -q
```

Expected: PASS for text/entity tests.

### Task 3: Typed Graph Construction

**Files:**
- Create: `graph_memory/graphs.py`
- Create: `scripts/build_graphs.py`
- Test: `tests/test_phase1_real_graphs.py`

- [ ] **Step 1: Write tests for graph edge semantics**

Test behavior:

```python
def test_graph_builds_typed_edges_without_label_fields():
    config = GraphBuildConfig(max_query_overlap=20, max_entity_neighbors=10, max_bridge_edges=50)
    graph = build_graph(input_task, config)
    encoded = json.dumps(graph)
    assert "gold_answer" not in encoded
    assert "gold_evidence_nodes" not in encoded
    assert any(e["edge_type"] == "sequential" for e in graph["edges"])
    assert any(e["source"] == "q" and e["edge_type"] == "query_overlap" for e in graph["edges"])
    assert any(e["edge_type"] == "entity_overlap" for e in graph["edges"])
    assert any(e["edge_type"] == "bridge" for e in graph["edges"])
```

- [ ] **Step 2: Implement graph builder config**

Required dataclass:

```python
@dataclass(frozen=True)
class GraphBuildConfig:
    max_query_overlap: int = 20
    max_entity_neighbors: int = 10
    max_bridge_edges: int = 50
    use_spacy: bool = False
```

- [ ] **Step 3: Implement typed edge builders**

Required public functions:

```python
def build_graph(task_input: dict, config: GraphBuildConfig) -> dict:
    ...

def build_graphs(task_inputs: Sequence[MemoryTaskInput], config: GraphBuildConfig) -> list[MemoryGraph]:
    ...
```

Each edge must include:

```text
source
target
edge_type
weight
directed
```

- [ ] **Step 4: Update CLI**

Required arguments:

```text
--input
--output
--max_query_overlap
--max_entity_neighbors
--max_bridge_edges
--use_spacy
```

- [ ] **Step 5: Run graph tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_graphs.py -q
```

Expected: PASS.

### Task 4: Real BM25 And Frozen Dense Retrievers

**Files:**
- Create: `graph_memory/indexes/__init__.py`
- Create: `graph_memory/indexes/bm25.py`
- Create: `graph_memory/indexes/dense.py`
- Create: `graph_memory/retrieval.py`
- Create: `scripts/run_retrieval.py`
- Test: `tests/test_phase1_real_retrieval.py`

- [ ] **Step 1: Write tests for ranked schema**

Test behavior:

```python
def test_bm25_and_dense_emit_same_ranked_schema():
    for method in ["bm25", "dense"]:
        result = run_retrieval(method, task_inputs, graphs=[], top_k=5, encoder_model="sentence-transformers/all-MiniLM-L6-v2")
        assert result[0]["method"] == method
        assert len(result[0]["ranked_nodes"]) == len(task_inputs[0]["memory_items"])
        assert "node_id" in result[0]["ranked_nodes"][0]
        assert "score" in result[0]["ranked_nodes"][0]
        assert len(result[0]["retrieved_subgraph"]["nodes"]) <= 5
        assert result[0]["retrieved_subgraph"]["edges"] == []
```

- [ ] **Step 2: Implement BM25 retriever**

Required public class:

```python
class BM25TaskRetriever:
    def rank(self, task_input: dict) -> list[tuple[str, float]]:
        ...
```

Use `rank_bm25.BM25Okapi` over content tokens from `source + ". " + text`.

- [ ] **Step 3: Implement dense retriever**

Required public class:

```python
class DenseTaskRetriever:
    def __init__(self, model_name: str, batch_size: int = 64):
        ...

    def rank(self, task_input: dict) -> list[tuple[str, float]]:
        ...
```

Use normalized embeddings and dot product. Prefixes must be encoder-specific and configurable:

```text
query_prefix = "query: "
passage_prefix = "passage: "
```

- [ ] **Step 4: Update retrieval dispatcher**

Supported methods:

```text
bm25
dense
bm25_graph_rerank
dense_graph_rerank
```

`scripts/run_retrieval.py` required arguments:

```text
--method
--tasks
--graphs
--output
--top_k
--encoder_model
--query_prefix
--passage_prefix
--graph_config
```

`--tasks` must point to `*_memory_tasks.input.json`, not a combined file with labels. `--graph_config` is required for graph-rerank methods and ignored for flat methods.

- [ ] **Step 5: Run retrieval tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_retrieval.py -q
```

Expected: PASS with the small local sentence-transformers model available. If the model is not cached and network is unavailable, mark the dense test with a skip condition that reports the missing model path.

### Task 5: Graph Rerank And Retrieved Subgraph

**Files:**
- Create: `graph_memory/rerank.py`
- Create/update: `graph_memory/retrieval.py`
- Test: `tests/test_phase1_real_retrieval.py`

- [ ] **Step 1: Write tests for graph rerank behavior**

Test behavior:

```python
def test_graph_rerank_uses_bridge_to_promote_connected_evidence():
    initial_scores = {"m0": 1.0, "m1": 0.2, "m2": 0.8}
    graph = {
        "task_id": "ex1",
        "nodes": [],
        "edges": [
            {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 2.0, "directed": False}
        ],
    }
    config = GraphRerankConfig(lambda_init=1.0, lambda_neighbor=0.2, lambda_bridge=0.2, lambda_query=0.0, lambda_path=0.0)
    ranked = graph_rerank(initial_scores, graph, config)
    assert ranked[0][0] in {"m0", "m2"}
    assert ranked[1][0] in {"m0", "m2"}
```

- [ ] **Step 2: Implement rerank config**

Required dataclass:

```python
@dataclass(frozen=True)
class GraphRerankConfig:
    lambda_init: float = 1.0
    lambda_query: float = 0.1
    lambda_neighbor: float = 0.2
    lambda_bridge: float = 0.1
    lambda_path: float = 0.0
    seed_top_s: int = 30
    max_hops: int = 2
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "query_overlap": 0.8,
        "sequential": 0.3,
        "entity_overlap": 0.7,
        "bridge": 1.0,
    })
```

- [ ] **Step 3: Implement score normalization**

Required behavior:

```text
If all initial scores are equal, return 0.0 for every normalized score.
Otherwise apply min-max normalization inside the task.
```

- [ ] **Step 4: Implement candidate expansion and final scoring**

Required public function:

```python
def graph_rerank(initial_scores: dict[str, float], graph: dict, config: GraphRerankConfig) -> list[tuple[str, float]]:
    ...
```

Rerank must return all memory nodes present in `initial_scores`, not only expanded candidates.

- [ ] **Step 5: Implement retrieved subgraph extraction**

Required public function:

```python
def induced_retrieved_subgraph(graph: dict, node_ids: list[str]) -> dict:
    ...
```

- [ ] **Step 6: Run rerank tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_retrieval.py -q
```

Expected: PASS.

### Task 6: Dev Parameter Tuning

**Files:**
- Create: `graph_memory/tuning.py`
- Create: `scripts/tune_graph_rerank.py`
- Test: `tests/test_phase1_real_retrieval.py`

- [ ] **Step 1: Write tests for deterministic tuning**

Test behavior:

```python
def test_grid_search_selects_highest_objective_then_latency_tiebreak():
    rows = [
        {"config": {"lambda_neighbor": 0.1}, "Full Support@5": 0.5, "Recall@5": 0.5, "Connected Evidence Recall@10": 0.5, "Retrieval Latency / Query": 20.0},
        {"config": {"lambda_neighbor": 0.2}, "Full Support@5": 0.5, "Recall@5": 0.5, "Connected Evidence Recall@10": 0.5, "Retrieval Latency / Query": 10.0},
    ]
    assert select_best_config(rows)["lambda_neighbor"] == 0.2
```

- [ ] **Step 2: Implement objective**

Required public function:

```python
def tuning_objective(row: dict) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )
```

- [ ] **Step 3: Implement grid generation**

Required public function:

```python
def graph_rerank_grid() -> list[GraphRerankConfig]:
    ...
```

Use the candidate values listed in this document.

- [ ] **Step 4: Implement tuning CLI**

Required arguments:

```text
--method bm25_graph_rerank|dense_graph_rerank
--tasks
--labels
--graphs
--output_config
--encoder_model
--top_k
```

- [ ] **Step 5: Run tuning tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_retrieval.py -q
```

Expected: PASS.

### Task 7: Evaluation And Aggregation

**Files:**
- Create: `graph_memory/evaluation.py`
- Create: `scripts/evaluate_retrieval.py`
- Create: `scripts/aggregate_tables.py`
- Test: `tests/test_phase1_real_evaluation.py`

- [ ] **Step 1: Write metric tests**

Test behavior:

```python
def test_full_support_and_connected_evidence_use_top_k_nodes_on_shared_graph():
    ranked = ["m0", "m2", "m1"]
    gold = {"m0", "m2"}
    graph = {"edges": [{"source": "m0", "target": "m2", "edge_type": "bridge"}]}
    assert full_support_at(ranked, gold, 2) == 1.0
    assert connected_evidence_at(ranked, gold, graph, 2) == 1.0
```

- [ ] **Step 2: Implement label-aware evaluation**

Required public signature:

```python
def evaluate_results(
    predictions: Sequence[RankedResult],
    labels: Sequence[MemoryTaskLabels],
    graphs: Sequence[MemoryGraph],
) -> list[EvaluationRow]:
    ...
```

Do not read gold labels from input task files.

`scripts/evaluate_retrieval.py` required arguments:

```text
--pred
--labels
--graphs
--output
```

For compatibility with the original experiment command template, `--gold` may be accepted as an alias for `--labels`, but documentation should prefer `--labels` to make leakage separation explicit.

- [ ] **Step 3: Implement fair connectivity**

Connectivity must use the shared constructed graph and the method's selected top-k nodes. For flat methods, do not require the method to output edges.

- [ ] **Step 4: Implement efficiency aggregation**

Average retrieval latency from predictions. Read graph construction and index build timing from optional run metadata files when present; use `0.0` only when metadata is absent and record that absence in the per-method metric output or run summary notes.

- [ ] **Step 5: Run evaluation tests**

Run:

```powershell
uv run pytest tests/test_phase1_real_evaluation.py -q
```

Expected: PASS.

### Task 8: Experiment Commands And Reproducibility

**Files:**
- Modify: `README.md`
- Modify: `docs/40-operations/commands.md`
- Create: `configs/phase1_default.json`
- Create: `configs/phase1_graph_rerank_grid.json`
- Test: `tests/test_phase1_real_data_structures.py`

- [ ] **Step 1: Add reproducibility config**

`phase1_default.json` must include:

```json
{
  "seed": 13,
  "dataset": "hotpotqa_distractor",
  "train_examples": 5000,
  "dev_examples": 500,
  "test_examples": 1000,
  "dense_encoder": "intfloat/e5-base-v2",
  "query_prefix": "query: ",
  "passage_prefix": "passage: ",
  "top_k": 10,
  "graph": {
    "max_query_overlap": 20,
    "max_entity_neighbors": 10,
    "max_bridge_edges": 50,
    "use_spacy": false
  }
}
```

- [ ] **Step 2: Document full command sequence**

`docs/40-operations/commands.md` is the canonical runbook and must show commands for:

```text
prepare train input and labels from labeled train, offset 0
prepare dev input and labels from labeled dev, offset 0
prepare test input and labels from labeled dev, offset 500
optionally write combined *_memory_tasks.json compatibility artifacts
build graphs
run bm25
run dense
tune bm25_graph_rerank on dev
tune dense_graph_rerank on dev
run fixed graph rerank configs on test
evaluate all methods
aggregate tables
```

The root `README.md` should provide a short quick-start and link to `docs/40-operations/commands.md` instead of duplicating the full command sequence.

- [ ] **Step 3: Add leakage check command**

`docs/40-operations/commands.md` must include:

```powershell
rg "gold_answer|gold_evidence_nodes|supporting_facts|is_gold" data/hotpotqa/processed/*input*.json data/hotpotqa/processed/*graphs*.json
```

Expected: no matches.

- [ ] **Step 4: Run all tests**

Run:

```powershell
uv run pytest tests -q
```

Expected: PASS.

## Final Acceptance Criteria

- `bm25`, `dense`, `bm25_graph_rerank`, and `dense_graph_rerank` produce ranked results in one schema.
- Dense retrieval uses a real frozen Sentence-Transformers encoder.
- Graph construction uses no label-only fields.
- Dev tuning selects graph rerank parameters and saves them before test evaluation.
- Test evaluation reads labels from label artifacts, not from model input artifacts.
- Final HotpotQA Phase 1 tables include Recall@k, Evidence F1@k, Full Support@k, MRR, Connected Evidence Recall@k, Query-Evidence Connectivity@10, and efficiency metrics.
- `Path Recall@10` and `Edge Recall@10` are emitted as `N/A` for HotpotQA-only Phase 1 unless a dataset with gold dependency edges is added.
- `docs/40-operations/commands.md` contains exact commands, dataset split sizes, random seed, encoder model, graph settings, top-k, and hardware notes; the root `README.md` links to it.
- README clearly states this plan satisfies the Phase 1 minimum runnable version, not the full paper-version baseline set with Dense-FT, Memory Stream, GraphRAG, and MemGPT-style memory.

## Self-Review

Spec coverage:

- HotpotQA conversion is covered by Task 1.
- Stopword-safe lexical scoring and entity extraction are covered by Task 2.
- Typed graph construction is covered by Task 3.
- Real BM25 and frozen dense baselines are covered by Task 4.
- Graph reranking is covered by Task 5.
- Automatic dev-set parameter selection is covered by Task 6.
- Metrics and fair connectivity evaluation are covered by Task 7.
- Reproducibility and command documentation are covered by Task 8.

Leakage review:

- Input tasks exclude `gold_answer`, `gold_evidence_nodes`, and `supporting_facts`.
- Graph construction consumes only input tasks.
- Evaluation consumes labels separately.
- Dev tuning may use labels, but test-time retrieval may not.

Scope review:

- This plan intentionally excludes Dense-FT, GNN training, GraphRAG, Memory Stream, MemGPT-style memory, answer generation, 2WikiMultiHopQA, MuSiQue, and tool trajectories.
- Those items belong after Phase 1 produces stable HotpotQA evidence retrieval results.
