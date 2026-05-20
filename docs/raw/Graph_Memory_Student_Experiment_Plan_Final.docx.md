**Execution-Provenance Graph Memory**

**Student-Executable Experiment Plan with Implementation Links**

Goal: convert public QA / trajectory data into evidence-tracing memory tasks and evaluate whether graph memory retrieves complete evidence nodes and evidence paths better than flat memory baselines.

# 0. Paper Background: What This Project Is About

This paper studies a memory problem in evidence-heavy LLM agent systems. In many multi-agent or tool-use workflows, the final answer is supported by multiple earlier observations, retrieved documents, tool outputs, intermediate claims, and parameter transfers. Standard flat memory or vector retrieval stores these records as isolated text chunks, so it may retrieve semantically similar items but miss the dependency chain that actually explains why an answer is supported.

The proposed method, Execution-Provenance Graph Memory, represents each sentence, observation, tool call, or tool result as a memory node, and explicitly links nodes with edges such as sequential context, entity overlap, bridge connections, evidence support, tool dependency, and parameter flow. The goal is not simply to answer QA questions, but to retrieve the complete evidence set and the connected evidence path behind a query or claim.

Therefore, the experiments should evaluate evidence tracing first. The main question is: given a query and a memory graph, can each method retrieve the supporting evidence nodes and their dependency path? Answer generation can be added later as an optional consistency check, but it should not be the main Phase 1 result.

# 1. Read This First: What the Student Must Build

* This experiment is not about proving a new QA dataset. It constructs evidence-tracing memory tasks from existing evidence-intensive QA datasets and a small optional agent-style trajectory set.
* Phase 1 only evaluates evidence retrieval / evidence tracing. Do not make answer generation the main experiment at first; it adds LLM noise.
* Each example must become: query + memory_items + graph_edges -> ranked evidence nodes / evidence subgraph.
* The core claim to test: flat memory retrieval can miss multi-step evidence dependencies, while execution-provenance graph memory better recovers complete evidence sets and connected evidence paths.

| Item | Decision |
| :---- | :---- |
| Primary dataset | HotpotQA, sentence-level evidence tracing. |
| Primary output | Ranked evidence nodes and top-k evidence subgraph. |
| Must-have baselines | BM25, Dense, Dense-FT, Memory Stream, GraphRAG, Ours. |
| Main metrics | Recall@k, Evidence F1@k, Full Support@k, Connected Evidence Recall@k, Path Recall@k. |
| Main paper structure | Experimental Setup, Main Results, Ablation Study, Error and Efficiency Analysis. |

## 1.1 Non-negotiable Implementation Notes

* **Use only labeled evaluation data.** For the first version, construct train/dev/test from available labeled splits. Do not use an unlabeled official test set for metric computation. Recommended first split: train 5,000 / dev 500 / test 1,000 sampled from labeled HotpotQA train/dev data.
* **Avoid answer leakage in Phase 1.** In Phase 1 evidence retrieval, do not include the answer node or gold answer text in model input. The answer node is only for optional path analysis or later answer-support experiments. Test-time model input must not contain is_gold_evidence, is_gold_edge, supporting_facts, or any label-only field.
* **Use simplified faithful baselines when full systems are too heavy.** For GraphRAG and MemGPT-style baselines, simplified faithful implementations are acceptable. The goal is not to reproduce their entire systems, but to compare the corresponding memory principle: entity/relation graph retrieval for GraphRAG, and recent-buffer plus archival-memory retrieval for MemGPT-style memory.

# 2. Execution Priority: Do Not Start with Everything

| Phase | Scope | Student output | Pass condition |
| :---- | :---- | :---- | :---- |
| Phase 1: Minimum runnable version | HotpotQA + BM25 + Dense + Ours Graph Rerank | memory_tasks, graphs, ranked_results, main_results | BM25/Dense/Ours all produce ranked nodes; metrics script runs without manual fixes. |
| Phase 2: Main paper version | Add Dense-FT, Memory Stream, GraphRAG, edge ablations | main_results, path_results, ablation_results | Ours is compared against at least one sparse, one dense, one agent-memory, and one graph-RAG baseline. |
| Phase 3: Enhanced version | Add MemGPT-style, 2WikiMultiHopQA, tool trajectories | generalization_results, tool_results, case studies | Shows generalization beyond HotpotQA and includes qualitative error analysis. |
| Optional hard setting | Add MuSiQue | musique_results | Stress test on longer 2-4 hop evidence chains. |

# 3. Dataset Sources and Download Links

| Dataset | Use in this project | Direct link | First-run setting |
| :---- | :---- | :---- | :---- |
| HotpotQA | Primary benchmark. Has natural multi-hop questions and sentence-level supporting facts. | Homepage: https://hotpotqa.github.io/<br>GitHub: https://github.com/hotpotqa/hotpot<br>HF: https://huggingface.co/datasets/hotpotqa/hotpot_qa | Use distractor setting first. Pilot: train 5k / dev 500 / test 1k. |
| 2WikiMultiHopQA | Generalization benchmark. Better for reasoning-path / evidence-path evaluation. | GitHub: https://github.com/Alab-NII/2wikimultihop<br>HF mirror: https://huggingface.co/datasets/xanhho/2WikiMultihopQA | Add after HotpotQA pipeline is stable. Start with test 1k. |
| MuSiQue | Optional hard setting. Longer 2-4 hop questions and reduced shortcut reasoning. | GitHub: https://github.com/stonybrooknlp/musique<br>Paper page: https://aclanthology.org/2022.tacl-1.31/ | Optional. Start with test 1k only. |
| Synthetic tool trajectories | Optional generality analysis for agent-style execution histories. | Create manually in this project; no external dataset required. | 100-300 examples, not the main result. |

**Paper wording to use:** We construct evidence-tracing memory tasks from public evidence-intensive QA datasets and an additional agent-style tool-use trajectory set.

# 4. Unified Task Definition and File Schema

Every dataset must be converted into the same task format. This keeps all baselines fair and makes evaluation reusable.

```json
{
  "task_id": "hotpot_000001",
  "query": "question text",
  "gold_answer": "answer text",
  "memory_items": [
    {
      "id": "m0",
      "node_type": "document_sentence",
      "text": "sentence text",
      "source": "Document_Title_1",
      "sentence_id": 0,
      "position": 0
    }
  ],
  "gold_evidence_nodes": ["m1", "m7"],
  "gold_dependency_edges": []
}
```

| Raw field | Converted field | Rule |
| :---- | :---- | :---- |
| question | query | Use as retrieval query. |
| answer | gold_answer | Keep for optional answer-support consistency; not the main metric in Phase 1. |
| context sentences | memory_items | Each sentence becomes one memory node. Do not use whole paragraph nodes for HotpotQA. |
| supporting_facts | gold_evidence_nodes | Map title + sentence_id to memory node id. |
| reasoning path / tool dependency | gold_dependency_edges | Use when available, especially 2Wiki and tool trajectories. |

# 5. Required Directory Structure

```text
data/
  hotpotqa/
    raw/
      train.json
      dev.json
    processed/
      train_memory_tasks.json
      dev_memory_tasks.json
      test_memory_tasks.json
      train_graphs.json
      dev_graphs.json
      test_graphs.json
      train_pairs.json
      dev_pairs.json
      test_pairs.json
  2wikimultihopqa/
    raw/
    processed/
      *_memory_tasks.json
      *_graphs.json
      *_pairs.json
  musique/
    raw/
    processed/
  tool_trajectory/
    raw/
      synthetic_tool_tasks.json
    processed/
      tool_memory_tasks.json
      tool_graphs.json
      tool_pairs.json
results/
  ranked_results_{method}.json
  main_results.csv
  path_results.csv
  ablation_results.csv
  efficiency_results.csv
  error_analysis.md
```

| File | Required content | Used by |
| :---- | :---- | :---- |
| `*_memory_tasks.json` | query, memory_items, gold_evidence_nodes, optional gold_answer | BM25, Dense, Dense-FT, Memory Stream, MemGPT-style. |
| `*_graphs.json` | task_id, nodes, edges; no gold label leakage in model input | Ours, GraphRAG-style graph baseline, ablation. |
| `*_pairs.json` | query-positive-negative pairs with hard negatives | Dense-FT and trainable Graph Retriever. |
| `ranked_results_{method}.json` | ranked_nodes, scores, optional retrieved_subgraph, latency | All evaluation scripts. |
| main/path/ablation/efficiency CSVs | Final tables with fixed columns | Paper tables. |

# 6. Graph Construction Rules

## 6.1 Node Types

| Node type | Node id | Used in model input? | Notes |
| :---- | :---- | :---- | :---- |
| question | q | Yes | Current query node. |
| document_sentence | m0, m1, ... | Yes | One node per candidate sentence. |
| answer | a | Careful | Can be used for training/evaluation path analysis; avoid leaking gold answer if the task is pure retrieval. |

## 6.2 Edge Types

| Edge type | Construction rule | Limit | Purpose |
| :---- | :---- | :---- | :---- |
| sequential | Same document: sentence_i <-> sentence_{i+1} | All adjacent pairs | Preserve local document context. |
| query_overlap | q -> sentence if shared content-word overlap is high | Top 20 sentences per task | Seed query-relevant nodes without making the graph too dense. |
| entity_overlap | Two sentences share title, capitalized phrase, answer string, query entity, or important non-stopword phrase | Top 10 neighbors per node | Capture cross-sentence / cross-document entity links. |
| bridge | Cross-document nodes share entity, title mention, or query-answer entity connection | Top 50 per task | Connect HotpotQA-style multi-hop supporting facts across documents. |
| evidence_support | gold supporting fact -> answer node | Gold only | Use for training/evaluation; hide is_gold_evidence/is_gold_edge at test time. |
| tool_dependency | tool result -> downstream tool call that uses it | Tool trajectory only | Model execution provenance in agent-style traces. |
| parameter_flow | observation parameter -> downstream decision/tool argument | Tool trajectory only | Trace parameter transfer across steps. |

**Leakage rule:** For test-time model input, remove or mask is_gold_evidence=true, is_gold_edge=true, supporting_facts, gold_answer if it would reveal the answer, and any label-only field. Keep labels only in evaluation files.

# 7. Training Pair Generation

```json
{
  "task_id": "hotpot_000001",
  "query": "question text",
  "positive_nodes": ["m1", "m7"],
  "negative_nodes": ["m2", "m3", "m4", "m9"],
  "hard_negative_nodes": ["m5", "m6"],
  "graph_id": "hotpot_000001"
}
```

| Type | How to generate | Required ratio |
| :---- | :---- | :---- |
| Positive nodes | gold_evidence_nodes | All gold nodes. If a task has two gold nodes, create two positive-centered instances. |
| Easy negatives | Random non-gold sentences | 4 per positive. |
| Hard negatives | BM25 top-ranked non-gold; query/entity-overlap non-gold; same document as gold but non-gold; answer-string-close but non-gold | 4 per positive. |
| Final ratio | positive : easy negatives : hard negatives | 1 : 4 : 4 |

# 8. Baselines: Exact Implementation Sources, Fallbacks, and What to Report

Rule: every baseline must answer one reviewer question. Do not add methods that do not directly test memory retrieval / evidence tracing.

| Baseline | Reviewer question answered | Primary implementation link | Fallback implementation | Required? |
| :---- | :---- | :---- | :---- | :---- |
| BM25 | Is keyword retrieval already enough? | Pyserini: https://github.com/castorini/pyserini<br>rank-bm25: https://github.com/dorianbrown/rank_bm25 | Use rank-bm25 for fastest local implementation. | Phase 1 |
| Dense / DPR / Dense RAG | Is ordinary vector memory enough? | DPR: https://github.com/facebookresearch/DPR<br>Sentence-Transformers: https://github.com/huggingface/sentence-transformers<br>Docs: https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html | Use sentence-transformers + cosine similarity / FAISS. | Phase 1 |
| Dense-FT | Is dense baseline too weak if not fine-tuned? | Use project train_pairs.json; DPR-style or sentence-transformers fine-tuning. | If fine-tuning is slow, first report frozen dense and add Dense-FT later. | Phase 2 |
| Generative Agents Memory Stream | Does classic agent memory already solve this? | Repo: https://github.com/joonspk-research/generative_agents | Simplified score = relevance + recency + importance. | Phase 2 |
| MemGPT-style memory | Does hierarchical long-context memory already solve this? | Letta / formerly MemGPT: https://github.com/letta-ai/letta | Simplified recent buffer + archival vector memory. Do not fully reproduce Letta. | Phase 3 |
| GraphRAG | Does existing entity/relation graph RAG already solve this? | Microsoft GraphRAG: https://github.com/microsoft/graphrag<br>Docs: https://microsoft.github.io/graphrag/ | Simplified entity-relation graph over memory sentences. | Phase 2 |
| HippoRAG | Alternative graph-memory baseline. | HippoRAG: https://github.com/OSU-NLP-Group/HippoRAG | Only use if GraphRAG is not suitable or time permits. | Optional |
| Ours: Graph Rerank | Does execution-provenance graph help without heavy training? | Implemented in this project. | Initial score + graph propagation. | Phase 1 |
| Ours: Trainable Graph Retriever | Does learned edge-type-aware graph memory improve evidence tracing? | Implemented in this project. | R-GCN / GAT / edge-type message passing with BCE loss. | Phase 2/3 |

**Recommended graph baseline choice:** Use GraphRAG first. It is easier to explain because GraphRAG constructs an entity/relation graph, while our method constructs an execution-provenance/evidence-dependency graph. HippoRAG is closer to memory retrieval but can be left optional if implementation time is limited.

# 9. Ours: Two Implementation Versions

## 9.1 Version A - Graph Reranking, no GNN training

1. Run BM25 or Dense retrieval to get initial_score(v) for every memory node.

2. Propagate score over graph neighbors using edge-type weights.

3. Re-rank nodes by final_score(v).

4. Return ranked evidence nodes and top-k induced subgraph.

```text
final_score(v) = initial_score(v)
               + alpha * sum(initial_score(u) for u in neighbors(v))
               + beta  * bridge_bonus(v)
               + gamma * query_overlap_bonus(v)
```

Suggested first parameters:
```text
alpha = 0.2
beta  = 0.1
gamma = 0.1
```

## 9.2 Version B - Trainable Graph Retriever

1. Read train_pairs.json and train_graphs.json.

2. Encode query and node text using the same text encoder used by Dense baseline.

3. Apply edge-type-aware message passing, e.g., R-GCN, GAT with edge type embeddings, or simple typed-neighbor aggregation.

4. Score each node as evidence or non-evidence.

5. Train first with BCE loss; add InfoNCE/contrastive loss only after BCE works.

```text
h_v = Encoder(node_text)
h_q = Encoder(query)
score(v) = MLP([h_v, h_q, h_v * h_q])
```

```text
label = 1 for gold evidence nodes
label = 0 for sampled negative nodes
first loss = BCE loss
```

# 10. Required Scripts and Command Templates

The student should implement these scripts with exactly these inputs and outputs. Names can change only if all results remain compatible with the same JSON schemas.

**prepare_hotpotqa.py:** Convert raw HotpotQA to sentence-level memory tasks.

```bash
python scripts/prepare_hotpotqa.py \
  --input data/hotpotqa/raw/train.json \
  --output data/hotpotqa/processed/train_memory_tasks.json \
  --max_examples 5000
```

**build_graphs.py:** Construct nodes and graph edges.

```bash
python scripts/build_graphs.py \
  --input data/hotpotqa/processed/train_memory_tasks.json \
  --output data/hotpotqa/processed/train_graphs.json \
  --max_query_overlap 20 \
  --max_entity_neighbors 10 \
  --max_bridge_edges 50
```

**build_pairs.py:** Create positive/easy-negative/hard-negative training pairs.

```bash
python scripts/build_pairs.py \
  --tasks data/hotpotqa/processed/train_memory_tasks.json \
  --graphs data/hotpotqa/processed/train_graphs.json \
  --output data/hotpotqa/processed/train_pairs.json \
  --easy_negatives 4 \
  --hard_negatives 4
```

**run_retrieval.py:** Run any baseline or ours; output ranked nodes in one schema.

```bash
python scripts/run_retrieval.py \
  --method bm25 \
  --tasks data/hotpotqa/processed/test_memory_tasks.json \
  --graphs data/hotpotqa/processed/test_graphs.json \
  --output results/ranked_results_bm25.json \
  --top_k 10
```

**evaluate_retrieval.py:** Compute node-level/path-level/efficiency metrics.

```bash
python scripts/evaluate_retrieval.py \
  --pred results/ranked_results_bm25.json \
  --gold data/hotpotqa/processed/test_memory_tasks.json \
  --graphs data/hotpotqa/processed/test_graphs.json \
  --output results/main_results_bm25.csv
```

**aggregate_tables.py:** Merge per-method metrics into final paper tables.

```bash
python scripts/aggregate_tables.py \
  --input_dir results/ \
  --output_main results/main_results.csv \
  --output_path results/path_results.csv \
  --output_efficiency results/efficiency_results.csv
```

# 11. Unified Ranked Result Format

```json
{
  "task_id": "hotpot_000001",
  "method": "graph_memory",
  "ranked_nodes": [
    {"node_id": "m3", "score": 12.4},
    {"node_id": "m8", "score": 11.9}
  ],
  "retrieved_subgraph": {
    "nodes": ["m3", "m8"],
    "edges": [["m3", "m8", "bridge"]]
  },
  "latency_ms": 23.5,
  "input_tokens": 640
}
```

All methods must output this format. If a baseline has no graph, leave retrieved_subgraph empty or include only retrieved nodes with no edges.

# 12. Metrics and Final Paper Tables

## 12.1 Metric Definitions

| Metric | Definition | Report where |
| :---- | :---- | :---- |
| Recall@k | retrieved gold evidence nodes / all gold evidence nodes | Main Results. Use @2, @5, @10. |
| Evidence F1@k | Harmonic mean of Precision@k and Recall@k | Main Results. Use @5, @10. |
| Full Support@k | 1 if top-k contains all gold evidence nodes, else 0; average over examples | Main Results. Very important for complete evidence set. |
| MRR | Reciprocal rank of first gold evidence node | Main Results as auxiliary metric. |
| Connected Evidence Recall@k | 1 if top-k contains all gold evidence nodes and they are connected in the induced graph | Path-level Results. Use @5, @10. |
| Edge Recall@k | retrieved gold dependency edges / all gold dependency edges | 2Wiki/tool trajectory mainly; HotpotQA optional. |
| Path Recall@k | 1 if retrieved subgraph contains a path connecting question -> evidence -> answer, or tool output -> downstream decision | Path-level Results. Use @10. |
| Latency / memory size | Retrieval latency per query, graph construction time, index time, memory size | Efficiency table. |

## 12.2 Fixed Table Columns

| Paper table | Exact columns |
| :---- | :---- |
| Table 1: Dataset / Graph Statistics | Dataset<br>#Examples<br>Avg #Nodes<br>Avg #Edges<br>Avg Degree<br>Avg #Gold Evidence Nodes<br>% Full Gold Evidence Connected<br>Avg Gold Evidence Distance |
| Table 2: Main Results | Method<br>Recall@2<br>Recall@5<br>Recall@10<br>Evidence F1@5<br>Evidence F1@10<br>Full Support@5<br>Full Support@10<br>MRR |
| Table 3: Path-level Results | Method<br>Connected Evidence Recall@5<br>Connected Evidence Recall@10<br>Path Recall@10<br>Edge Recall@10 |
| Table 4: Ablation Study | Variant<br>Recall@5<br>Full Support@5<br>Connected Evidence Recall@10<br>Path Recall@10<br>Latency |
| Table 5: Efficiency | Method<br>Index Build Time<br>Graph Construction Time<br>Retrieval Latency / Query<br>Memory Size<br>Avg Retrieved Nodes<br>Avg Retrieved Edges |

# 13. Ablation Study: Required Variants

| Variant | Purpose | Required phase |
| :---- | :---- | :---- |
| Full Graph Memory | Complete method. | Phase 2 |
| w/o bridge edges | Tests whether cross-document evidence connection is important. | Phase 2 |
| w/o entity-overlap edges | Tests whether entity links drive multi-hop tracing. | Phase 2 |
| w/o sequential edges | Tests whether local document adjacency matters. | Phase 2/3 |
| w/o query-overlap edges | Tests query-to-memory seed connection. | Phase 2/3 |
| w/o graph propagation | Shows gains are not only from BM25/Dense initial score. | Phase 1/2 |
| w/o hard negatives | Tests robustness from hard-negative training. | Phase 2 if Dense-FT/Graph Retriever exists |
| w/o edge type | Tests typed edge modeling. | Phase 3 if trainable graph model exists |
| random edges | Tests whether the constructed graph semantics matter. | Optional but useful |

# 13. Optional Agent-style Tool Trajectory Set

Use only after the HotpotQA pipeline is stable. This is a generality analysis, not the main experiment.

```json
{
  "task_id": "tool_001",
  "user_goal": "Find the cheapest flight and book a hotel near the conference venue.",
  "trajectory": [
    {"step_id": "s1", "node_type": "tool_call", "tool": "search_flight",
     "text": "search_flight(from=Sydney, to=Boston, date=2026-07-10)"},
    {"step_id": "s2", "node_type": "tool_result",
     "text": "Flight A costs $1200 and arrives at 8pm."},
    {"step_id": "s3", "node_type": "tool_call", "tool": "search_hotel",
     "text": "search_hotel(location=conference venue, check_in=2026-07-10)"}
  ],
  "query": "Which previous result should be used to decide the hotel check-in time?",
  "gold_evidence_nodes": ["s2"],
  "gold_dependency_edges": [["s2", "s3"]],
  "gold_answer": "Use the flight arrival result from s2."
}
```

| Edge type | Meaning |
| :---- | :---- |
| temporal | Execution order: s1 -> s2 -> s3. |
| tool_dependency | A downstream tool call depends on an earlier tool result. |
| parameter_flow | A value from an observation becomes a later argument or decision. |
| entity_overlap | Shared place, date, organization, person, or object. |

# 14. How This Maps to the Paper Experiments Section

| Paper subsection | Content |
| :---- | :---- |
| Experimental Setup | Dataset construction, graph construction, baseline links, model setup, metrics, dataset statistics. |
| Main Results | Table 2 + Table 3. Emphasize Evidence Recall, Full Support, Connected Evidence Recall, and Path Recall. |
| Ablation Study | Table 4. Emphasize that removing bridge/entity/provenance edges reduces Full Support and Path Recall. |
| Error and Efficiency Analysis | Table 5 + 3 case studies. Explain graph success/failure and show practical latency/memory cost. |

# 15. Final Deliverable Checklist for Student

| Deliverable | File name | Minimum acceptance standard |
| :---- | :---- | :---- |
| Processed memory tasks | `data/*/processed/*_memory_tasks.json` | query, memory_items, gold_evidence_nodes all exist. |
| Memory graphs | `data/*/processed/*_graphs.json` | nodes/edges complete; test-time input contains no gold label leakage. |
| Training pairs | `data/*/processed/*_pairs.json` | positive/easy/hard negative ratio = 1:4:4 where possible. |
| Retrieval outputs | `results/ranked_results_{method}.json` | All methods use the same ranked_nodes schema. |
| Main table | results/main_results.csv | Contains BM25, Dense, Dense-FT, Memory Stream, GraphRAG, Ours. |
| Path table | results/path_results.csv | Contains Connected Evidence Recall and Path Recall. |
| Ablation table | results/ablation_results.csv | Contains full graph, w/o bridge, w/o entity, w/o graph propagation. |
| Efficiency table | results/efficiency_results.csv | Contains latency, memory size, graph construction time, index time. |
| Error analysis | results/error_analysis.md | At least 3 cases: dense failure, graph success, graph failure. |
| Reproducibility notes | README.md | Exact dataset version, commands, random seed, encoder model, top-k, hardware. |

# 16. One-sentence Task Definition for the Student

**Convert QA / trajectory data into memory retrieval tasks:** query = question/goal, node = sentence/step, gold = supporting facts, edge = sequential/overlap/bridge/dependency; train and evaluate whether each method retrieves the complete evidence set and connected evidence path.

# 17. Link Summary

| Resource | URL |
| :---- | :---- |
| HotpotQA homepage | https://hotpotqa.github.io/ |
| HotpotQA GitHub | https://github.com/hotpotqa/hotpot |
| HotpotQA HuggingFace | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| 2WikiMultiHopQA GitHub | https://github.com/Alab-NII/2wikimultihop |
| 2WikiMultiHopQA HuggingFace mirror | https://huggingface.co/datasets/xanhho/2WikiMultihopQA |
| MuSiQue GitHub | https://github.com/stonybrooknlp/musique |
| MuSiQue paper page | https://aclanthology.org/2022.tacl-1.31/ |
| Pyserini | https://github.com/castorini/pyserini |
| rank-bm25 | https://github.com/dorianbrown/rank_bm25 |
| DPR | https://github.com/facebookresearch/DPR |
| Sentence-Transformers | https://github.com/huggingface/sentence-transformers |
| Sentence-Transformers semantic search docs | https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html |
| Generative Agents | https://github.com/joonspk-research/generative_agents |
| Letta / MemGPT-style memory | https://github.com/letta-ai/letta |
| Microsoft GraphRAG | https://github.com/microsoft/graphrag |
| GraphRAG docs | https://microsoft.github.io/graphrag/ |
| HippoRAG | https://github.com/OSU-NLP-Group/HippoRAG |
