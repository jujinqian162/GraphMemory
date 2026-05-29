# Project Overview

## Metadata

| Field | Value |
|---|---|
| Project | Execution-Provenance Graph Memory |
| Current scope | HotpotQA evidence-tracing retrieval with Phase 1 baselines and the Phase 2 R-GCN trainable retriever slice |
| Primary task | Retrieve complete supporting evidence nodes and connected evidence subgraphs. |
| Primary dataset | HotpotQA distractor setting with labeled splits only. |
| Primary methods | BM25, frozen dense retrieval, graph-aware reranking, checkpoint-backed R-GCN graph retrieval. |
| Source material | `docs/archive/original-student-experiment-plan.md` |
| Current implementation plans | `docs/10-plans/phase1-real-graph-memory.md`; `docs/10-plans/phase2-rgcn-trainable-retriever.md` |

## Background

This project studies memory retrieval for evidence-heavy LLM and agent workflows. Standard flat memory stores observations, documents, and tool outputs as isolated text chunks. That can retrieve semantically similar records, but it often misses the dependency chain that explains why an answer is supported.

Execution-Provenance Graph Memory represents each evidence-bearing sentence or step as a node and links nodes through typed edges such as sequential context, query overlap, entity overlap, bridge connections, tool dependency, and parameter flow.

The key research question is not only whether a retriever finds one relevant sentence. The stricter question is whether it recovers the complete evidence set and the connected evidence path behind a query.

## Core Requirements

- Convert public evidence-intensive QA data into a unified memory retrieval task format.
- Use sentence-level memory nodes for HotpotQA retrieval experiments.
- Evaluate evidence retrieval and evidence tracing before answer generation.
- Keep label-only fields separate from retrieval and graph-construction inputs.
- Compare flat retrieval, hand-written graph reranking, and trainable graph retrieval under a shared ranked-result schema.
- Report node-level metrics, graph connectivity metrics, and efficiency metrics.
- Prefer fail-fast validation over silent fallback behavior.
- Keep each phase slice narrow enough to be runnable, auditable, and scientifically interpretable.

## Phase Roadmap

| Phase | Scope | Expected output |
|---|---|---|
| Phase 1 | HotpotQA + BM25 + frozen dense + graph rerank | Runnable evidence retrieval system and main metrics. |
| Phase 2 R-GCN slice | Train-pair artifact, R-GCN binary node scorer, checkpoint-backed retrieval, trainable method integration | Implemented trainable graph retriever path that emits standard ranked results. |
| Remaining Phase 2 paper scope | Add Dense-FT, Memory Stream, GraphRAG, full edge/model ablations | Main paper comparison and ablation tables. |
| Phase 3 | Add MemGPT-style memory, 2WikiMultiHopQA, tool trajectories | Generalization and agent-style provenance analysis. |
| Optional | Add MuSiQue | Harder multi-hop stress test. |

## Current Implemented Boundary

The stable Phase 1 path implements:

- HotpotQA conversion into input and label artifacts.
- Typed graph construction from input-visible text.
- BM25 retrieval.
- Frozen dense retrieval.
- BM25-seeded and dense-seeded graph reranking.
- Dev-set graph parameter tuning.
- Evaluation of Recall@k, Evidence F1@k, Full Support@k, MRR, Connected Evidence Recall@k, Query-Evidence Connectivity@10, and efficiency.

The implemented Phase 2 R-GCN slice adds:

- `*_pairs.json` train-pair construction from input, label, and graph artifacts.
- `GraphBatch` / `TrainingBatch` tensor contracts for trainable graph retrieval.
- Frozen-encoder R-GCN binary evidence node scoring.
- Checkpoint save/load and `dense_rgcn_graph_retriever` inference.
- Experiment-runner stages for trainable pair building, training, retrieval, evaluation, and aggregation.

The broader Phase 2 paper scope still needs:

- Dense-FT baseline.
- Memory Stream baseline.
- GraphRAG-style baseline.
- Full ablation run orchestration and `ablation_results.csv`.
- MemGPT-style memory, answer generation, 2WikiMultiHopQA, MuSiQue, and tool trajectory experiments.

## Documentation Flow

Start here, then read:

1. `docs/archive/original-student-experiment-plan.md` for full source context.
2. `docs/10-plans/phase1-real-graph-memory.md` for the Phase 1 implementation plan.
3. `docs/10-plans/phase2-rgcn-trainable-retriever.md` for the implemented R-GCN trainable retriever slice.
4. `docs/20-contracts/data-contracts.md`, `docs/20-contracts/retrieval-contracts.md`, and `docs/20-contracts/model-contracts.md` for current artifact, retrieval, and model contracts.
5. `docs/10-plans/engineering-quality-brainstorm.md` for evolving engineering decisions.
