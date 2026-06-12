# Project Overview

## Metadata

| Field | Value |
|---|---|
| Project | Execution-Provenance Graph Memory |
| Current scope | HotpotQA evidence-tracing retrieval with flat baselines, graph reranking, Dense-FT, and trainable R-GCN retrieval |
| Primary task | Retrieve complete supporting evidence nodes and connected evidence subgraphs. |
| Primary dataset | HotpotQA distractor setting with labeled splits only. |
| Primary methods | BM25, frozen dense retrieval, Dense-FT, graph-aware reranking, checkpoint-backed R-GCN graph retrieval. |
| Source material | `docs/archive/original-student-experiment-plan.md` |
| Current implementation plans | `docs/10-plans/phase1-real-graph-memory.md`; `docs/10-plans/phase2-rgcn-trainable-retriever.md`; `docs/10-plans/dense-ft-implementation-plan.md` |

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
| Phase 1 | HotpotQA + BM25 + frozen dense + graph rerank | Implemented runnable evidence retrieval system and main metrics. |
| Implemented Phase 2 methods | Dense-FT; train-pair artifacts; R-GCN binary node scorer; checkpoint/model-directory retrieval; edge/model ablations | Implemented trainable retrieval paths, standard ranked results, and `ablation_results.csv`. |
| Remaining Phase 2 paper scope | Add Memory Stream and GraphRAG-style baselines; produce one final comparison package across all required methods | Complete the original paper baseline matrix. |
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

The implemented trainable stack adds:

- `*_pairs.json` train-pair construction from input, label, and graph artifacts.
- `GraphBatch` / `TrainingBatch` tensor contracts for trainable graph retrieval.
- Frozen-encoder R-GCN binary evidence node scoring.
- Checkpoint save/load and `dense_rgcn_graph_retriever` inference.
- SentenceTransformers-based Dense-FT training, model-directory metadata, and `dense_ft` inference.
- Strict current-only method configs under `configs/methods/`.
- Precompiled pair, train, retrieve, and evaluate stage configs consumed through `--config`.
- Runtime-produced model, device, and encoder provenance in retrieval summaries.
- Experiment-runner stages for pair building, training, retrieval, evaluation, aggregation, resume, and artifact status.
- R-GCN ablation orchestration for edge views, graph structure, edge typing/weighting, seed scores, and hard negatives.

The broader Phase 2 paper scope still needs:

- Memory Stream baseline.
- GraphRAG-style baseline.
- One final comparison package containing all required Phase 2 methods.

The current HotpotQA graph evaluates evidence-node recovery and graph connectivity, but it does not contain gold execution dependency paths. Consequently, `Path Recall@10` and `Edge Recall@10` remain `N/A`. MemGPT-style memory, answer generation, 2WikiMultiHopQA, MuSiQue, and tool-trajectory experiments remain outside the implemented boundary.

## Current Architecture Boundary

- `graph_memory/registry/methods.py` is the source of truth for method lifecycle, dependencies, encoder/model sources, and train artifact shape.
- Trainable method configs are strict current-only files under `configs/methods/`; old config schemas and migrations are unsupported.
- Workflow manifests and generated stage configs are strict current contracts. Low-level trainable scripts consume complete stage configs only.
- R-GCN checkpoints and Dense-FT model metadata are current-only artifacts. Old runs and artifacts must be deleted and regenerated.
- Retrieval builders return the actual runtime provenance serialized into run summaries.

## Documentation Flow

Start here, then read:

1. `docs/archive/original-student-experiment-plan.md` for full source context.
2. `docs/10-plans/phase1-real-graph-memory.md` for the Phase 1 implementation plan.
3. `docs/10-plans/phase2-rgcn-trainable-retriever.md` for the implemented R-GCN trainable retriever slice.
4. `docs/10-plans/dense-ft-implementation-plan.md` for the Dense-FT implementation.
5. `docs/10-plans/trainable-stack-zero-compatibility-refactor-plan.md` for the current trainable-stack boundary.
6. `docs/20-contracts/data-contracts.md`, `docs/20-contracts/retrieval-contracts.md`, and `docs/20-contracts/model-contracts.md` for current artifact, retrieval, and model contracts.
7. `docs/10-plans/engineering-quality-brainstorm.md` for evolving engineering decisions.
