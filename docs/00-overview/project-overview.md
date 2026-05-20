# Project Overview

## Metadata

| Field | Value |
|---|---|
| Project | Execution-Provenance Graph Memory |
| Current scope | Phase 1 HotpotQA evidence-tracing retrieval |
| Primary task | Retrieve complete supporting evidence nodes and connected evidence subgraphs. |
| Primary dataset | HotpotQA distractor setting with labeled splits only. |
| Primary methods | BM25, frozen dense retrieval, graph-aware reranking. |
| Source material | `docs/archive/original-student-experiment-plan.md` |
| Current implementation plan | `docs/10-plans/phase1-real-graph-memory.md` |

## Background

This project studies memory retrieval for evidence-heavy LLM and agent workflows. Standard flat memory stores observations, documents, and tool outputs as isolated text chunks. That can retrieve semantically similar records, but it often misses the dependency chain that explains why an answer is supported.

Execution-Provenance Graph Memory represents each evidence-bearing sentence or step as a node and links nodes through typed edges such as sequential context, query overlap, entity overlap, bridge connections, tool dependency, and parameter flow.

The key research question is not only whether a retriever finds one relevant sentence. The stricter question is whether it recovers the complete evidence set and the connected evidence path behind a query.

## Core Requirements

- Convert public evidence-intensive QA data into a unified memory retrieval task format.
- Use sentence-level memory nodes for HotpotQA Phase 1.
- Evaluate evidence retrieval and evidence tracing before answer generation.
- Keep label-only fields separate from retrieval and graph-construction inputs.
- Compare flat retrieval against graph-aware retrieval under a shared ranked-result schema.
- Report node-level metrics, graph connectivity metrics, and efficiency metrics.
- Prefer fail-fast validation over silent fallback behavior.
- Keep Phase 1 narrow enough to be runnable and scientifically interpretable.

## Phase Roadmap

| Phase | Scope | Expected output |
|---|---|---|
| Phase 1 | HotpotQA + BM25 + frozen dense + graph rerank | Runnable evidence retrieval system and main metrics. |
| Phase 2 | Add Dense-FT, Memory Stream, GraphRAG, edge ablations | Main paper comparison and ablation tables. |
| Phase 3 | Add MemGPT-style memory, 2WikiMultiHopQA, tool trajectories | Generalization and agent-style provenance analysis. |
| Optional | Add MuSiQue | Harder multi-hop stress test. |

## Current Phase 1 Boundary

Phase 1 should implement:

- HotpotQA conversion into input and label artifacts.
- Typed graph construction from input-visible text.
- BM25 retrieval.
- Frozen dense retrieval.
- BM25-seeded and dense-seeded graph reranking.
- Dev-set graph parameter tuning.
- Evaluation of Recall@k, Evidence F1@k, Full Support@k, MRR, Connected Evidence Recall@k, Query-Evidence Connectivity@10, and efficiency.

Phase 1 should not implement:

- Answer generation.
- Dense fine-tuning.
- Trainable GNN retrievers.
- GraphRAG.
- Memory Stream or MemGPT-style memory.
- 2WikiMultiHopQA, MuSiQue, or tool trajectory experiments.

Those belong after Phase 1 is stable.

## Documentation Flow

Start here, then read:

1. `docs/archive/original-student-experiment-plan.md` for full source context.
2. `docs/10-plans/phase1-real-graph-memory.md` for the current implementation plan.
3. `docs/20-contracts/phase1-data-contracts.md` for exact artifact schemas.
4. `docs/10-plans/engineering-quality-brainstorm.md` for evolving engineering decisions.
