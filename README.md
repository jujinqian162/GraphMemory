# Execution-Provenance Graph Memory

Current scope: HotpotQA evidence-tracing retrieval with flat baselines, graph reranking, Dense-FT, and a trainable R-GCN graph retriever.

The runnable HotpotQA stack includes leakage-safe data preparation, typed graph construction, BM25, frozen dense retrieval, BM25- and dense-seeded graph reranking, Dense-FT, and checkpoint-backed `dense_rgcn_graph_retriever` training and retrieval. R-GCN edge/model ablations and unified result aggregation are implemented. Trainable methods use strict current-only method configs, stage configs, manifests, checkpoints, and model metadata; old trainable artifacts are not migrated.

The original Phase 2 paper matrix is not complete yet. Memory Stream and GraphRAG-style baselines are still missing, and HotpotQA does not provide gold dependency paths, so `Path Recall@10` and `Edge Recall@10` remain `N/A`. MemGPT-style memory, answer generation, 2WikiMultiHopQA, MuSiQue, and tool-trajectory provenance experiments remain later work.

Start here:

- Documentation map: `docs/README.md`
- Phase 1 implementation plan: `docs/10-plans/phase1-real-graph-memory.md`
- Phase 2 R-GCN trainable retriever plan: `docs/10-plans/phase2-rgcn-trainable-retriever.md`
- Dense-FT implementation plan: `docs/10-plans/dense-ft-implementation-plan.md`
- Current trainable-stack refactor plan: `docs/10-plans/trainable-stack-zero-compatibility-refactor-plan.md`
- Data contracts: `docs/20-contracts/data-contracts.md`
- Retrieval contracts: `docs/20-contracts/retrieval-contracts.md`
- Model contracts: `docs/20-contracts/model-contracts.md`
- Architecture: `docs/30-design/architecture.md`
- Command runbook: `docs/40-operations/commands.md`
- Implementation handoff: `docs/40-operations/implementation-handoff.md`

## Quick Start

Install dependencies with your preferred Python 3.12 environment manager, then run tests:

```powershell
uv run pytest tests -q
```

Use the experiment runner for normal runs:

```powershell
python scripts/experiment.py init quick_valid_100 --profile quick --methods bm25,dense,bm25_graph_rerank,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft
python scripts/experiment.py plan quick_valid_100 --stages prepare,graphs,pairs,train,retrieve,evaluate,aggregate
python scripts/experiment.py run quick_valid_100 --stages prepare,graphs,pairs,train,retrieve,evaluate,aggregate
python scripts/experiment.py status quick_valid_100
```

Run artifacts are isolated under `runs/<experiment_name>/`. The low-level command sequence is maintained in `docs/40-operations/commands.md` for contract review and debugging.
