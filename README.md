# Execution-Provenance Graph Memory

Current scope: HotpotQA evidence-tracing retrieval with Phase 1 baselines and the Phase 2 R-GCN trainable retriever slice.

Phase 1 is the stable runnable baseline: HotpotQA conversion, typed graph construction, BM25, frozen dense retrieval, graph reranking, dev tuning, and retrieval evaluation. The current Phase 2 implementation adds a checkpoint-backed `dense_rgcn_graph_retriever` path with train-pair construction, R-GCN training, checkpoint loading, and standard ranked-result evaluation. The broader Phase 2 paper scope is not complete yet: Dense-FT, Memory Stream, GraphRAG, full ablation runs, MemGPT-style memory, answer generation, and additional datasets remain later work.

Start here:

- Documentation map: `docs/README.md`
- Phase 1 implementation plan: `docs/10-plans/phase1-real-graph-memory.md`
- Phase 2 R-GCN trainable retriever plan: `docs/10-plans/phase2-rgcn-trainable-retriever.md`
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
python scripts/experiment.py init quick_valid_100 --profile quick --methods bm25,dense,bm25_graph_rerank,dense_graph_rerank,dense_rgcn_graph_retriever
python scripts/experiment.py plan quick_valid_100 --stages prepare,graphs,retrieve,evaluate,aggregate
python scripts/experiment.py run quick_valid_100 --stages prepare,graphs,retrieve,evaluate,aggregate
python scripts/experiment.py status quick_valid_100
```

Run artifacts are isolated under `runs/<experiment_name>/`. The low-level command sequence is maintained in `docs/40-operations/commands.md` for contract review and debugging.
