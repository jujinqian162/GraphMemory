# Execution-Provenance Graph Memory

Current scope: Phase 1 HotpotQA evidence-tracing retrieval.

Phase 1 is the minimum runnable version: HotpotQA conversion, typed graph construction, BM25, frozen dense retrieval, graph reranking, dev tuning, and retrieval evaluation. Dense-FT, Memory Stream, GraphRAG, MemGPT-style memory, answer generation, and additional datasets are later-phase work.

Start here:

- Documentation map: `docs/README.md`
- Phase 1 implementation plan: `docs/10-plans/phase1-real-graph-memory.md`
- Data contracts: `docs/20-contracts/phase1-data-contracts.md`
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
python scripts/experiment.py init quick_valid_100 --profile quick --methods bm25
python scripts/experiment.py plan quick_valid_100 --stages prepare,graphs,retrieve,evaluate,aggregate
python scripts/experiment.py run quick_valid_100 --stages prepare,graphs,retrieve,evaluate,aggregate
python scripts/experiment.py status quick_valid_100
```

Run artifacts are isolated under `runs/<experiment_name>/`. The low-level command sequence is maintained in `docs/40-operations/commands.md` for contract review and debugging.
