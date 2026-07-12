# Execution-Provenance Graph Memory

Current scope: request-first evidence retrieval on HotpotQA, 2WikiMultiHopQA, and MuSiQue-Ans with flat baselines, graph reranking, Dense-FT, a trainable R-GCN graph retriever, and a Dense-FT-seeded R-GCN retriever.

The runnable stack includes dataset-specific leakage-safe preparation, typed graph construction, BM25, frozen dense retrieval, BM25- and dense-seeded graph reranking, Dense-FT, checkpoint-backed `dense_rgcn_graph_retriever` training and retrieval, and `dense_ft_rgcn_graph_retriever` training seeded from the Dense-FT checkpoint. HotpotQA uses sentence-level evidence, 2WikiMultiHopQA uses sentence-level evidence and dependency supervision, and MuSiQue-Ans uses paragraph-level evidence and decomposition-derived dependency supervision. R-GCN edge/model ablations and unified result aggregation are implemented. Hydra YAML and closed Pydantic models are the only experiment configuration contract; local typed state remains authoritative while MLflow mirrors each Hydra job as one summary parent plus one child per selected baseline.

The original Phase 2 paper matrix is not complete yet. Memory Stream and GraphRAG-style baselines are still missing, and HotpotQA does not provide gold dependency paths, so its `Path Recall@10` and `Edge Recall@10` remain `N/A`. MemGPT-style memory, answer generation, MuSiQue-Full answerability/sufficiency evaluation, and tool-trajectory provenance experiments remain later work.

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
- MuSiQue data and run guide: `docs/40-operations/musique.md`
- Implementation handoff: `docs/40-operations/implementation-handoff.md`

## Quick Start

Install dependencies with your preferred Python 3.12 environment manager, then run tests:

```powershell
uv run pytest tests -q
```

Use the Hydra experiment entrypoints for normal runs:

```powershell
uv run python experiment/plan.py name=quick_valid_100 profile=quick
uv run python experiment/run.py name=quick_valid_100 profile=quick
uv run python experiment/status.py name=quick_valid_100
uv run python experiment/inspect.py kind=methods
uv run python experiment/reset.py name=quick_valid_100
```

Run artifacts are isolated under `runs/<experiment_name>/`. Multirun leaves use concise selectors such as `0_num_layers=2`; `experiment/inspect.py kind=jobs name=<sweep>` lists them. Planning is implicit on both `plan` and `run`; low-level scripts accept exactly `--config <resolved-stage-yaml>`. See `docs/40-operations/commands.md` for overrides, status, reset, multirun, parent/baseline artifact ownership, and MLflow Compare Runs.
