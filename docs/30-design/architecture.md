# Architecture

```text
dataset adapter
  -> concrete retrieval request
  -> Registry validation
  -> method builder
  -> ranking + optional native_trace
```

## Package map

```text
graph_memory/
  contracts/          shared types, graphs, ranking, metrics
  datasets/           HotpotQA, 2Wiki, MuSiQue, twowiki_provenance
  graphs/             evidence construction; provenance values
  embeddings/         frozen dense encoders
  retrieval/          requests, flat / graphrag / epgm methods
  models/             dense_finetune, graph_retriever (R-GCN)
  registry/           method IDs, settings, builders
  stages/             prepare, graphs, pairs, models, retrieve, evaluate
  experiment/         Hydra job, Prefect Flow, artifacts, tracking
  evaluation/         metrics and tables
  training_pairs/     pair sampling
  validation/         fail-fast validators
  analysis/           post-run aggregation helpers
```

## Ownership rules

- `datasets/` projects source records into consumer-specific requests; it does not invent cross-domain graphs.
- `contracts/graphs.py` owns traditional `EvidenceGraph`.
- `graphs/provenance/` owns `ExecutionProvenanceGraph` values and validation.
- `retrieval/requests/` owns the closed request union.
- GraphRAG owns its entity graph end-to-end; Registry may assemble mentions before the method runs.
- EPGM lives under `retrieval/methods/epgm/`.
- `models/graph_retriever/` owns node-wise R-GCN train/infer for both evidence and provenance families (separate method IDs and pair protocols).
- `registry/` owns public IDs, request/family compatibility, and builders. Workflow scheduling stays in `experiment/workflow.py`, not Registry metadata.
- `experiment/` schedules stages from real artifact dependencies of the selected method/variant.
- Root `io.py` / `compat.py` are thin ports only.

## Three graphs, no translation

1. **EvidenceGraph** — question/evidence nodes and evidence relations; required only by the two evidence R-GCN methods.
2. **GraphRAG entity graph** — rebuilt from candidates inside the method; never an `EvidenceGraph` or provenance graph.
3. **ExecutionProvenanceGraph** — Task / Agent / ToolCall / ToolOutput / Answer nodes with typed execution and dataflow edges (`invokes`, `returns`, `feeds`, `grounds`, optional chronology). Carried on the request.

No compatibility alias converts one graph domain into another. Flat and GraphRAG jobs must not schedule EvidenceGraph construction.

## Runtime boundary

```text
experiment/run.py  (one method, optional one R-GCN variant)
  -> graph_memory/experiment/workflow.py  (one Prefect Flow)
  -> graph_memory/stages/*
  -> registry builders + domain packages
  -> runs/<name>/  (output-only reports + asset refs)
  -> data/processed/  (reusable scientific assets)
```

One Hydra job = one Flow run = one MLflow run. Compare methods/variants by launching independent jobs that share Prefect cache storage.
