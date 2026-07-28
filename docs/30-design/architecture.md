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
  contracts/          low-level scalar/model primitives and shared IDs
  datasets/           dataset-owned Pydantic records and projectors
  graphs/             Pydantic evidence/provenance contracts and construction
  embeddings/         frozen dense encoders
  retrieval/          requests, flat / graphrag / epgm methods
  models/             dense_finetune, graph_retriever (R-GCN)
  registry/           method IDs, settings, builders
  stages/             prepare, graphs, pairs, models, retrieve, evaluate
  experiment/         Hydra job, Prefect Flow, artifacts, tracking
  evaluation/         Pydantic labels, metric rows, suites, and tables
  training_pairs/     Pydantic pair contracts and pair sampling
  analysis/           post-run aggregation helpers
```

## Ownership rules

- `datasets/` owns closed Pydantic source/prepared records and projects them into consumer-specific requests; it does not invent cross-domain graphs.
- `graphs/contracts.py` owns the closed `EvidenceGraph` model.
- `graphs/provenance/` owns the closed `ExecutionProvenanceGraph` and field-binding models.
- `retrieval/requests/` owns the closed request union; `retrieval/results.py` owns ranked results and request/result aggregates.
- GraphRAG owns its entity graph end-to-end; Registry may assemble mentions before the method runs.
- EPGM lives under `retrieval/methods/epgm/`.
- `models/graph_retriever/` owns node-wise R-GCN train/infer for both evidence and provenance families (separate method IDs and pair protocols).
- `registry/` owns public IDs, request/family compatibility, and builders. Workflow scheduling stays in `experiment/workflow.py`, not Registry metadata.
- `experiment/` schedules stages from real artifact dependencies of the selected method/variant.
- `training_pairs/`, `evaluation/`, and each model package own their Pydantic artifact/config contracts. There is no central validation package.
- Root `io.py` / `compat.py` are thin ports only.

## Contract and artifact rule

A scientific field is declared once, in the Pydantic model owned by its domain. Custom scientific invariants are model validators on that model or on a typed aggregate joining related models. Contributors must not add parallel allowed-field sets, central `validate_*` functions, or hand-written serializers.

Project-owned JSON is validated immediately at the consuming stage boundary with the owning model or a cached `TypeAdapter`. Inside the pipeline, code passes frozen model instances and uses attributes. Publication uses `model_dump(mode="json", by_alias=True)` (directly or through the Pydantic-aware IO port). Training-pair and dev-input aggregates are constructed before encoder/model loading or optimizer creation, so malformed artifacts fail before expensive work.

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
