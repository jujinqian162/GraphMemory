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
  trajectories/       dataset-neutral ordered Agent execution events
  graphs/             evidence plus query-independent provenance graphs
  query_synthesis/    motif specs and leakage-safe query verbalization
  embeddings/         frozen dense encoders
  retrieval/          requests, flat and GraphRAG methods
  models/             dense_finetune, graph_retriever (evidence R-GCN)
  registry/           method IDs, settings, builders
  stages/             prepare, graphs, pairs, models, retrieve, evaluate
  experiment/         Hydra job, Prefect Flow, artifacts, tracking
  evaluation/         Pydantic labels, metric rows, suites, and tables
  training_pairs/     Pydantic pair contracts and pair sampling
  analysis/           post-run aggregation helpers
```

## Ownership rules

- `datasets/` owns closed Pydantic source/prepared records and dataset adapters; it does not own cross-dataset canonical trajectory semantics.
- `trajectories/` owns canonical ordered message/tool event contracts and stable source-span anchors.
- `graphs/contracts.py` owns the closed `EvidenceGraph` model; `graphs/provenance/` owns the separate query-independent provenance graph and its deterministic builder.
- `query_synthesis/provenance/` owns motif supervision, query-intent labels, the versioned template catalog, and safe-slot verbalization. None of those labels enter graph construction.
- `retrieval/requests/` owns the closed request union; `retrieval/results.py` owns ranked results and request/result aggregates.
- GraphRAG owns its entity graph end-to-end; Registry may assemble mentions before the method runs.
- `models/graph_retriever/` owns the evidence R-GCN train/infer implementation.
- `registry/` owns public IDs, request/family compatibility, and builders. Workflow scheduling stays in `experiment/workflow.py`, not Registry metadata.
- `experiment/` schedules stages from real artifact dependencies of the selected method/variant.
- `training_pairs/`, `evaluation/`, and each model package own their Pydantic artifact/config contracts. There is no central validation package.
- Root `io.py` / `compat.py` are thin ports only.

## Contract and artifact rule

A scientific field is declared once, in the Pydantic model owned by its domain. Custom scientific invariants are model validators on that model or on a typed aggregate joining related models. Contributors must not add parallel allowed-field sets, central `validate_*` functions, or hand-written serializers.

Project-owned JSON is validated immediately at the consuming stage boundary with the owning model or a cached `TypeAdapter`. Inside the pipeline, code passes frozen model instances and uses attributes. Publication uses `model_dump(mode="json", by_alias=True)` (directly or through the Pydantic-aware IO port). Training-pair and dev-input aggregates are constructed before encoder/model loading or optimizer creation, so malformed artifacts fail before expensive work.

## Current graphs

1. **EvidenceGraph** — question/evidence nodes and evidence relations; required only by the two evidence R-GCN methods.
2. **GraphRAG entity graph** — rebuilt from candidates inside the method; never an `EvidenceGraph` artifact.
3. **ProvenanceGraph** — one query-independent execution graph per canonical trajectory. The v1 core contains `execution.tool_call`, `execution.tool_output`, and `resource.artifact`; future namespaced semantic annotation kinds are representable through source spans but are not emitted by the core builder.

`ProvenanceGraph` is currently a domain artifact only. It is not an `EvidenceGraph`, has no query node or edge weights, and is not yet connected to Registry, Hydra, Prefect, retrieval, training, or evaluation. No compatibility alias translates ISETrace into the deleted Task/Answer schema.

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
