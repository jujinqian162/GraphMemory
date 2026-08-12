# Architecture

```text
dataset selection
  -> concrete retrieval request
  -> one explicit static method dispatch
  -> ranking + optional native_trace
```

## Package map

```text
graph_memory/
  contracts/          low-level scalar/model primitives and shared IDs
  datasets/           dataset-owned Pydantic records and direct selection
  trajectories/       dataset-neutral ordered Agent execution events
  graphs/             evidence plus query-independent provenance graphs
  query_synthesis/    internal motif planning and v7 authoring contracts
  embeddings/         frozen dense encoders
  retrieval/          requests, flat and GraphRAG methods
  models/             dense_finetune, shared graph_retriever R-GCN runtime
  registry/           static retrieval settings and construction dispatch
  stages/             prepare, graphs, pairs, models, retrieve, evaluate
  experiment/         Hydra job, Prefect Flow, artifacts, tracking
  evaluation/         Pydantic labels, metric rows, suites, and tables
  training_pairs/     Pydantic pair contracts and pair sampling
  analysis/           post-run aggregation helpers
```

## Ownership rules

- `datasets/` owns closed Pydantic source/prepared records and direct dataset selection; it does not own cross-dataset canonical trajectory semantics.
- `trajectories/` owns canonical ordered message/tool event contracts and stable source-span anchors.
- `graphs/contracts.py` owns the closed `EvidenceGraph` model; `graphs/provenance/` owns the separate query-independent provenance graph, deterministic builder, and logical output-dependency projection shared by motifs, pair adaptation, and provenance retrieval.
- `query_synthesis/provenance/` owns internal motif/source planning plus the four-field v7 authoring contract and deterministic handle/span helpers. It exposes no legacy answer/support query labels or generation-provenance envelope.
- `retrieval/requests/` owns the closed request union; `retrieval/results.py` owns ranked results and request/result aggregates.
- GraphRAG owns its text-derived entity graph end-to-end. It splits retrieval candidates into private small text units, extracts deterministic noun/structured phrases, builds a pruned co-occurrence graph, and projects query-personalized PageRank scores back to the original candidates. GraphRAG never receives the native provenance graph.
- `models/graph_retriever/` owns one R-GCN encoder, scorer, disconnected-union batcher, optimizer loop, and checkpoint format. Evidence and provenance use domain-specific tensorization adapters around that shared runtime.
- `registry/retrieval_builders.py` is a closed static dispatch for the eight repository-owned methods. It has no runtime registry, plugin metadata, request/family compatibility table, or builder registration layer.
- `experiment/` schedules stages from real artifact dependencies of the selected method/variant.
- `training_pairs/`, `evaluation/`, and each model package own their Pydantic artifact/config contracts. There is no central validation package.
- Root `io.py` / `compat.py` are thin ports only.

## Contract and artifact rule

A scientific field is declared once, in the Pydantic model owned by its domain. Custom scientific invariants are model validators on that model or on a typed aggregate joining related models. Contributors must not add parallel allowed-field sets, central `validate_*` functions, or hand-written serializers.

Project-owned JSON is validated immediately at the consuming stage boundary with the owning model or a cached `TypeAdapter`. Inside the pipeline, code passes frozen model instances and uses attributes. Publication uses `model_dump(mode="json", by_alias=True)` (directly or through the Pydantic-aware IO port). Training-pair and dev-input aggregates are constructed before encoder/model loading or optimizer creation, so malformed artifacts fail before expensive work.

## Current graphs

1. **EvidenceGraph** — question/evidence nodes and evidence relations; required only by the two evidence R-GCN methods.
2. **GraphRAG entity graph** — rebuilt from candidates inside the method; never an `EvidenceGraph` artifact.
3. **ProvenanceGraph** — one query-independent execution graph per canonical trajectory. The v1 core contains tool calls, tool outputs, argument/output content chunks, and resource artifacts with exact source spans.

`ProvenanceGraph` is not an `EvidenceGraph`, has no persisted query node or label-derived edge weights, and remains independent of query/label artifacts. ISETrace preparation persists it beside candidates, exact spans, and natural-query evaluation metadata. BM25 and GraphRAG consume the shared lossless flat trajectory chunks. Dense and Dense-FT expose `variant=flat|provenance_unit`: `flat` retains that same chunk view, while the ISETrace-only `provenance_unit` control ranks source-backed provenance content units through `TextRankingRequest` without receiving either graph domain. Dense-FT derives positives only through exact-span overlap in the selected view. `provenance_path` and `provenance_rgcn` consume the native graph directly; provenance R-GCN may use either the registered base encoder or the cached provenance-unit Dense-FT checkpoint as its seed provider. Shared R-GCN scoring preserves the seed cosine score and learns a zero-initialized graph residual; `wo_graph` is an exact seed-score passthrough rather than a replacement MLP. The R-GCN tensorizer appends an ephemeral disconnected `q` node and maps only the fixed physical relation vocabulary to uniform forward/reverse message edges. It excludes `temporal.precedes` and metadata-derived numeric features. No conversion to `EvidenceGraph`, compatibility alias, edge head, or second graph-convolution stack exists. The LLM-authoring script remains an offline utility and not a label authority.

## Runtime boundary

```text
experiment/run.py  (one method, optional one R-GCN variant)
  -> graph_memory/experiment/workflow.py  (one Prefect Flow)
  -> graph_memory/stages/*
  -> static retrieval dispatch + concrete domain packages
  -> runs/<name>/  (output-only reports + asset refs)
  -> data/processed/  (reusable scientific assets)
```

One Hydra job = one Flow run = one MLflow run. Compare methods/variants by launching independent jobs that share Prefect cache storage.
