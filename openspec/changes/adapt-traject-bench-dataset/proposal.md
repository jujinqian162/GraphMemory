## Why

The repository cannot currently evaluate retrieval methods on TRAJECT-Bench because its official data is organized as shared tool catalogs plus parallel and sequential tool-use queries rather than per-query evidence paragraphs. Adding a dataset-owned adapter now makes the benchmark's offline tool-retrieval problem reproducible without coupling the core retrieval runtime to upstream agent execution code or leaking gold tool descriptions into candidate pools.

## What Changes

- Add typed TRAJECT-Bench records, parsers, canonical tool-catalog conversion, request projectors, and validators.
- Map the public benchmark partitions to the experiment workflow as `train=parallel/simple`, `dev=parallel/hard`, and `test=sequential`, while documenting that these are workflow partitions and not official training splits.
- Use domain tool catalogs as the candidate pool, collapse duplicate catalog entries deterministically, preserve repeated gold calls in label metadata, and reject queries whose gold tool is absent from the public catalog.
- Project candidate tools to the existing text-ranking, EvidenceGraph, and execution-provenance request boundaries; catalog-declared tool connections may become input-visible graph edges, while query-specific gold order remains label-only.
- Extend dataset source bindings so the workflow can declare a raw directory as an external input instead of pretending every dataset source is a file.
- Add TRAJECT-Bench experiment configuration, tests, operational documentation, and pinned Hugging Face/GitHub download commands for fast BM25, Dense, and execution-provenance baselines.
- Report retrieval-only metrics and scope explicitly; the change does not execute tools, call an LLM, or claim the benchmark's end-to-end agent metrics.

## Capabilities

### New Capabilities

- `traject-bench-dataset-adapter`: Typed ingestion, leakage-safe tool-catalog projection, validation, retrieval requests, workflow partitions, and baseline evaluation for TRAJECT-Bench.
- `dataset-source-artifacts`: Explicit file-versus-directory raw dataset source bindings in resolved experiment plans and stage status checks.

### Modified Capabilities


## Impact

The change affects `graph_memory/datasets`, dataset selection and validation dispatch, experiment config/planning models, the prepare stage, Hydra dataset configuration, focused tests, and dataset operations documentation. It adds no runtime dependency on the upstream TRAJECT-Bench repository and no model-provider or tool-execution dependency; data can be downloaded with the Hugging Face CLI or an official Git clone.
