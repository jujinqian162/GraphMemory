## Why

The legacy RQ2 stack has been removed because its 2Wiki-derived graph was gold-conditioned and did not represent native Agent execution. ISETrace now provides revision-pinned raw trajectories with real tool calls and outputs, but the repository has no canonical trajectory domain, query-independent execution graph, or leakage-safe motif/query construction boundary.

The next increment should establish those scientific contracts before any new retriever, training workflow, LLM-generated benchmark, or paper experiment is implemented.

## What Changes

- Add a dataset-neutral canonical trajectory domain with ordered message/tool events, stable source references, typed tool definitions, and strict call/output pairing.
- Add a streaming ISETrace adapter that converts raw JSONL records into canonical trajectories and emits only a compact ingestion summary rather than a separate audit subsystem.
- Add a query-independent, namespaced provenance multigraph with a minimal execution vocabulary: tool calls, tool outputs, artifacts, and native/deterministic execution relations.
- Keep graph node/relation identifiers extensible so later NLP annotators can add namespaced semantic nodes such as claims and decisions with source-span provenance, without changing the canonical trajectory or conditioning graph construction on a query.
- Add deterministic provenance motif extraction and schema-aware template query synthesis with multiple query intents and a versioned, diverse template catalog.
- Keep motif labels separate from graph and query inputs. Do not add model training, experiment workflow integration, LLM query generation, or General-AgentBench support in this change.

## Capabilities

### New Capabilities

- `canonical-agent-trajectory`: ISETrace records can be normalized into a reusable, dataset-neutral ordered event contract.
- `query-independent-provenance-graph`: canonical trajectories can be projected into stable execution graphs that contain no query or gold supervision.
- `provenance-motif-query-synthesis`: high-confidence graph motifs can be converted into leakage-safe pseudo-query examples through diverse deterministic templates.

### Modified Capabilities

None. The maintained evidence-retrieval workflow remains unchanged.

## Impact

- New production packages under `graph_memory/trajectories/`, `graph_memory/datasets/isetrace/`, `graph_memory/graphs/provenance/`, and `graph_memory/query_synthesis/provenance/`.
- New focused tests for raw adaptation, canonical invariants, graph independence, extension points, motif extraction, and query diversity.
- Maintained architecture/contract documentation will describe the new domains while continuing to mark ISETrace as outside the experiment registry until a later change.
- No external dependency, method ID, Hydra dataset, Prefect stage, checkpoint, metric, or paper-result change is included.
