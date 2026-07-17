# Data contracts

Root constraint: [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Evidence workflow artifacts

| Artifact | Producer | Consumer |
| --- | --- | --- |
| `*.input.json` | dataset preparation | text projection, GraphRAG, EvidenceGraph construction |
| `*.labels.json` | dataset preparation | evaluation and training only |
| prepared task/label payloads | `graph_memory.stages.prepare` | graph, pair, ranking, training, and evaluation Tasks |
| evidence graph payload | `graph_memory.stages.graphs` | evidence R-GCN training/ranking and graph-dependent evaluation |
| training pair payload | `graph_memory.stages.pairs` | Dense-FT or R-GCN training |
| ranked prediction payload | `graph_memory.stages.retrieve` | evaluation |

Reusable payloads are immutable processed assets below `data/processed/`. Prefect persists small typed references containing kind, URI, digest, manifest URI, origin, size/shape, and metadata; it does not serialize large scientific payloads into the run output. `runs/<name>/assets/manifest.yaml` is a delivery reference surface, not a computation input.

EvidenceGraph construction consumes input-visible question/candidate fields only. Labels never enter retrieval graph construction.

## Synthetic 2Wiki execution-provenance artifacts

`scripts/data/convert_2wiki_to_execution_provenance.py` creates a separately named `twowiki_provenance` dataset. Each raw record has disjoint `ranking` and `label` objects. Ranking owns the question, ToolOutput candidates, and a typed Task/Agent/ToolCall/ToolOutput graph; label owns the answer, ordered gold output IDs, and contracted dependency edge. Standard `twowiki` remains unchanged.

The graph is explicitly synthetic. It is built from a recoverable two-evidence gold chain, then completed with structurally matched query-relevant branches. Every candidate output has a paired call, `returns` edge, and equal public `feeds` degree. Gold flags, final answers, source support annotations, and gold-only topology are forbidden from ranking input. Source hashes, seed, filtering counts, split policy, and structural statistics are recorded in generated manifest files.

## Execution provenance domain

`ExecutionProvenanceGraph` contains typed nodes and edges supplied by a compatible dataset adapter. Core node types are Task, Agent, ToolCall, ToolOutput, and Answer. Core transitions include `invokes`, `returns`, `feeds`, and `grounds`; `feeds` requires field-binding evidence. `precedes` expresses chronology only. Claim, Verification, contradiction, invalidation, and impact nodes/edges are accepted only when the source provides them.

The repository has one concrete adapter for this contract: the explicitly synthetic `twowiki_provenance` benchmark.
