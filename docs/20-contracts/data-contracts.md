# Data contracts

Root constraint: [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Evidence workflow artifacts

| Artifact | Producer | Consumer |
| --- | --- | --- |
| `*.input.json` | dataset preparation | text projection, GraphRAG, EvidenceGraph construction |
| `*.labels.json` | dataset preparation | evaluation and training only |
| `*.evidence_graphs.json` | `scripts/build_evidence_graphs.py` | R-GCN training/retrieval and graph-dependent evaluation |
| `*.pairs.json` | `scripts/build_train_pairs.py` | Dense-FT or R-GCN training |
| ranked result JSON | `scripts/run_retrieval.py` | evaluation |

EvidenceGraph construction consumes input-visible question/candidate fields only. Labels never enter retrieval graph construction.

## TRAJECT-Bench retrieval artifacts

TRAJECT-Bench uses the same `*.input.json`, `*.labels.json`, and ranked-result workflow, but its raw source is a directory rather than one split file. The dataset configuration maps workflow split names to published benchmark partitions:

| Workflow split | Published files |
| --- | --- |
| `train` | `parallel/*/simple_ver.json` |
| `dev` | `parallel/*/hard_ver.json` |
| `test` | `sequential/*/traj_query.json` and `sequential/Travel/simple_ver.json` |

This is an operational mapping only; TRAJECT-Bench does not publish an official training split.

Each input record contains the query and a domain-local candidate pool built exclusively from `tools/<domain>.json`. Exact duplicate catalog names are canonicalized deterministically. Candidate IDs are stable hashes of exact tool names. Candidate text may contain only catalog-visible name, provider, API, descriptions, and parameter schemas. Catalog-declared tool connections may appear as input-visible directed edges.

Each label record contains the distinct gold tool IDs, the ordered gold sequence with repeated calls preserved, final answer, and derived adjacent dependency edges. Gold-call descriptions, arguments, outputs, final answer, and trajectory order are forbidden from ranking inputs and graph construction. A query referencing a tool absent from its public domain catalog is rejected before seeded sampling; the adapter never invents a candidate from gold fields.

The generic Recall, Evidence F1, Full Support, and MRR fields are interpreted here as tool-retrieval metrics. They are not the benchmark's end-to-end Exact Match, Usage, Trajectory Satisfaction, or Solution Accuracy metrics.

## Execution provenance domain

`ExecutionProvenanceGraph` contains typed nodes and edges supplied by a future dataset adapter. Core node types are Task, Agent, ToolCall, ToolOutput, and Answer. Core transitions include `invokes`, `returns`, `feeds`, and `grounds`; `feeds` requires field-binding evidence. `precedes` expresses chronology only. Claim, Verification, contradiction, invalidation, and impact nodes/edges are accepted only when the source provides them.

The current repository defines this domain contract and retriever but no concrete adapter that emits a native `ExecutionProvenanceGraph`. The TRAJECT-Bench integration above deliberately targets leakage-safe offline tool retrieval instead.
