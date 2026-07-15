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

## Synthetic 2Wiki execution-provenance artifacts

`scripts/data/convert_2wiki_to_execution_provenance.py` creates a separately named `twowiki_provenance` dataset. Each raw record has disjoint `ranking` and `label` objects. Ranking owns the question, ToolOutput candidates, and a typed Task/Agent/ToolCall/ToolOutput graph; label owns the answer, ordered gold output IDs, and contracted dependency edge. Standard `twowiki` remains unchanged.

The graph is explicitly synthetic. It is built from a recoverable two-evidence gold chain, then completed with structurally matched query-relevant branches. Every candidate output has a paired call, `returns` edge, and equal public `feeds` degree. Gold flags, final answers, source support annotations, and gold-only topology are forbidden from ranking input. Source hashes, seed, filtering counts, split policy, and structural statistics are recorded in generated manifest files.

## Execution provenance domain

`ExecutionProvenanceGraph` contains typed nodes and edges supplied by a compatible dataset adapter. Core node types are Task, Agent, ToolCall, ToolOutput, and Answer. Core transitions include `invokes`, `returns`, `feeds`, and `grounds`; `feeds` requires field-binding evidence. `precedes` expresses chronology only. Claim, Verification, contradiction, invalidation, and impact nodes/edges are accepted only when the source provides them.

The repository has one concrete adapter for this contract: the explicitly synthetic `twowiki_provenance` benchmark.
