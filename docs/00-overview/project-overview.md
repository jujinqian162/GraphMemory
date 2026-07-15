# Project overview

The project studies whether explicit graph structure improves recovery of complete evidence or execution paths beyond flat semantic retrieval. The governing implementation plan is [`execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Supported domains

| Domain | Inputs | Methods |
| --- | --- | --- |
| Evidence retrieval | text candidates; optional `EvidenceGraph` for R-GCN | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | flat text requests or native `ExecutionProvenanceRankingRequest` | BM25, Dense, GraphRAG, stateless Execution-Provenance Retriever, Provenance R-GCN |

`EvidenceGraph` contains question/evidence nodes and evidence relations. `ExecutionProvenanceGraph` contains Task, Agent, ToolCall, ToolOutput, and Answer nodes plus typed execution/dataflow edges. GraphRAG owns a third, private entity graph and does not consume either public graph artifact.

HotpotQA, standard 2Wiki, and MuSiQue remain evidence-retrieval datasets. `twowiki_provenance` is a separately generated synthetic benchmark whose recoverable gold support chain is embedded among structurally matched typed branches.

## Research boundary

- Flat methods establish lexical and semantic baselines.
- GraphRAG tests method-owned entity propagation without a dataset graph artifact.
- The two R-GCN methods learn independent evidence-node scores from `EvidenceGraph` inputs.
- The stateless provenance retriever combines semantic seeds with directed typed beam expansion, field binding, grounding, and revision invalidation.
- Provenance R-GCN performs relation-aware message passing across connector nodes, ranks ToolOutput candidates from materialized easy/BM25/dense pairs, and scores contracted logical edges independently; it has no beam decoder or dynamic oracle.
- Retrieval outputs preserve complete rankings. Native traces contain only edges and paths actually used by the method.
