# Graph Memory Documentation

Maintained reference docs only. Historical plans, brainstorms, and paper sources live elsewhere.

| Path | Owns |
|---|---|
| [`00-overview/project-overview.md`](00-overview/project-overview.md) | Domains, methods, research boundary |
| [`30-design/architecture.md`](30-design/architecture.md) | Package ownership, request/graph separation, workflow |
| [`20-contracts/data-contracts.md`](20-contracts/data-contracts.md) | Artifacts and graph/input contracts |
| [`20-contracts/retrieval-contracts.md`](20-contracts/retrieval-contracts.md) | Method matrix, requests, ranking/trace surface |
| [`40-operations/commands.md`](40-operations/commands.md) | How to run jobs |
| [`40-operations/reproducibility.md`](40-operations/reproducibility.md) | Cache, run layout, delivery |
| [`40-operations/twowiki-provenance.md`](40-operations/twowiki-provenance.md) | Synthetic provenance benchmark |
| [`40-operations/stateless-graph-retrieval.md`](40-operations/stateless-graph-retrieval.md) | GraphRAG + EPGM |
| [`40-operations/implementation-handoff.md`](40-operations/implementation-handoff.md) | Extension points |
| [`configs/README.md`](configs/README.md) | Hydra config layout |
| `raw/` | Paper TeX and imported drafts (not edited by this index) |

Truth for schemas and behavior is code under `graph_memory/` plus the contracts above. Do not duplicate field lists across docs.
