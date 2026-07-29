# GraphRAG and EPGM

Covers non-trained graph methods only. No dataset migration.

## GraphRAG

Dense ranks all candidates first. The method derives typed title/body entity evidence with the same frozen encoder, then applies local stable insertion of bridge partners. If nothing moves, the ranking is exactly Dense. Entity structure is method-private and never an EvidenceGraph.

## EPGM (`execution_provenance_retriever`)

One public method id; reported default `ppr_steiner`.

| Variant | Role |
|---|---|
| `ppr_steiner` | Default: query-conditioned typed PPR + budgeted connected subgraph |
| `typed_beam` | Historical diagnostic |
| `dependency_path` | Historical diagnostic |

Default sketch:

1. Dense ranks candidates.
2. Query is compared to schema-owned natural-language relation descriptions.
3. Stored edges become directed arcs (relation affinity, type prior, direction, degree, recorded weight); strengths normalize per source.
4. PPR diffuses a Dense-derived teleport over candidates and connectors.
5. Budgeted connected selection keeps ≤ `top_k` candidates with positive marginal prize.
6. Collapsed logical edges for evaluation are `feeds`; native relations stay in `native_trace`.

No positive multi-candidate connection ⇒ ranking equals Dense.

The previous synthetic 2Wiki provenance adapter has been removed. This method becomes runnable again after the ISETrace adapter supplies native execution-provenance requests.

Variant + frozen behavior fingerprint enter Prefect ranking identity and `graph_memory.variant` tags.
