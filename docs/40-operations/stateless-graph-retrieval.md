# Stateless GraphRAG retrieval

Covers the maintained non-trained entity-graph baseline.

## FastGraphRAG-style retrieval

The implementation is a deterministic, retrieval-only adaptation of Microsoft FastGraphRAG. It reproduces the entity-graph retrieval principle while preserving this repository's candidate-ranking and exact-source-span contracts; it does not claim to reproduce GraphRAG's LLM community reports or answer-generation layer.

For each unique candidate collection, the method:

1. splits candidate text into private 100-token text units with 20-token overlap;
2. extracts English noun-like phrases and structured identifiers without downloading NLP corpora at runtime;
3. retains repeated, connected entities and creates entity-entity edges when phrases co-occur in one text unit;
4. applies the frequency-scaled positive PMI weighting and graph-pruning defaults used by FastGraphRAG-style indexing;
5. links exact/lexical query phrases and the top frozen-Dense entity matches into personalization seeds;
6. runs deterministic Personalized PageRank over the entity graph;
7. projects entity mass through text units to the original retrieval candidates and softly fuses min-max-normalized graph and Dense scores.

The graph-index text units are private indexing units, not ranked evidence units. BM25, Dense, and GraphRAG continue to rank the same dataset-provided candidates, so exact source-span evaluation remains comparable. Graph construction is query-independent and reused for requests with the same candidate IDs.

The default config is [`configs/method/graphrag.yaml`](../../configs/method/graphrag.yaml). Its `fast_graphrag_ppr` trace records graph size, query-linked and seeded entities, top PageRank scores, per-candidate Dense/graph/final scores, convergence, and exact-Dense fallback state.

## Boundary from the official product

Microsoft FastGraphRAG uses NLP noun-phrase extraction and text-unit co-occurrence to cheaply construct the entity graph. The full GraphRAG product then adds community detection, LLM-generated community reports, local/global context builders, and answer generation. Those generation components are intentionally excluded here because this benchmark evaluates evidence ranking rather than generated answers. The fidelity reference is the official [indexing-method description](https://microsoft.github.io/graphrag/index/methods/) and [local-search dataflow](https://microsoft.github.io/graphrag/query/local_search/).

GraphRAG never receives `EvidenceGraph`, `ProvenanceGraph`, query labels, or gold spans. ISETrace provenance retrieval remains the separate `provenance_path` method. See [`isetrace-nontrain-retrieval.md`](isetrace-nontrain-retrieval.md).
