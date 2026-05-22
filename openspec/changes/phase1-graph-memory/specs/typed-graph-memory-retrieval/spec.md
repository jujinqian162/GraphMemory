## ADDED Requirements

### Requirement: Build deterministic typed memory graphs
The system SHALL build one graph per task input using only input-visible fields. Each graph MUST contain exactly one question node `q`, all memory sentence nodes, and Phase 1 edge types `sequential`, `query_overlap`, `entity_overlap`, and `bridge`.

#### Scenario: Graph contains required nodes
- **WHEN** graph construction receives a valid task input
- **THEN** the graph contains the `q` question node and every memory node from the task input exactly once

#### Scenario: Sequential edges connect adjacent sentences in one source
- **WHEN** two memory items share the same source and their sentence IDs differ by one
- **THEN** graph construction creates a non-directed `sequential` edge between those memory nodes

#### Scenario: Query overlap edges originate from question node
- **WHEN** memory items share positive content-token overlap with the query
- **THEN** graph construction creates directed `query_overlap` edges from `q` to at most the configured number of top-scoring memory nodes

#### Scenario: Graph construction contains no labels
- **WHEN** a graph artifact is serialized
- **THEN** the serialized graph contains no gold answer, supporting fact, gold evidence, gold dependency, or `is_gold*` fields

### Requirement: Provide stopword-safe lexical and entity utilities
The system SHALL provide deterministic text and entity utilities for graph construction and BM25 tokenization. Content-token processing MUST remove common stopwords and punctuation, drop short non-title tokens, and preserve meaningful title/entity terms.

#### Scenario: Content tokens drop stopwords
- **WHEN** text contains stopwords and meaningful entity words
- **THEN** `content_tokens` excludes stopwords such as `the`, `and`, and `of` while preserving meaningful tokens such as `eiffel`, `tower`, and `river`

#### Scenario: Lexical score rewards content overlap
- **WHEN** a query and passage share content tokens with positive IDF
- **THEN** `lexical_score` produces a higher score than a passage sharing only stopwords

#### Scenario: Entity extraction is optional and deterministic without spaCy
- **WHEN** spaCy is not requested or no model object is provided
- **THEN** entity extraction uses deterministic heuristics and does not download a model

### Requirement: Run flat retrievers with complete ranked output
The system SHALL support `bm25` and `dense` retrieval methods. Each method MUST return every memory node exactly once in descending score order and assemble ranked results in the shared ranked-result schema.

#### Scenario: BM25 ranks all task memory nodes
- **WHEN** `bm25` retrieval runs for a valid task input
- **THEN** the result contains a complete ranked list over all task memory node IDs

#### Scenario: Dense ranks all task memory nodes
- **WHEN** `dense` retrieval runs with an available Sentence-Transformers encoder
- **THEN** the result contains a complete ranked list over all task memory node IDs using normalized embedding dot products

#### Scenario: Dense model unavailable during tests skips clearly
- **WHEN** a real dense model test cannot load a local model without network access
- **THEN** the test is skipped with a clear reason rather than downloading a model

### Requirement: Compose graph rerank from initial retrieval scores
The system SHALL support `bm25_graph_rerank` and `dense_graph_rerank` as composed methods that first run the corresponding flat retriever and then apply graph reranking. Graph reranking MUST consume explicit initial scores, a graph, and a graph-rerank config, and MUST return a complete ranking over all original memory nodes.

#### Scenario: Graph rerank promotes connected candidates
- **WHEN** a graph contains a bridge or entity edge from a high-scoring seed to another memory node
- **THEN** graph rerank can add the configured graph bonus and promote the connected node while preserving all original nodes in the final ranking

#### Scenario: Equal initial scores normalize safely
- **WHEN** all initial scores for a task are equal
- **THEN** normalized initial scores are `0.0` and rerank produces finite final scores

#### Scenario: Graph methods require graph input and config
- **WHEN** a graph-rerank method is requested without matching graphs or a graph-rerank config
- **THEN** retrieval fails fast before writing ranked results

### Requirement: Extract retrieved subgraphs for top-k analysis
The system SHALL include a retrieved subgraph in every ranked result. Flat methods MAY include selected top-k nodes with an empty edge list, while graph methods SHOULD include the induced top-k graph edges from the constructed graph.

#### Scenario: Flat retrieved subgraph has no edges
- **WHEN** `bm25` or `dense` retrieval writes ranked results
- **THEN** `retrieved_subgraph.nodes` contains at most `top_k` node IDs and `retrieved_subgraph.edges` is empty

#### Scenario: Graph retrieved subgraph includes induced edges
- **WHEN** a graph-rerank method writes ranked results
- **THEN** `retrieved_subgraph.edges` contains graph edges whose endpoints are both selected top-k memory nodes
