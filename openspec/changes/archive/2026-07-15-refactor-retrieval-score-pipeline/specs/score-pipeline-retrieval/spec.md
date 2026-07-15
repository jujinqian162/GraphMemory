## ADDED Requirements

### Requirement: Run retrieval methods through a stable method boundary
The system SHALL execute public retrieval method names through a retrieval-method boundary that produces ranked results in the existing shared schema.

#### Scenario: Public method names remain compatible
- **WHEN** retrieval is run with `bm25`, `dense`, `bm25_graph_rerank`, or `dense_graph_rerank`
- **THEN** the output method name, ranked node schema, retrieved subgraph schema, latency field, and input-token field remain compatible with existing evaluation scripts

#### Scenario: Unsupported method fails fast
- **WHEN** retrieval is run with an unsupported method name
- **THEN** the system raises a clear unsupported-method error before processing task inputs

### Requirement: Compose score-based baselines from node-score components
The system SHALL provide a score-pipeline retrieval implementation that combines reusable node-score components into a final complete node ranking.

#### Scenario: Flat baseline uses one component
- **WHEN** a flat score-based method such as `bm25` or `dense` is run
- **THEN** the score pipeline combines a single baseline score component and returns every memory node exactly once

#### Scenario: Graph rerank baseline combines baseline and graph components
- **WHEN** `bm25_graph_rerank` or `dense_graph_rerank` is run with valid graph inputs and graph config
- **THEN** the score pipeline combines the baseline score with query-overlap, neighbor-propagation, and bridge graph components using the configured weights

#### Scenario: Component scores are normalized before weighted combination
- **WHEN** score components have different natural score scales
- **THEN** the score pipeline normalizes component scores according to each component's declared normalization before applying weights

### Requirement: Keep graph requirements owned by graph-aware methods
The system SHALL make graph input and graph config requirements explicit in graph-aware retrieval methods instead of scattering graph checks through the task loop.

#### Scenario: Flat methods do not require graph inputs
- **WHEN** `bm25` or `dense` is run with no graphs
- **THEN** retrieval succeeds and emits an empty edge list in the retrieved subgraph

#### Scenario: Graph methods require graph inputs and config
- **WHEN** `bm25_graph_rerank` or `dense_graph_rerank` is run without graph inputs or graph-rerank config
- **THEN** retrieval fails fast with a graph-method requirement error

### Requirement: Preserve graph-rerank scoring semantics
The system SHALL preserve existing graph-rerank ranking behavior while moving graph-rerank execution into the score-pipeline method.

#### Scenario: Pipeline matches graph rerank helper
- **WHEN** a graph-rerank method receives initial scores, a graph, and a graph-rerank config
- **THEN** the score-pipeline ranking matches the ranking produced by the graph-rerank helper for the same inputs

#### Scenario: Graph candidates gate graph components
- **WHEN** a memory node is outside the configured graph candidate expansion
- **THEN** query-overlap, neighbor-propagation, and bridge components do not contribute to that node's final score

#### Scenario: Top-k graph methods return induced edges
- **WHEN** a graph-rerank method emits a retrieved subgraph
- **THEN** the retrieved subgraph includes only graph edges whose endpoints are both selected top-k memory nodes
