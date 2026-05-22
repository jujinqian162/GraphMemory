## ADDED Requirements

### Requirement: Locate graph-rerank scoring in the rerank module
The system SHALL keep graph-rerank scoring abstractions, graph component score calculation, graph candidate expansion, and graph-rerank score combination in `graph_memory/rerank.py`.

#### Scenario: Retrieval orchestration delegates graph rerank scoring
- **WHEN** `bm25_graph_rerank` or `dense_graph_rerank` is executed through public retrieval
- **THEN** `graph_memory/retrieval.py` selects and runs the seed retriever, then delegates graph-rerank score computation to `graph_memory/rerank.py`

#### Scenario: Rerank module owns graph component composition
- **WHEN** graph-rerank final scores are computed from initial, query-overlap, neighbor-propagation, and bridge components
- **THEN** the component normalization and weighted combination logic is owned by `graph_memory/rerank.py`

#### Scenario: Public retrieval schema remains unchanged
- **WHEN** graph-rerank scoring is delegated to `graph_memory/rerank.py`
- **THEN** public retrieval still emits the existing ranked-result schema, method names, latency field, input-token field, and top-k retrieved subgraph shape

### Requirement: Distinguish final component weights from neighbor edge-type weights
The graph-rerank configuration SHALL distinguish final score component weights from graph edge-type calibration weights.

#### Scenario: Lambdas control final component weights
- **WHEN** graph-rerank final scores are combined
- **THEN** `lambda_init`, `lambda_query`, `lambda_neighbor`, and `lambda_bridge` control the contribution of the final score components

#### Scenario: Neighbor type weights calibrate memory-to-memory graph edges
- **WHEN** neighbor-propagation or bridge-neighbor graph scores are computed
- **THEN** `neighbor_type_weights` calibrates the contribution of memory-to-memory graph edge types before component normalization

#### Scenario: Query overlap is not a neighbor type weight
- **WHEN** graph-rerank config validation checks `neighbor_type_weights`
- **THEN** `query_overlap` is not required as a `neighbor_type_weights` entry

### Requirement: Keep query-overlap scoring independent from neighbor type weights
The graph-rerank query-overlap component SHALL be controlled by `lambda_query` and SHALL NOT use `neighbor_type_weights`.

#### Scenario: Query-overlap component ignores neighbor type weights
- **WHEN** a graph contains `q -> memory` query-overlap edges and graph-rerank scoring computes the query-overlap component
- **THEN** the query-overlap component uses the query-overlap edge weights and `lambda_query` without multiplying by `neighbor_type_weights`

#### Scenario: Query-overlap ablation remains explicit
- **WHEN** `lambda_query` is set to `0.0`
- **THEN** query-overlap edges do not contribute to final graph-rerank scores regardless of any neighbor type weight values

### Requirement: Rename graph-rerank type weights with compatibility input
The system SHALL write graph-rerank config artifacts with `neighbor_type_weights` and SHALL accept deprecated `type_weights` as read-only compatibility input.

#### Scenario: New selected configs use neighbor type weights
- **WHEN** graph-rerank tuning writes a selected config or candidate row
- **THEN** the written config uses `neighbor_type_weights` and does not write `type_weights`

#### Scenario: Deprecated type weights remain readable
- **WHEN** graph-rerank config loading receives a record containing `type_weights` and no `neighbor_type_weights`
- **THEN** the loader interprets the memory-to-memory entries as `neighbor_type_weights` and continues execution

#### Scenario: Canonical field wins over deprecated field
- **WHEN** graph-rerank config loading receives both `neighbor_type_weights` and deprecated `type_weights`
- **THEN** `neighbor_type_weights` is used as the canonical value

#### Scenario: Historical query overlap type weight is ignored during compatibility loading
- **WHEN** deprecated `type_weights` contains `query_overlap`
- **THEN** compatibility loading ignores that entry for neighbor type weight construction
