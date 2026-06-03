## ADDED Requirements

### Requirement: Foundational contracts are domain-owned
The system SHALL move artifact-shaped foundational contracts into explicit `graph_memory/contracts/` modules without changing field names, required fields, optional fields, dataclass defaults, or serialized artifact schemas.

#### Scenario: Contract fields are preserved
- **WHEN** migrated contract tests inspect task, graph, ranking, training-pair, metric, and observability records
- **THEN** the public artifact shapes remain equivalent to the current root `types.py` definitions

### Requirement: Validation behavior is preserved
The system SHALL split validators into `graph_memory/validation/` modules while preserving valid-input success, invalid-input fail-fast behavior, exception types, and error message semantics.

#### Scenario: Validation tests remain behavior-equivalent
- **WHEN** existing and new validation tests run against migrated validators
- **THEN** legal artifacts continue to pass and illegal artifacts fail for the same reasons as before the migration

### Requirement: Infrastructure helpers have narrow workflow ports
The system SHALL move IO and run-summary implementation into `graph_memory/infrastructure/` while keeping root `graph_memory/io.py` and `graph_memory/observability.py` as narrow workflow integration ports.

#### Scenario: Workflow imports keep working through narrow ports
- **WHEN** workflow code imports approved IO and observability names from the root integration ports
- **THEN** those imports continue to work without exposing unrelated domain logic or expanding the compatibility surface

### Requirement: Dataset and text helpers are split without semantic changes
The system SHALL move HotpotQA parsing, compatibility conversion, split helpers, tokenization, lexical scoring, and entity helpers into `graph_memory/datasets/` and `graph_memory/text/` without changing parsing errors, conversion order, tokenization, IDF, lexical score, or entity extraction semantics.

#### Scenario: Dataset and text fixtures remain stable
- **WHEN** focused dataset and text tests run before and after the migration
- **THEN** converted artifacts, token sequences, IDF values, lexical scores, and entity outputs remain equivalent

### Requirement: Graph construction domain is explicit
The system SHALL move graph config, construction, index/statistics, and graph-view logic into `graph_memory/graphs/` while preserving node creation, edge rule order, edge weights, deduplication, bridge behavior, and graph statistics.

#### Scenario: Graph artifacts remain stable
- **WHEN** graph fixture tests build a graph from the same tiny task input
- **THEN** nodes, edges, ordering, weights, statistics, and validation outcomes remain equivalent to the current implementation

### Requirement: Evaluation domain is explicit
The system SHALL move metric primitives, connectivity derivation, evaluation service, table rows, and failure-case generation into `graph_memory/evaluation/` without changing metric definitions, joins, fail-fast behavior, CSV columns, or failure-case output.

#### Scenario: Evaluation outputs remain stable
- **WHEN** evaluation fixture tests run against the same ranked results and labels
- **THEN** metric rows, aggregate rows, connectivity values, and failure-case records remain equivalent to the current implementation
