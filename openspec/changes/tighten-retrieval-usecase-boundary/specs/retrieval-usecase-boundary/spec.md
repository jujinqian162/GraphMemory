## ADDED Requirements

### Requirement: Retrieval run orchestration belongs to application
The system SHALL expose complete retrieval run orchestration through an application use-case service instead of the retrieval execution service.

#### Scenario: Scripts call the application use case
- **WHEN** retrieval scripts run public methods through loaded task, graph, config, checkpoint, and runtime inputs
- **THEN** they call `graph_memory.application.run_retrieval.run_retrieval` with a single `RunRetrievalRequest`

#### Scenario: Public CLI contracts remain stable
- **WHEN** parser contract tests inspect retrieval, trainable retrieval, and graph-rerank tuning scripts
- **THEN** CLI argument names, defaults, required flags, choices, and artifact outputs remain unchanged

### Requirement: Retrieval execution runs an already-built method
The system SHALL keep retrieval execution at one abstraction level by executing an already-built `RetrievalMethod`.

#### Scenario: Execution service has no loose runtime parameters
- **WHEN** `graph_memory/retrieval/execution/service.py` is inspected
- **THEN** its public execution function does not accept `encoder_model`, `query_prefix`, `passage_prefix`, `graph_config`, `checkpoint_path`, `text_embedding_provider`, `seed_signal_provider`, or `device`

#### Scenario: Method resolution stays before execution
- **WHEN** a retrieval run is prepared
- **THEN** method-family request resolution and factory construction happen before `retrieval.execution.service.run_retrieval` is called

### Requirement: Tuning uses typed dense runtime state internally
The system SHALL keep dense model name and prefixes inside `DenseRuntime` once graph-rerank tuning leaves the CLI/parser boundary.

#### Scenario: Tuning service has no loose dense prefix fields
- **WHEN** `graph_memory/retrieval/tuning/service.py` is inspected
- **THEN** graph-rerank tuning internals accept typed dense runtime state rather than loose `query_prefix` or `passage_prefix` fields

#### Scenario: Initial-score precomputation is tuning-owned
- **WHEN** graph-rerank tuning precomputes seed retrieval scores
- **THEN** the cache helper lives under `graph_memory.retrieval.tuning` and not in the retrieval execution service

### Requirement: Retrieval artifacts remain behavior-compatible
The system SHALL preserve ranked result artifacts and validation semantics while moving orchestration.

#### Scenario: Frozen retrieval fixtures remain equivalent
- **WHEN** BM25, fake dense, graph-rerank, and trainable retrieval fixture tests run
- **THEN** ranking order, scores, retrieved subgraph shape, method names, and result validation remain compatible with the previous behavior
