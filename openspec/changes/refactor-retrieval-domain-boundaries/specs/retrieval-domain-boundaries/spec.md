## ADDED Requirements

### Requirement: Retrieval construction uses method-family requests
The system SHALL construct retrieval methods through precise method-family build requests instead of a universal `RetrievalBuildContext`.

#### Scenario: Universal context is absent
- **WHEN** repository imports and source text are scanned after Change B
- **THEN** no production, script, or test code defines or imports `RetrievalBuildContext`

#### Scenario: Method family inputs are explicit
- **WHEN** flat, graph-rerank, and trainable graph methods are built
- **THEN** each method receives only the request fields required by its method family

### Requirement: Dense runtime details stay inside dense-owned objects
The system SHALL keep dense encoder model name, query prefix, passage prefix, injected encoder, and batch-size behavior inside dense config/runtime objects after CLI-shaped input has been resolved.

#### Scenario: Dense prefixes do not leak through high-level factory calls
- **WHEN** retrieval resolver/factory tests inspect method-family requests
- **THEN** `query_prefix` and `passage_prefix` appear only in dense-owned config/runtime objects, not as loose fields on a universal context

#### Scenario: Dense ranking behavior remains stable
- **WHEN** fake dense retrieval runs with the same task input, prefixes, and injected encoder
- **THEN** encoder call text, ranking order, scores, and returned artifact schema remain equivalent to the pre-refactor behavior

### Requirement: Retrieval execution preserves ranked result artifacts
The system SHALL move ranked-result assembly and token approximation into the retrieval execution boundary without changing output fields, ordering, latency accounting semantics, or validation.

#### Scenario: BM25 result artifacts remain stable
- **WHEN** BM25 retrieval runs on the frozen tiny fixture
- **THEN** ranked nodes, retrieved subgraph shape, method name, and token approximation remain equivalent to the pre-refactor behavior

### Requirement: Public retrieval surface remains compatible
The system SHALL preserve public retrieval method names, script parser contracts, workflow command compatibility, and the public `run_retrieval()` behavior used by scripts and tests.

#### Scenario: Script contracts remain stable
- **WHEN** parser contract tests inspect `scripts/run_retrieval.py` and related workflow-generated commands
- **THEN** argument names, defaults, choices, required flags, and compatibility aliases remain unchanged

#### Scenario: Retrieval registry remains the workflow integration port
- **WHEN** workflow code queries public method metadata
- **THEN** it continues to use `graph_memory/retrieval_registry.py` without depending on retrieval implementation modules

### Requirement: Trainable graph retrieval remains behavior-compatible
The system SHALL keep `dense_rgcn_graph_retriever` available through the new retrieval factory while delegating model internals to the existing learned implementation until Change C.

#### Scenario: Trainable method still accepts checkpoint inputs
- **WHEN** trainable retrieval tests call `run_retrieval()` with graphs and a checkpoint path
- **THEN** behavior and validation remain equivalent to the pre-refactor implementation
