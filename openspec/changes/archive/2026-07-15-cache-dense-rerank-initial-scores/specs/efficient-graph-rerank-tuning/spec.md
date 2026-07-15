## ADDED Requirements

### Requirement: Graph rerank tuning reuses seed retrieval scores
The system SHALL compute initial seed-retriever scores at most once per task for a single graph-rerank tuning invocation and reuse those scores across all graph-rerank candidate configurations.

#### Scenario: Dense tuning with multiple candidates
- **WHEN** dense-seeded graph rerank tuning evaluates multiple graph-rerank configurations over the same tasks
- **THEN** the dense seed retriever is invoked once per task rather than once per task per candidate

#### Scenario: Candidate metrics remain equivalent
- **WHEN** cached initial scores are used to evaluate a graph-rerank candidate
- **THEN** the candidate metrics match the metrics produced by evaluating the same candidate through the normal retrieval path

### Requirement: Existing tuning interfaces remain compatible
The system MUST keep the existing `tune_graph_rerank(...)` callable contract and `scripts/tune_graph_rerank.py` command-line arguments compatible with current usage.

#### Scenario: Existing CLI command
- **WHEN** a user runs the existing `scripts/tune_graph_rerank.py` command with dense or BM25 graph-rerank arguments
- **THEN** the command accepts the same arguments and writes the same selected-config and candidate-row artifact shapes
