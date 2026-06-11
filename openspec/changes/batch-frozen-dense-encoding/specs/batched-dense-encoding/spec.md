## ADDED Requirements

### Requirement: Dense encoding batches texts across task boundaries
The system SHALL support encoding an ordered bounded group of task/node requests through one logical sentence-encoder invocation while preserving the configured encoder text mini-batch size and normalized-embedding behavior.

#### Scenario: Variable-length tasks are flattened and restored
- **WHEN** a bulk encoding request contains tasks with different ordered node-ID lists
- **THEN** the encoder receives the fully prefixed texts in deterministic flattened order and each result contains exactly the embedding rows for its original task and node order

#### Scenario: Encoder mini-batch size is preserved
- **WHEN** the shared dense service encodes a bulk task group
- **THEN** it passes the configured encoder text mini-batch size to the sentence encoder rather than using the task-graph training batch size

#### Scenario: Invalid encoder shape is rejected
- **WHEN** the sentence encoder returns a row count or embedding shape inconsistent with the flattened request
- **THEN** the system raises a validation error before constructing task-aligned results

### Requirement: Dense bulk ranking preserves single-task semantics
The system SHALL rank multiple dense retrieval tasks from normalized query and passage embeddings with the same score formula, descending score order, node-ID tie-break, and complete-node coverage as single-task dense ranking.

#### Scenario: Bulk and single ranking are equivalent
- **WHEN** the same tasks are ranked through the bulk and single-task dense paths with a deterministic encoder
- **THEN** every task has identical node IDs, scores, ordering, and tie-break behavior

#### Scenario: Single-task compatibility remains available
- **WHEN** an existing caller invokes `DenseTaskRetriever.rank(task_input)`
- **THEN** it receives the same public result contract while the implementation uses the shared dense encoding and scoring behavior

### Requirement: Bulk capabilities have deterministic compatibility fallbacks
The system SHALL preserve existing single-task provider and ranker contracts and SHALL use deterministic input-order fallback loops when an injected implementation does not provide the optional bulk capability.

#### Scenario: Custom text provider lacks bulk support
- **WHEN** graph batching receives a text embedding provider that implements only `encode_task_nodes`
- **THEN** the system invokes that method once per task in input order and produces the same graph batch contract

#### Scenario: Custom seed ranker lacks bulk support
- **WHEN** initial-score or hard-negative generation receives a seed ranker that implements only `rank`
- **THEN** the system invokes `rank` once per task in input order and preserves existing results

### Requirement: Collection-oriented dense consumers use bounded bulk ranking
The system SHALL use bounded bulk dense ranking for collection-oriented initial-score precomputation and hard-dense negative preparation when the supplied ranker supports it.

#### Scenario: Initial scores use bulk ranker
- **WHEN** dense initial scores are precomputed for multiple tasks with a bulk-capable ranker
- **THEN** tasks are processed in bounded groups without invoking single-task dense encoding for every task

#### Scenario: Hard-dense negatives preserve ordering
- **WHEN** hard-dense negative candidates are prepared through bulk dense ranking
- **THEN** hard-pool truncation, positive exclusion, de-duplication, selected node order, and pair artifact order remain equivalent to the single-task path

### Requirement: Normal retrieval latency semantics remain unchanged
The system MUST keep normal retrieval execution task-oriented until a separate contract defines latency attribution for cross-task ranking.

#### Scenario: Existing retrieval execution
- **WHEN** `run_retrieval()` produces ranked result artifacts
- **THEN** it continues to measure latency around each task's `rank_task()` call and does not silently replace that measurement with amortized batch latency
