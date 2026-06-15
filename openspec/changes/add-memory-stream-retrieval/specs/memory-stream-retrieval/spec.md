## ADDED Requirements

### Requirement: Memory Stream combines relevance, pseudo-recency, and importance
The system SHALL rank every memory item with a weighted sum of normalized dense relevance, normalized position-derived pseudo-recency, and normalized offline importance.

#### Scenario: Raw signals are computed deterministically
- **WHEN** a task is ranked
- **THEN** dense cosine score supplies relevance, `recency_decay ** (max_position - position)` supplies pseudo-recency, and the validated sidecar integer supplies importance

#### Scenario: Conservative default weights protect dense relevance
- **WHEN** the default Memory Stream settings are used
- **THEN** relevance has weight 1.0, pseudo-recency has weight 0.0, and importance has weight 0.01
- **AND** pseudo-recency is inactive unless explicitly assigned a positive weight

#### Scenario: Constant signal has no ranking effect
- **WHEN** all memory items have the same raw value for one signal
- **THEN** that signal normalizes to zero for every item

### Requirement: Memory Stream uses the existing dense relevance contract
The system SHALL construct the relevance component through the existing dense encoder settings, passage formatting, normalized embeddings, batching, and cosine scoring path.

#### Scenario: Relevance text matches frozen dense retrieval
- **WHEN** Memory Stream and frozen dense retrieval use the same encoder settings for the same task
- **THEN** they encode identical query and passage text and produce identical raw relevance scores

#### Scenario: Tests may inject a fake encoder
- **WHEN** a sentence encoder is supplied through the retrieval build payload
- **THEN** Memory Stream uses it without loading a real SentenceTransformer model

### Requirement: Artifact loading and validation precede method execution
The retrieval stage SHALL read the cleaned importance artifact once before
constructing Memory Stream, and the builder SHALL select and validate all
records required by the current retrieval task list before ranking begins.

#### Scenario: Compact schema is required
- **WHEN** the loaded artifact does not have `schema_version=1`,
  `method=memory_stream`, and only the compact artifact fields
- **THEN** retrieval fails before method construction

#### Scenario: Builder accepts a canonical superset
- **WHEN** the loaded artifact contains all requested tasks plus extra tasks
- **THEN** the builder injects only the requested validated records into the method

#### Scenario: Invalid subset fails before timed ranking
- **WHEN** a requested task is missing, duplicated, stale, or has mismatched nodes
- **THEN** method construction fails before any task ranking latency is measured

#### Scenario: Method performs no file IO
- **WHEN** `MemoryStreamMethod.rank_task()` is called
- **THEN** it uses injected dense and importance dependencies without reading a path or JSON artifact

### Requirement: Global importance artifacts cover every retrieval task
The system SHALL reject Memory Stream retrieval unless the global importance artifact contains a valid record for every retrieval task and every memory node exactly, while permitting additional unselected canonical tasks.

#### Scenario: Missing task fails before ranking
- **WHEN** a retrieval task id is absent from the importance artifact
- **THEN** retrieval fails before producing predictions and names the missing task id

#### Scenario: Node mismatch fails before ranking
- **WHEN** an importance task record has missing or extra node ids relative to the retrieval task
- **THEN** retrieval fails before producing predictions and names the mismatched node ids

#### Scenario: Changed memory content fails digest validation
- **WHEN** node ids match but source, text, position, or item order differs from the annotated content
- **THEN** retrieval fails because the task content digest no longer matches

#### Scenario: Artifact superset is accepted
- **WHEN** the global artifact contains valid records for all retrieval tasks plus additional canonical tasks
- **THEN** retrieval selects requested records by task id and continues

### Requirement: Memory Stream ranking and output are deterministic
The system SHALL sort method-level `RankedNode` values by descending final
score and ascending node id, return `RetrievalMethodResult`, and leave
`RetrievalTrace.retrieved_edges` empty so execution can assemble the shared
`RankedResult` artifact.

#### Scenario: Score ties use node id
- **WHEN** two nodes have the same final score
- **THEN** the lexicographically smaller node id ranks first

#### Scenario: Shared result contract is preserved
- **WHEN** Memory Stream retrieval completes
- **THEN** every task produces a complete ranked node list, top-k retrieved node ids, empty retrieved edges, latency, and input-token estimate in the standard schema

### Requirement: Memory Stream settings are valid
The system SHALL require non-negative signal weights with at least one positive
weight and SHALL require `0 < recency_decay <= 1`.

#### Scenario: Invalid weight fails during configuration
- **WHEN** any weight is negative or all three weights are zero
- **THEN** configuration fails before method construction

#### Scenario: Invalid decay fails during configuration
- **WHEN** recency decay is not greater than zero and at most one
- **THEN** configuration fails before method construction

### Requirement: Timed retrieval consumes cleaned importance
The system SHALL consume a completed compact importance artifact and SHALL NOT
call an LLM or import an annotation runtime during Memory Stream retrieval.

#### Scenario: Retrieval has no annotation runtime dependency
- **WHEN** a valid compact importance artifact exists
- **THEN** Memory Stream retrieval completes using only the artifact and dense encoder

#### Scenario: Retrieval latency covers online ranking only
- **WHEN** Memory Stream predictions are written
- **THEN** each prediction latency includes dense relevance and score combination
- **AND** excludes artifact file reading, hashing, subset validation, and offline cleaning

### Requirement: Runtime provenance records effective Memory Stream inputs
The system SHALL record the effective dense encoder settings and compact
importance artifact metadata used to build Memory Stream retrieval.

#### Scenario: Run summary identifies artifact provenance
- **WHEN** Memory Stream retrieval succeeds
- **THEN** its run summary identifies the importance artifact path and hash,
  schema version, score weights, recency decay, and effective dense encoder settings
