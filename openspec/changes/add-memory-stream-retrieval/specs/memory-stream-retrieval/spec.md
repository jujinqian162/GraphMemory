## ADDED Requirements

### Requirement: Memory Stream combines relevance, pseudo-recency, and importance
The system SHALL rank every memory item with a weighted sum of normalized dense relevance, normalized position-derived pseudo-recency, and normalized offline importance.

#### Scenario: Raw signals are computed deterministically
- **WHEN** a task is ranked
- **THEN** dense cosine score supplies relevance, `recency_decay ** (max_position - position)` supplies pseudo-recency, and the validated sidecar integer supplies importance

#### Scenario: Equal default weights implement the simplified baseline
- **WHEN** the default Memory Stream settings are used
- **THEN** relevance, pseudo-recency, and importance each have weight 1.0

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
The system SHALL sort Memory Stream results by descending final score and ascending node id, emit the shared `RankedResult` shape, and produce no retrieved graph edges.

#### Scenario: Score ties use node id
- **WHEN** two nodes have the same final score
- **THEN** the lexicographically smaller node id ranks first

#### Scenario: Shared result contract is preserved
- **WHEN** Memory Stream retrieval completes
- **THEN** every task produces a complete ranked node list, top-k retrieved node ids, empty retrieved edges, latency, and input-token estimate in the standard schema

### Requirement: Timed retrieval excludes importance generation
The system SHALL consume a completed importance artifact and SHALL NOT call an LLM during Memory Stream retrieval.

#### Scenario: Retrieval runs without the local LLM runtime
- **WHEN** a valid importance artifact exists and the Transformers model environment or model files are unavailable
- **THEN** Memory Stream retrieval still completes using only the artifact and dense encoder

#### Scenario: Retrieval latency covers online ranking only
- **WHEN** Memory Stream predictions are written
- **THEN** each prediction latency includes dense relevance and score combination but excludes offline annotation time

### Requirement: Runtime provenance records effective Memory Stream inputs
The system SHALL record the effective dense encoder settings and importance artifact metadata used to build Memory Stream retrieval.

#### Scenario: Run summary identifies annotation provenance
- **WHEN** Memory Stream retrieval succeeds
- **THEN** its run summary identifies the importance artifact path, model id, prompt version, content digests, score weights, recency decay, and effective dense encoder settings
