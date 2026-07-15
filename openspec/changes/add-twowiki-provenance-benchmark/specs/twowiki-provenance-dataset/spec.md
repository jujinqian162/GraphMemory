## ADDED Requirements

### Requirement: Conversion is deterministic and source-auditable
The system SHALL convert labeled 2Wiki source data into separately named `twowiki_provenance` raw train/dev/test artifacts and SHALL emit a manifest containing source hashes, schema version, seed, split policy, conversion parameters, and accepted/rejected counts.

#### Scenario: Repeat conversion
- **WHEN** the converter is run twice with identical source files and options
- **THEN** every split artifact, manifest field, record order, node ID, edge ID, and statistic is identical

#### Scenario: Unlabeled official test source
- **WHEN** labeled source train and dev files are provided but the official test file lacks gold support labels
- **THEN** source train becomes target train and source dev is deterministically partitioned into disjoint target dev and test splits

### Requirement: Only recoverable ordered gold chains are emitted
The converter SHALL emit only records whose ordered gold support chain resolves to distinct candidate sentences and a connected dependency path, and MUST NOT infer or repair the path from the final answer.

#### Scenario: Ambiguous support chain
- **WHEN** a source example has missing, duplicated, self-looping, unordered, or unresolved gold supports
- **THEN** the converter rejects it with a typed reason before target split sampling

#### Scenario: Ambiguous evidence-to-support mapping
- **WHEN** two support sentences tie for the best evidence-triple mapping score
- **THEN** the converter rejects the example as `ambiguous_gold_chain` instead of selecting the first support by order

### Requirement: Native graph schema represents calls, outputs, and data flow
Each emitted task SHALL contain Task and Agent context nodes, one ToolCall/ToolOutput pair per candidate, a `returns` edge per pair, and directed `feeds` edges with FieldBinding metadata for dependency branches; only ToolOutput IDs SHALL be ranking candidates.

#### Scenario: Recover gold dependency
- **WHEN** a gold first-hop output supports a gold second-hop output
- **THEN** the graph contains `gold_output_1 -> feeds -> gold_call_2 -> returns -> gold_output_2` and the label contains the contracted ordered output dependency edge

#### Scenario: Candidate identity
- **WHEN** a ranking request is projected
- **THEN** its candidate IDs equal the graph's retrievable ToolOutput IDs and exclude Task, Agent, and ToolCall connector nodes

### Requirement: Negative branches are structurally comparable and connected
The converter SHALL add query-relevant non-gold branches by ranking successor candidates with a versioned BM25, dense, or hybrid scorer over `question + source evidence`, using the same public node types, edge types, directions, weights, and binding schema as the gold branch, and SHALL NOT emit isolated candidate outputs or randomly shuffled dependency edges.

#### Scenario: Hard-negative successor
- **WHEN** a gold first-hop output is included
- **THEN** at least one eligible non-gold successor branch is added when the candidate pool permits it, without a gold-only relation or field distinguishing the correct successor

#### Scenario: Non-gold first hop
- **WHEN** sufficient hard negatives exist
- **THEN** at least one non-gold first-hop branch has a complete call/output continuation with a path length comparable to the gold chain

#### Scenario: Semantic successor selection
- **WHEN** a source output has more eligible successors than the configured successor count
- **THEN** emitted `feeds` edges target the highest-scoring BM25/dense/hybrid successors with deterministic tie-breaking, while a missed gold successor is added only through the recorded same-schema fallback

### Requirement: Field bindings are endpoint-verifiable
Every generated `feeds` edge SHALL bind a declared source output field to a declared target input parameter and SHALL reuse the deterministic source field-value hash; edge-specific random binding hashes are forbidden.

#### Scenario: Validate evidence binding
- **WHEN** an output feeds a downstream call through `evidence -> context`
- **THEN** the binding hash equals the source output metadata hash for `evidence` and the target call declares `context` as an accepted input parameter

### Requirement: Ranking input is leakage-audited
Ranking artifacts MUST NOT contain final answers, gold flags, gold-only metadata, source support annotations, or topology conventions that uniquely identify the gold path; labels SHALL be stored separately.

#### Scenario: Validate generated ranking record
- **WHEN** the dataset validator inspects a generated task
- **THEN** it verifies forbidden-field absence, legal graph transitions, complete gold-node inclusion, gold-path recoverability, non-isolation, unique identities, and graph/candidate consistency

### Requirement: Dataset projects through repository-owned requests
The `twowiki_provenance` adapter SHALL project one parsed record to TextRankingRequest, GraphRAGRequest, ExecutionProvenanceRankingRequest, and EvidenceEvaluationRequest without retriever-specific raw-data branches.

#### Scenario: Flat baseline
- **WHEN** BM25 or Dense consumes a generated task
- **THEN** it receives the ToolOutput candidate texts and no graph or label-only fields

#### Scenario: Provenance method
- **WHEN** either provenance retriever consumes a generated task
- **THEN** it receives the full typed ExecutionProvenanceGraph and candidate mapping but no label object
