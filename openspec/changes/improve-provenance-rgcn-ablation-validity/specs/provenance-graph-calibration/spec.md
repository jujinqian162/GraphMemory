## ADDED Requirements

### Requirement: Schema-v3 graphs use a hidden gold spine with matched branches
The converter SHALL materialize the ordered gold output dependency as an ordinary `feeds` plus `returns` path before adding distractors. Every eligible source output SHALL expose the same configured feed out-degree and the same semantic-head plus rank-banded-branch selection policy, and the visible gold path MUST NOT use a gold-only node type, edge type, binding schema, metadata shape, degree, order, rank bucket, or weight bucket.

#### Scenario: Gold target misses semantic head
- **WHEN** the gold target is not the source's rank-1 successor
- **THEN** the converter places it in the source's branch slot and also emits at least one non-gold branch from the same rank bucket when the candidate pool permits

#### Scenario: Gold target is semantic head
- **WHEN** the gold target ranks first for the gold source
- **THEN** it occupies the ordinary head slot and the source still receives an ordinary non-gold branch slot

#### Scenario: Matching branch cannot be formed
- **WHEN** fixed out-degree and a non-gold rank-bucket match cannot both be satisfied
- **THEN** the converter rejects the record with a typed construction reason instead of emitting a gold-unique topology

### Requirement: Branch proposals come from a complete source-aware semantic ranking
For every source output, the converter SHALL rank every non-self candidate using a versioned source-aware scorer and SHALL select one semantic head plus one deterministic proposal from the configured near, mid, or tail rank bucket. The default construction scorer SHALL be the pinned hybrid BM25/frozen-dense scorer; BM25-only and dense-only SHALL remain explicit construction interventions.

#### Scenario: Build a source successor set
- **WHEN** a source has more eligible targets than the configured successor count
- **THEN** its head is complete-ranking position 1 and its branch is selected deterministically from the source's assigned rank bucket with stable tie-breaking

#### Scenario: Repeat conversion
- **WHEN** sources, model digest, prefixes, scorer settings, branch buckets, seed, and converter schema are identical
- **THEN** candidate selection, branch endpoints, record order, serialized graph, manifest, and statistics are byte-stable

#### Scenario: Construction intervention changes
- **WHEN** BM25, dense, hybrid weight, semantic temperature, query template, or branch policy changes
- **THEN** the generated dataset identity changes and the manifest records the effective intervention

### Requirement: Feed weights are bounded and source-mass preserving
The converter SHALL normalize selected successor confidence within each source and SHALL compute each feed weight as `floor + (1 - floor) * probability`, with default floor `0.5`. Non-feed structural edges SHALL have weight `1.0`; every source with the same out-degree SHALL have the same total feed-message mass.

#### Scenario: Two successors are emitted
- **WHEN** a source emits two feed successors and the configured floor is `0.5`
- **THEN** both weights lie in `[0.5, 1.0]` and their sum is `1.5`

#### Scenario: Low-ranked gold branch is retained
- **WHEN** the gold edge has low source-local semantic confidence
- **THEN** its artifact weight remains at or above the configured floor and no `1 / rank` attenuation is applied

#### Scenario: Structural shell edge is emitted
- **WHEN** the converter emits `contains`, `invokes`, or `returns`
- **THEN** the edge weight is `1.0` and is excluded from feed-confidence normalization

### Requirement: Every feed edge has the same auditable confidence schema
Every visible feed edge SHALL record scorer identity, complete semantic rank, raw normalized score, source-local probability, calibrated weight, branch role, and rank bucket under one common metadata schema. Gold-only construction facts SHALL remain label-side.

#### Scenario: Ranking record is validated
- **WHEN** validation inspects gold and non-gold feed edges
- **THEN** both expose identical metadata keys and value types and neither exposes `gold`, `fallback`, support annotations, answer text, or another correctness field

#### Scenario: Label record is inspected
- **WHEN** construction diagnostics require the gold semantic rank or whether it occupied the head slot
- **THEN** those facts are available only in label metadata and aggregate statistics, not ranking input

### Requirement: Generated statistics detect structural and weight signatures
Conversion SHALL emit per-split statistics for candidate/node/edge counts, feed out-degree, head/branch role counts, rank-bucket counts by gold/non-gold status, semantic-rank distributions, weight distributions, source-mass checks, gold-head rate, and construction rejection reasons.

#### Scenario: Gold branch occupies a tail bucket
- **WHEN** a generated split contains a gold edge in the tail bucket
- **THEN** statistics show at least one non-gold tail-bucket edge when an accepted record was required to provide a match

#### Scenario: Source mass is audited
- **WHEN** statistics scan an accepted graph
- **THEN** every fixed-degree source passes the configured feed-weight sum invariant and any violation fails conversion validation

### Requirement: Schema-v3 artifacts are isolated from earlier benchmark data
The converter SHALL bump the raw record schema and construction implementation identity. Prepare, pair, model, prediction, and evaluation artifacts derived from pre-v3 records MUST NOT satisfy v3 cache or provenance checks.

#### Scenario: Existing v2 raw data is selected
- **WHEN** the v3 adapter or validator receives a v2 generated record
- **THEN** it fails with an explicit schema mismatch before training or ranking

#### Scenario: V3 data is regenerated
- **WHEN** accepted v3 raw files are prepared
- **THEN** every downstream artifact origin records the v3 prepared digest and cannot alias a v2-derived artifact
