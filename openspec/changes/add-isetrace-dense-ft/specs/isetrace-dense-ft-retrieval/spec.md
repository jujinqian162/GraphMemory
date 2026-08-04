## Purpose

Defines a supervised flat-text Dense-FT baseline for the current ISETrace benchmark that shares flat chunks and exact-span evaluation with non-training methods while remaining graph-free.

## ADDED Requirements

### Requirement: ISETrace Dense-FT uses the shared flat retrieval view
The system SHALL train and retrieve under the existing `dense_ft` method ID using the same flat trajectory chunk candidates consumed by ISETrace BM25 and frozen Dense. It MUST NOT provide the model with `ProvenanceGraph`, EvidenceGraph, motif metadata, dependency edges, graph or node identifiers, query origin, or native retrieval traces.

#### Scenario: Build an ISETrace Dense-FT request
- **WHEN** an ISETrace train, dev, or test record is projected for `dense_ft`
- **THEN** the request contains query text and the record's flat candidates with exact source spans
- **AND** it contains no graph or provenance-only metadata input

#### Scenario: Compare flat retrieval methods
- **WHEN** frozen Dense and Dense-FT run on the same resolved ISETrace test split
- **THEN** both methods rank the same candidate IDs with the same candidate text and exact source spans

### Requirement: Exact spans define flat-chunk positives
The system SHALL map every ISETrace natural or template gold span to every flat candidate whose exact source spans overlap it. Every resulting positive ID MUST belong to that task's flat candidate set, and preparation or pair construction MUST fail when a training or dev task maps to no positive candidate.

#### Scenario: Map a natural span
- **WHEN** a natural query gold span overlaps one or more flat chunks
- **THEN** every overlapping chunk is a positive Dense-FT candidate

#### Scenario: Map template supervision
- **WHEN** a template query's focused output content has been materialized as exact gold spans
- **THEN** those spans are mapped through the same flat-chunk overlap rule as natural spans
- **AND** motif focus, participant IDs, and graph topology are not passed to Dense-FT

#### Scenario: Reject unmappable supervision
- **WHEN** a train or dev query has exact gold spans but none overlaps a flat candidate
- **THEN** Dense-FT pair or dev-label construction fails with the task ID

### Requirement: ISETrace Dense-FT uses text-only negative sampling
The ISETrace Dense-FT pair stage SHALL use an effective configuration with `hard_graph_neighbor_per_positive=0`. Configured easy-random, BM25-hard, and Dense-hard negatives SHALL remain available, while existing evidence-dataset Dense-FT sampling behavior SHALL remain unchanged.

#### Scenario: Build ISETrace Dense-FT pairs
- **WHEN** a run selects `dense_ft` on ISETrace
- **THEN** pair construction receives flat requests and exact-span-derived labels
- **AND** no EvidenceGraph or provenance candidate-neighbor edges are supplied
- **AND** the effective graph-neighbor negative count is zero

#### Scenario: Preserve evidence Dense-FT pairs
- **WHEN** a run selects `dense_ft` on HotpotQA, 2Wiki, or MuSiQue
- **THEN** its configured graph input and graph-neighbor negative behavior are unchanged

#### Scenario: Track one effective policy
- **WHEN** ISETrace Dense-FT pair and train stages are materialized and tracked
- **THEN** their scientific configuration and artifact identity expose the same effective zero graph-neighbor count

### Requirement: Dev selection matches task-local retrieval semantics
The ISETrace Dense-FT trainer SHALL evaluate each dev query only against candidates from that query's trajectory task. It SHALL report natural and template dev metrics separately, select mixed-dev checkpoints by natural Recall@5, and select template-only-dev checkpoints by template Recall@5.

#### Scenario: Evaluate mixed dev
- **WHEN** dev contains natural and template queries
- **THEN** each query is ranked only over its own candidates
- **AND** natural and template Recall@5 are reported separately
- **AND** natural Recall@5 controls best-checkpoint selection

#### Scenario: Evaluate template-only dev
- **WHEN** dev contains no natural query and at least one template query
- **THEN** template Recall@5 controls best-checkpoint selection
- **AND** model metadata records template as the selection origin

#### Scenario: Exclude foreign task candidates
- **WHEN** two dev tasks contain distinct candidate sets or reuse a candidate-local identifier
- **THEN** neither task's ranking or relevant set contains the other task's candidates

### Requirement: Dense-FT batches avoid trajectory-local false negatives
The ISETrace Dense-FT training loader SHALL prevent two examples owned by the same trajectory from entering one in-batch-negative batch. Example ordering and batch composition MUST be deterministic for the configured training seed.

#### Scenario: Batch repeated supervision over one trajectory
- **WHEN** multiple natural or template examples refer to the same trajectory
- **THEN** no generated training batch contains more than one of those examples

#### Scenario: Repeat a seeded run
- **WHEN** the same examples, batch size, and training seed are used twice
- **THEN** the generated batch sequence is identical

### Requirement: Natural-only test uses existing exact-span evaluation
The ISETrace Dense-FT workflow SHALL retrieve only the resolved natural test split, save the existing SentenceTransformer model-directory artifact and metadata, return a complete flat ranking, and run the existing exact-span evaluation suite without path or edge claims.

#### Scenario: Complete a Dense-FT test run
- **WHEN** an ISETrace Dense-FT model is trained and evaluated
- **THEN** the workflow produces pair, model-directory, training-metric, prediction, exact-span metric, and aggregate artifacts
- **AND** no EvidenceGraph or frozen R-GCN embedding stage is scheduled

#### Scenario: Evaluate formal test
- **WHEN** Dense-FT predictions are evaluated on ISETrace test
- **THEN** every evaluated query has natural origin
- **AND** Recall, MRR, token-budget Coverage, Full Support, span F1, and evidence density are produced
- **AND** path and edge metrics remain unavailable without independent labels
