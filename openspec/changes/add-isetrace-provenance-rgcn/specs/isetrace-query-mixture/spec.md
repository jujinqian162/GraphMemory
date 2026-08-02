## ADDED Requirements

### Requirement: Concise ISETrace query configuration
The system SHALL configure the trainable ISETrace query corpus with a trajectory source, a natural-query source, natural split ratios, and train/dev natural-template mix ratios, without exposing fixed implementation policies.

#### Scenario: Parse the supported configuration
- **WHEN** an ISETrace dataset config provides `trajectory_source`, `natural_query_source`, `queries.split_ratio`, `queries.mix_ratio.train`, and `queries.mix_ratio.dev`
- **THEN** strict config parsing accepts the values and preserves them in the resolved scientific config
- **AND** test mixture configuration is absent because test is fixed natural-only

#### Scenario: Reject implementation-policy fields
- **WHEN** an ISETrace config adds fields such as query `kind`, invalid policy, grouping policy, manifest name, extension policy, schema version, or mix-policy kind
- **THEN** strict config parsing rejects the extra fields

#### Scenario: Resolve the source revision
- **WHEN** preparation reads the configured trajectory and natural-query sources
- **THEN** it obtains the pinned ISETrace revision from repository data registration
- **AND** validates compatible authoring run metadata when available
- **AND** fails before graph construction when the recorded source identity conflicts

### Requirement: Deterministic trajectory-grouped natural splitting
The system SHALL split valid natural queries using normalized configured weights and the fixed experiment `split_seed`, while keeping all queries over one trajectory in one split.

#### Scenario: Allocate a 6000-query corpus
- **WHEN** 6,000 valid natural queries are evenly groupable and split weights are train/dev/test `8:2:5`
- **THEN** the target allocation is 3,200 train, 800 dev, and 2,000 test queries

#### Scenario: Keep one graph in one split
- **WHEN** multiple natural queries resolve to the same trajectory
- **THEN** they receive the same split assignment
- **AND** no prepared graph ID occurs across train, dev, and test

#### Scenario: Keep model seeds independent
- **WHEN** the same corpus and `split_seed` are prepared for model training seeds 13, 17, and 29
- **THEN** every natural query receives the same split assignment

#### Scenario: Exclude invalid authored records before splitting
- **WHEN** a natural query is malformed, unmatched, uncompilable, or ambiguous against the configured trajectories
- **THEN** it is excluded before ratio allocation
- **AND** the prepared summary records the exclusion reason and count
- **AND** the user does not need to create a filtered query file

### Requirement: Simple natural-template composition
The system SHALL retain all natural queries assigned to train or dev and deterministically select template queries according to the configured relative weights.

#### Scenario: Compose a three-to-one train pool
- **WHEN** train contains 3,200 natural queries and its mix ratio is natural/template `1:3`
- **THEN** the prepared train pool contains all 3,200 natural queries and 9,600 template queries

#### Scenario: Compose an equal dev pool
- **WHEN** dev contains 800 natural queries and its mix ratio is natural/template `1:1`
- **THEN** the prepared dev pool contains 800 natural and 800 template queries

#### Scenario: Keep test natural-only
- **WHEN** the prepared test split is materialized
- **THEN** every test query originates from the configured natural corpus
- **AND** no template query is present

#### Scenario: Prevent template graph leakage
- **WHEN** a trajectory is assigned to natural test
- **THEN** template queries from that trajectory cannot be selected for train or dev

#### Scenario: Report an insufficient template pool
- **WHEN** the configured mix requires more eligible template queries than the assigned trajectories can provide
- **THEN** preparation fails with the requested and available counts rather than silently changing the ratio

### Requirement: Preserve query origin outside model inputs
The system SHALL retain natural/template origin for preparation, selection, and reporting while excluding it from retrieval requests and model features.

#### Scenario: Prepare mixed train and dev data
- **WHEN** natural and template records are combined
- **THEN** prepared metadata identifies each record's origin
- **AND** model-facing query text, candidate text, graph tensors, and node features contain no origin flag

#### Scenario: Select a checkpoint
- **WHEN** a mixed dev split is evaluated during training
- **THEN** natural-query metrics are available as the primary checkpoint-selection input
- **AND** template-query metrics are reported separately
