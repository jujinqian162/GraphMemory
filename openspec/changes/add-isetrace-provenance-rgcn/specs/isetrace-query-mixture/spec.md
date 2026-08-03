## ADDED Requirements

### Requirement: Explicit ISETrace query counts
The system SHALL configure the trainable ISETrace query corpus with a trajectory source, a natural-query source, and exact natural/template task counts under each train/dev/test split, without ratio-derived or profile-derived query counts.

#### Scenario: Parse the supported configuration
- **WHEN** an ISETrace dataset config provides `trajectory_source`, `natural_query_source`, and `queries.splits.<split>.natural/template`
- **THEN** strict config parsing accepts nonnegative integer counts and preserves them in the resolved scientific config
- **AND** train and dev each require at least one total query
- **AND** test requires natural greater than zero and template equal to zero

#### Scenario: Configure template-only supervision
- **WHEN** train or dev configures natural equal to zero and template greater than zero
- **THEN** strict config parsing accepts the split
- **AND** preparation emits exactly the requested template count and no natural records

#### Scenario: Reject retired ratio and implementation-policy fields
- **WHEN** an ISETrace config adds `split_ratio`, `mix_ratio`, query `kind`, invalid policy, grouping policy, manifest name, extension policy, schema version, or mix-policy kind
- **THEN** strict config parsing rejects the fields

#### Scenario: Resolve registered data identity
- **WHEN** preparation reads the configured trajectory and natural-query sources
- **THEN** it obtains the pinned ISETrace revision and natural ownership weights from repository data registration
- **AND** validates compatible authoring run metadata when available
- **AND** fails before graph construction when the recorded source identity conflicts

### Requirement: Deterministic trajectory-grouped ownership
The system SHALL derive exact split targets from registered ownership weights and fixed experiment `split_seed`, while keeping all queries over one trajectory in one split. Supervision counts MUST NOT alter this ownership.

#### Scenario: Allocate an exact 6000-query corpus
- **WHEN** 6,000 valid natural queries are evenly groupable and registered ownership weights resolve to 3,200 train, 800 dev, and 2,000 test
- **THEN** preparation assigns exactly those natural-query capacities

#### Scenario: Keep one graph in one split
- **WHEN** multiple natural queries resolve to the same trajectory
- **THEN** they receive the same split assignment
- **AND** no prepared graph ID occurs across train, dev, and test

#### Scenario: Keep model seeds and supervision counts independent
- **WHEN** the same corpus and `split_seed` are prepared with different model seeds or natural/template task counts
- **THEN** every trajectory retains the same split assignment

#### Scenario: Exclude invalid authored records before allocation
- **WHEN** a natural query is malformed, unmatched, uncompilable, or ambiguous against the configured trajectories
- **THEN** it is excluded before registered ownership allocation
- **AND** the prepared summary records the exclusion reason and count
- **AND** the user does not need to create a filtered query file

### Requirement: Exact natural-template composition
The system SHALL deterministically select exactly the configured natural and template query counts from the trajectory partition owned by each split.

#### Scenario: Compose an explicit mixed train pool
- **WHEN** train configures natural 3,200 and template 9,600
- **THEN** the prepared train pool contains exactly 3,200 natural and 9,600 template queries

#### Scenario: Compose a template-only dev pool
- **WHEN** dev configures natural zero and template 800
- **THEN** the prepared dev pool contains exactly 800 template queries and no natural queries

#### Scenario: Keep test natural-only
- **WHEN** the prepared test split is materialized
- **THEN** every test query originates from the configured natural corpus
- **AND** no template query is present

#### Scenario: Prevent template graph leakage
- **WHEN** a trajectory is assigned to natural test
- **THEN** template queries from that trajectory cannot be selected for train or dev

#### Scenario: Report an insufficient origin pool
- **WHEN** a configured count exceeds the available natural or eligible template records in its split
- **THEN** preparation fails with requested and available counts rather than silently truncating

### Requirement: Preserve query origin outside model inputs
The system SHALL retain natural/template origin for preparation, selection, and reporting while excluding it from retrieval requests and model features.

#### Scenario: Prepare mixed train and dev data
- **WHEN** natural and template records are combined
- **THEN** prepared metadata identifies each record's origin
- **AND** model-facing query text, candidate text, graph tensors, and node features contain no origin flag

#### Scenario: Select a mixed-dev checkpoint
- **WHEN** a mixed dev split includes natural records
- **THEN** natural-query Recall@5 is the checkpoint-selection input
- **AND** template-query metrics are reported separately

#### Scenario: Select a template-only-dev checkpoint
- **WHEN** dev contains no natural records and at least one template record
- **THEN** template-query Recall@5 is the checkpoint-selection input
- **AND** model metadata records template as the selection origin
