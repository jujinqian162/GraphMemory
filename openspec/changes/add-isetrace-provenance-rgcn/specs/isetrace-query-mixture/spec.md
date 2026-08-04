## MODIFIED Requirements

### Requirement: Explicit ISETrace trajectory counts
The system SHALL configure train/dev/test selection with exact natural/template trajectory counts under `trajectories.splits`.

#### Scenario: Parse the supported configuration
- **WHEN** an ISETrace dataset config provides `trajectory_source`, `natural_query_source`, and `trajectories.splits.<split>.natural/template`
- **THEN** strict config parsing accepts nonnegative trajectory counts
- **AND** test requires natural greater than zero and template equal to zero

### Requirement: Deterministic trajectory split
The system SHALL deterministically select trajectories using `split_seed`. Natural and template sets MAY overlap within one split, while the union of selected trajectories MUST be disjoint across train, dev, and test.

#### Scenario: Materialize natural supervision
- **WHEN** a split selects a natural trajectory
- **THEN** preparation includes every valid authored natural query resolved to that trajectory
- **AND** it does not impose one-query-per-trajectory or an additional query-count cap

#### Scenario: Materialize template supervision
- **WHEN** a split selects a template trajectory
- **THEN** preparation emits exactly one deterministic template query for that trajectory

#### Scenario: Reject an insufficient trajectory pool
- **WHEN** the configured split counts require more distinct trajectories than the resolved natural-query pool contains
- **THEN** preparation fails with requested and available trajectory counts

#### Scenario: Prevent cross-split leakage
- **WHEN** train/dev/test trajectories are selected
- **THEN** no trajectory occurs in more than one split

### Requirement: Preserve query origin outside model inputs
The system SHALL retain natural/template origin for preparation, selection, and reporting while excluding it from retrieval requests and model features.

#### Scenario: Select a mixed-dev checkpoint
- **WHEN** mixed dev data includes natural records
- **THEN** natural-query Recall@5 is the checkpoint-selection input
- **AND** template-query metrics are reported separately

#### Scenario: Select a template-only-dev checkpoint
- **WHEN** dev contains no natural records and at least one template record
- **THEN** template-query Recall@5 is the checkpoint-selection input
- **AND** model metadata records template as the selection origin
