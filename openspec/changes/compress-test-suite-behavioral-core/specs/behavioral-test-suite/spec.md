## ADDED Requirements

### Requirement: Compact collected test budget
The repository test suite SHALL collect no more than 100 pytest cases and SHALL use fewer test files and fewer test-code lines than the pre-change baseline of 69 test files and approximately 15,000 test lines.

#### Scenario: Test inventory is collected
- **WHEN** the repository pytest suite is collected after the change
- **THEN** at most 100 cases are reported and the test file and line inventories are lower than the recorded baseline

### Requirement: Scientific and leakage behavior remains protected
The compact suite SHALL retain behavioral checks for dataset conversion and leakage boundaries, graph construction, retrieval and ranking, evidence metrics, train-pair generation, tensorization, trainable model behavior, beam decoding and training, checkpoint inference, and representative experiment workflows.

#### Scenario: A scientifically relevant regression is introduced
- **WHEN** a protected calculation, leakage boundary, ranking rule, training contract, or workflow connection changes incorrectly
- **THEN** at least one owning-boundary test fails for that behavior

### Requirement: Snapshot and third-party mechanics are excluded
The compact suite SHALL NOT retain tests whose sole purpose is to duplicate exact configuration defaults, preserve deleted compatibility names or modules, snapshot internal exports, inspect source spelling, or exhaustively demonstrate validation behavior owned by Pydantic or Hydra.

#### Scenario: A configuration-only value changes
- **WHEN** a valid configuration value changes without changing custom resolution or runtime behavior
- **THEN** no permanent test requires a duplicated literal value to be updated

### Requirement: Custom business invariants remain covered
The compact suite SHALL cover custom cross-field, scientific, and workflow invariants implemented by repository code even when Pydantic or Hydra performs the underlying parsing.

#### Scenario: A custom cross-field invariant is violated
- **WHEN** repository configuration or artifact data violates a custom relationship such as capacity bounds, propagated seed/device identity, or matched beam widths
- **THEN** an owning-boundary behavioral test demonstrates rejection or safe failure

### Requirement: Future tests require behavioral justification
Repository testing guidance SHALL require every new permanent test to identify a plausible user-visible, scientific, artifact, or workflow regression that it protects. Configuration-only and implementation-detail changes SHALL NOT automatically require new tests.

#### Scenario: A contributor proposes a new test
- **WHEN** the proposed test only repeats a literal, deleted surface, source token, or library validation rule
- **THEN** the guidance directs the contributor to omit it or replace an existing broader test instead
