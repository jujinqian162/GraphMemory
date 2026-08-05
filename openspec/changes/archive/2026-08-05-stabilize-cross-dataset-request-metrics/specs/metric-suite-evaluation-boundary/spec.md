## ADDED Requirements

### Requirement: Evaluation runs through explicit metric suites
The system SHALL evaluate predictions through an explicit metric suite that owns the metric family being computed.

#### Scenario: Evidence suite preserves current metrics
- **WHEN** the evidence metric suite evaluates current HotpotQA evidence retrieval predictions
- **THEN** it returns the existing evidence, support, connectivity, latency, and graph summary metric columns with unchanged formulas

#### Scenario: Alternate suite can use different metric columns
- **WHEN** a future LongMemEval metric suite evaluates ranking or answer predictions
- **THEN** it can return LongMemEval-specific columns such as turn support, session support, and answer quality without requiring evidence-only columns

### Requirement: Metric row validation is suite-owned
The system SHALL validate metric rows according to the selected metric suite rather than one global fixed column set.

#### Scenario: Evidence rows keep strict validation
- **WHEN** evidence metric rows are validated
- **THEN** the validator enforces the evidence suite's required columns and value ranges

#### Scenario: Non-evidence rows are not forced into evidence schema
- **WHEN** non-evidence metric rows are validated by their suite
- **THEN** missing evidence-only columns such as `Evidence F1@10` do not fail validation unless that suite declares them required

### Requirement: Failure-case generation is metric-suite aware
The system SHALL keep failure-case generation behind the metric suite or an explicitly paired failure-case builder.

#### Scenario: Evidence failure cases preserve current behavior
- **WHEN** evidence failure cases are generated for current HotpotQA-style evidence retrieval
- **THEN** missing support and connected-gold behavior remains unchanged

#### Scenario: Alternate suite can define alternate failures
- **WHEN** a future LongMemEval suite defines turn or session support failures
- **THEN** those failure cases are generated without adding LongMemEval concepts to retriever methods

### Requirement: Dataset-aware stages project labels before evaluation
The system SHALL require dataset-aware stages or adapters to project dataset label artifacts into the eval request or eval units accepted by the selected metric suite before generic evaluation runs.

#### Scenario: HotpotQA stage projects to evidence evaluation
- **WHEN** the HotpotQA evaluation stage calls the evidence metric suite
- **THEN** it first projects HotpotQA label records into evidence labels or evidence eval units

#### Scenario: Generic evaluator does not import dataset projectors
- **WHEN** reusable evaluation service code is inspected
- **THEN** it does not import HotpotQA projectors or future LongMemEval projectors

### Requirement: Retrievers remain metric-agnostic
The system SHALL keep retrieval methods independent of dataset metric suites.

#### Scenario: Retriever output has no metric-only fields
- **WHEN** a retrieval method returns ranked nodes and optional trace
- **THEN** it does not include gold labels, turn support, session support, answer correctness, or metric column names
