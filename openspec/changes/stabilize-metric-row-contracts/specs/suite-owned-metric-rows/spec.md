## ADDED Requirements

### Requirement: Metric suites own metric row shape
The system SHALL define metric row types by metric suite rather than requiring every suite to satisfy one evidence-shaped row contract.

#### Scenario: Evidence rows use evidence columns
- **WHEN** the evidence metric suite evaluates predictions
- **THEN** it SHALL return rows conforming to the evidence metric row contract with existing evidence column names and formulas preserved.

#### Scenario: LongMemEval rows use LongMemEval columns
- **WHEN** the LongMemEval metric suite evaluates predictions
- **THEN** it SHALL return rows conforming to the LongMemEval metric row contract without requiring evidence-only columns such as `Evidence F1@10`.

### Requirement: Metric suites own table schemas
The system SHALL use metric-suite-owned table schemas to choose main, path, efficiency, and wide CSV columns.

#### Scenario: Evaluation writes suite wide columns
- **WHEN** an evaluation stage writes a per-method metrics CSV
- **THEN** it SHALL use the selected metric suite wide-column schema instead of inferring columns from row contents.

#### Scenario: Aggregation writes suite table columns
- **WHEN** aggregate tables are written for a workflow run
- **THEN** the main, path, and efficiency outputs SHALL use the schema associated with the run metric suite.

### Requirement: Generic table code does not infer LongMemEval by row sniffing
Generic metric table code MUST NOT detect LongMemEval behavior by checking whether a row contains LongMemEval-only columns.

#### Scenario: Empty inputs remain deterministic
- **WHEN** aggregation receives no metric rows for a configured metric suite
- **THEN** the output columns SHALL still be selected from the configured suite schema.

#### Scenario: Suite selection is explicit
- **WHEN** generic table code needs a schema
- **THEN** it SHALL receive a suite/schema identifier or schema object from evaluation or workflow context rather than checking `Turn Recall@5` in a row.
