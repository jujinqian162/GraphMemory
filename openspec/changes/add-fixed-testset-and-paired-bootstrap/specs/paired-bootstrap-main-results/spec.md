## ADDED Requirements

### Requirement: Trainable methods report multi-seed mean and standard deviation
Main-results aggregation SHALL report, for each trainable method, the per-seed metric rows plus the mean and standard deviation of each metric across the provided seeds. Deterministic methods, run once, SHALL contribute their single metric row directly without a standard deviation.

#### Scenario: Trainable method with multiple seeds
- **WHEN** aggregation receives multiple seed rows for one trainable method
- **THEN** it emits each seed row plus the cross-seed mean and standard deviation for every shared metric

#### Scenario: Deterministic method run once
- **WHEN** aggregation receives a single row for a deterministic method
- **THEN** it emits that single value with no standard deviation and does not fabricate additional seeds

### Requirement: Paired bootstrap 95% CI against a baseline
Main-results aggregation SHALL compute, for each compared method against a declared baseline method, a per-query paired bootstrap 95% confidence interval on the baseline-minus-method delta for each shared per-task metric. The interval SHALL be produced by resampling per-query paired deltas with a fixed bootstrap seed and SHALL report the mean delta, the lower and upper bounds, and the number of paired queries.

#### Scenario: Method compared to baseline
- **WHEN** a method's per-task metrics are compared to the baseline's per-task metrics on the shared test split
- **THEN** aggregation reports the mean baseline-minus-method delta, its 95% paired bootstrap interval, and the paired query count

#### Scenario: Interval spans zero
- **WHEN** the paired 95% interval for a metric includes zero
- **THEN** aggregation reports the interval without declaring a significant difference

### Requirement: Compared methods must share the same test split
Paired analysis SHALL require that every compared method and seed evaluated identical test `task_id` sets. If any two compared per-task metric sets differ in their `task_id` membership, aggregation SHALL fail with an explicit error identifying the mismatch rather than silently intersecting or truncating.

#### Scenario: Task id sets match
- **WHEN** all compared methods and seeds share the same set of test `task_id` values
- **THEN** paired analysis proceeds and pairs each query by `task_id`

#### Scenario: Task id sets differ
- **WHEN** two compared per-task metric sets have different `task_id` membership
- **THEN** aggregation raises an error naming the differing methods or seeds and does not emit a paired interval
