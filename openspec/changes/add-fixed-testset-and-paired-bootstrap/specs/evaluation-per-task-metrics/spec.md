## ADDED Requirements

### Requirement: Evaluation persists per-task metrics keyed by task id
The evaluation stage SHALL persist per-task metric rows keyed by `task_id` alongside the aggregate metric row. Each per-task row SHALL contain the same node-ranking metrics that are averaged into the aggregate (at minimum Recall@2, Recall@5, Recall@10, Evidence F1@5, Evidence F1@10, Full Support@5, Full Support@10, MRR). The per-task payload SHALL be published as an evaluation artifact output and projected into the run output directory.

#### Scenario: Evaluation produces per-task output
- **WHEN** the evaluation stage completes for one method on the test split
- **THEN** it writes a per-task artifact whose entries are indexed by `task_id` and whose per-task metric values, when averaged, reproduce the corresponding aggregate metric values

#### Scenario: Per-task rows align with predictions
- **WHEN** the per-task artifact is produced
- **THEN** its set of `task_id` values equals the set of evaluated prediction `task_id` values with no duplicates

### Requirement: Per-task metrics are the source for paired statistics
The per-task metrics artifact SHALL be structured so that paired per-query statistics can be computed across methods and seeds without recomputing rankings. Aggregate-only metrics that are not defined per task (for example micro edge precision/recall) MAY be omitted from the per-task payload.

#### Scenario: Downstream paired analysis reads per-task metrics
- **WHEN** a paired analysis consumes evaluation outputs for multiple methods
- **THEN** it can read each method's per-task metrics by `task_id` and align them without invoking retrieval or evaluation again
