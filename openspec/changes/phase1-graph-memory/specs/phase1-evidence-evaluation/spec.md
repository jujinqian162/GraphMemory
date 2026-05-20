## ADDED Requirements

### Requirement: Tune graph rerank parameters on dev labels only
The system SHALL provide graph-rerank grid search that uses dev task inputs, dev labels, and dev graphs to select graph-rerank parameters. The system MUST NOT use test labels for parameter selection.

#### Scenario: Objective selects best dev configuration
- **WHEN** tuning receives candidate metric rows
- **THEN** it selects the config maximizing `0.50 * Full Support@5 + 0.30 * Recall@5 + 0.20 * Connected Evidence Recall@10`

#### Scenario: Ties prefer lower latency after metric tie-breakers
- **WHEN** candidate configs have equal objective and equal higher-priority tie-break metrics
- **THEN** tuning selects the candidate with lower retrieval latency

#### Scenario: Selected config is persisted
- **WHEN** tuning completes
- **THEN** it writes a graph-rerank config JSON that can be supplied to fixed test retrieval runs

### Requirement: Evaluate predictions from labels and shared graphs
The system SHALL evaluate ranked predictions by joining predictions, label artifacts, and graph artifacts by `task_id`. Evaluation MUST NOT read gold labels from input task artifacts and MUST fail if prediction, label, and graph task IDs do not align exactly.

#### Scenario: Node metrics use top-k ranked nodes
- **WHEN** evaluation receives a complete ranking and matching gold evidence nodes
- **THEN** it computes Recall@2, Recall@5, Recall@10, Evidence F1@5, Evidence F1@10, Full Support@5, Full Support@10, and MRR from ranked node IDs

#### Scenario: Connected evidence uses shared graph
- **WHEN** a flat method has no emitted graph edges
- **THEN** Connected Evidence Recall@k is still computed from the shared constructed graph and the method's top-k selected nodes

#### Scenario: Query-evidence connectivity includes question node
- **WHEN** top-10 contains all gold evidence nodes
- **THEN** Query-Evidence Connectivity@10 checks reachability from `q` to each gold evidence node in the induced graph over `q` plus selected top-10 nodes

#### Scenario: HotpotQA path and edge recall are not applicable
- **WHEN** evaluating HotpotQA-only Phase 1 predictions
- **THEN** `Path Recall@10` and `Edge Recall@10` are emitted as `N/A`

### Requirement: Aggregate Phase 1 result tables
The system SHALL aggregate per-method wide metric CSVs into canonical Phase 1 `main_results.csv`, `path_results.csv`, and `efficiency_results.csv` outputs with the documented columns.

#### Scenario: Main table includes node metrics
- **WHEN** aggregation reads per-method metric rows
- **THEN** `main_results.csv` contains Method, Recall@2, Recall@5, Recall@10, Evidence F1@5, Evidence F1@10, Full Support@5, Full Support@10, and MRR

#### Scenario: Path table includes connectivity metrics
- **WHEN** aggregation reads per-method metric rows
- **THEN** `path_results.csv` contains Method, Connected Evidence Recall@5, Connected Evidence Recall@10, Query-Evidence Connectivity@10, Path Recall@10, and Edge Recall@10

#### Scenario: Efficiency table includes operational metrics
- **WHEN** aggregation reads per-method metric rows
- **THEN** `efficiency_results.csv` contains Method, Index Build Time, Graph Construction Time, Retrieval Latency / Query, Memory Size, Avg Retrieved Nodes, and Avg Retrieved Edges

### Requirement: Preserve run summaries and debug artifacts
Every runnable script SHALL write a compact run summary near its primary output when it has enough path context. Large debug artifacts MUST be optional and bounded.

#### Scenario: Successful script writes run summary
- **WHEN** a CLI script completes successfully
- **THEN** it writes a JSON run summary containing script name, timestamps, status, effective config, inputs, outputs, counts, timings, environment, and notes

#### Scenario: Failed script writes summary when possible
- **WHEN** a CLI script fails after output paths are known
- **THEN** it attempts to write a failed run summary with the error message

#### Scenario: Debug output is bounded
- **WHEN** a script writes per-task debug records
- **THEN** it respects the configured debug limit and records truncation in a summary or debug metadata

### Requirement: Document commands and implementation handoff
The system SHALL update the command runbook and implementation handoff after the Phase 1 implementation exists. The implementation MUST NOT be considered review-ready until these documents contain real script paths, function names, command examples, verification commands, and known limitations.

#### Scenario: Command runbook reflects actual CLI arguments
- **WHEN** implementation is complete
- **THEN** `docs/40-operations/commands.md` shows the actual leakage-safe command sequence for preparation, graph construction, retrieval, tuning, evaluation, aggregation, leakage checks, and tests

#### Scenario: Handoff identifies review entry points
- **WHEN** implementation is complete
- **THEN** `docs/40-operations/implementation-handoff.md` explains the reading order, main control flow, key abstractions, file map, review checklist, verification results, known limitations, and extension points
