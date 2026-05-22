## ADDED Requirements

### Requirement: Convert HotpotQA into separated task and label artifacts
The system SHALL convert labeled HotpotQA examples into input-visible task records and separate label records. Input-visible records MUST contain `task_id`, `query`, and `memory_items`, and MUST NOT contain `gold_answer`, `gold_evidence_nodes`, `gold_dependency_edges`, `supporting_facts`, `is_gold`, `is_gold_evidence`, or `is_gold_edge`.

#### Scenario: Supporting facts map to stable memory node IDs
- **WHEN** a raw HotpotQA example contains `_id`, `question`, `answer`, `context`, and `supporting_facts`
- **THEN** conversion produces `task_id = "hotpot_" + _id`, memory node IDs `m{position}`, local `sentence_id` values, flattened `position` values, and label `gold_evidence_nodes` mapped from `(title, sentence_id)`

#### Scenario: Input artifact excludes label-only fields
- **WHEN** conversion writes `*_memory_tasks.input.json`
- **THEN** the written records contain no gold answer, supporting fact, gold evidence, gold dependency, or `is_gold*` fields anywhere

#### Scenario: Missing raw ID fails conversion
- **WHEN** a raw HotpotQA example does not contain `_id`
- **THEN** conversion fails with a clear exception instead of inventing a position-based task ID

### Requirement: Produce deterministic labeled splits
The system SHALL sample train, dev, and test records from labeled HotpotQA files using deterministic seed and offset parameters. The system MUST fail when `count` or `offset` is negative, or when `offset + count` exceeds available examples.

#### Scenario: Same seed and offset reproduce the same split
- **WHEN** split sampling is called twice with the same examples, count, seed, and offset
- **THEN** both calls return the same ordered examples

#### Scenario: Dev and test are disjoint by offset
- **WHEN** dev is sampled with offset `0` and test is sampled with offset `500` from the same shuffled labeled dev examples
- **THEN** the selected example sets do not overlap if enough examples are available

#### Scenario: Oversized split request fails
- **WHEN** a split request asks for more records than are available at the requested offset
- **THEN** the system raises an exception instead of reducing the count or falling back to another file

### Requirement: Validate artifacts fail-fast
The system SHALL validate task inputs, labels, graphs, ranked results, graph-rerank configs, metric rows, and task ID alignment at script boundaries. Validators MUST raise `ContractValidationError` for contract violations and MUST NOT repair, sort, drop, coerce, or infer invalid data.

#### Scenario: Label leakage is rejected
- **WHEN** a task input artifact contains `gold_evidence_nodes`
- **THEN** validation fails and names the offending field and task when available

#### Scenario: Label node must exist in task input
- **WHEN** a label artifact references a gold evidence node that is not present in the matching task input
- **THEN** validation fails before evaluation or tuning proceeds

### Requirement: Preserve compatibility output without weakening leakage-safe inputs
The `prepare_hotpotqa.py` script MAY write a combined `*_memory_tasks.json` compatibility artifact, but graph construction and retrieval SHALL consume only `*_memory_tasks.input.json` task artifacts.

#### Scenario: Compatibility artifact is optional
- **WHEN** `prepare_hotpotqa.py` is run without an `--output_combined` path
- **THEN** it writes input and label artifacts and records that no compatibility artifact was requested

#### Scenario: Retrieval rejects combined labels as task input
- **WHEN** a retrieval or graph-building command is given an artifact containing label-only fields
- **THEN** validation fails instead of ignoring the extra fields
