## ADDED Requirements

### Requirement: CLI parser contracts are frozen before refactor
The system SHALL provide tests that assert the parser contract for each public script affected by the core refactor without relying on formatted help output.

#### Scenario: Parser action contract remains stable
- **WHEN** the parser contract tests inspect affected script parsers
- **THEN** they assert argument names, required flags, defaults, choices, and compatibility aliases that workflow or users depend on

### Requirement: Workflow planning contracts are frozen before refactor
The system SHALL provide tests that assert current experiment workflow planning behavior using a tiny temporary run root.

#### Scenario: Workflow plan contract remains stable
- **WHEN** the workflow contract tests initialize and plan representative experiment configurations
- **THEN** they assert manifest core fields, stage order, command arguments, method narrowing, profile mapping, ablation selection, and `--ablations-only` fail-fast behavior after normalizing time and absolute temporary paths

### Requirement: Deterministic domain fixtures are frozen before refactor
The system SHALL provide tiny deterministic fixtures for low-level domain behavior that this change will move.

#### Scenario: Foundation fixture outputs remain stable
- **WHEN** tests exercise HotpotQA conversion, text helpers, graph construction, BM25 or fake-dense ranking, graph rerank, train-pair construction, graph tensorization, model forward, and one-step CPU training where practical
- **THEN** they compare deterministic artifacts, scores, tensor values, logits, and state updates against the current implementation behavior

### Requirement: Baseline validation commands are recorded
The system SHALL record the pre-migration validation baseline for tests, type checking, and OpenSpec strict validation.

#### Scenario: Baseline commands have current evidence
- **WHEN** Batch 0 is completed
- **THEN** the change tasks or implementation notes identify the commands used for pytest, type checking at error level, and strict OpenSpec validation, including any local fallback from `uv run` to the repository virtual environment
