## ADDED Requirements

### Requirement: Obsolete internal import paths are removed
The system SHALL remove obsolete root compatibility modules and old package facades after callers are migrated to domain-owned import paths.

#### Scenario: old modules cannot be imported
- **WHEN** code attempts to import `graph_memory.types`, `graph_memory.hotpotqa`, `graph_memory.splits`, `graph_memory.entities`, `graph_memory.indexes`, or `graph_memory.learned`
- **THEN** those old compatibility paths are absent rather than broad re-export facades

#### Scenario: source imports use owned domains
- **WHEN** repository Python source is scanned for imports
- **THEN** production code, scripts, and tests do not import from the removed compatibility paths

### Requirement: Workflow integration ports remain narrow
The system SHALL retain only the approved root workflow integration ports and keep them as thin imports over their owned implementations.

#### Scenario: approved root ports remain importable
- **WHEN** workflow-facing code imports `graph_memory.io`, `graph_memory.observability`, `graph_memory.retrieval_registry`, `graph_memory.training_config`, or `graph_memory.experiment`
- **THEN** the import succeeds without requiring old internal compatibility modules

#### Scenario: root ports do not expand into broad facades
- **WHEN** the root integration port files are inspected
- **THEN** each port exposes only its approved workflow-facing API and imports only approved implementation modules

### Requirement: Package dependency direction is enforced
The system SHALL provide automated architecture tests that enforce the approved core package dependency direction.

#### Scenario: lower-level packages do not import higher-level packages
- **WHEN** imports under `graph_memory/contracts`, `graph_memory/graphs`, `graph_memory/retrieval`, `graph_memory/models/graph_retriever`, and `graph_memory/infrastructure` are scanned
- **THEN** forbidden cross-domain dependencies are reported as test failures

#### Scenario: removed paths cannot return silently
- **WHEN** repository Python source is scanned
- **THEN** imports from removed old paths and newly created files at those removed paths are reported as test failures

### Requirement: Final validation proves behavior equivalence
The system SHALL validate the final package refactor with full tests, type checking, OpenSpec strict validation, and a workflow-level quick R-GCN behavior comparison.

#### Scenario: full static and dynamic validation passes
- **WHEN** final validation is run
- **THEN** the full test suite passes, basedpyright reports zero errors, ruff reports no lint failures, and OpenSpec strict validation passes

#### Scenario: quick R-GCN workflow output is behavior-equivalent
- **WHEN** a new workflow run named `rgcn_quick_train_after_refactor` is run with the same effective configuration as `rgcn_quick_train`
- **THEN** behavior-bearing intermediate artifacts match the `rgcn_quick_train` artifacts exactly, except for normalized run-local names, paths, timestamps, durations, and environment-only metadata
