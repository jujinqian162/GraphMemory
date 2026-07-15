## ADDED Requirements

### Requirement: Registry-driven evidence graph scheduling
Evidence-graph artifacts for retrieval and evaluation SHALL be scheduled exactly when selected method metadata requires `EVIDENCE_GRAPH`.

#### Scenario: Evidence R-GCN evaluation
- **WHEN** either evidence R-GCN method is selected
- **THEN** retrieval and evaluation consume the test EvidenceGraph artifact and depend on its build invocation

#### Scenario: Flat and GraphRAG evaluation
- **WHEN** selected methods require no prebuilt artifact
- **THEN** retrieval and evaluation omit EvidenceGraph inputs and dependencies

### Requirement: Active capability surfaces are current-only
The active OpenSpec change list MUST NOT expose deleted Memory Stream as an available change and active documentation MUST NOT claim beam R-GCN or retired graph-rerank methods are current behavior.

#### Scenario: Inspect active changes after repair
- **WHEN** maintainers run `openspec list` and scan active contracts
- **THEN** deleted Memory Stream directories are absent and superseded beam/graph-rerank changes are no longer active sources of truth
