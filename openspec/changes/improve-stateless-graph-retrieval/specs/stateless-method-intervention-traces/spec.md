## ADDED Requirements

### Requirement: Native traces preserve method proposal evidence
GraphRAG and execution-provenance native traces SHALL record original Dense rank/score, proposal evidence, confidence, gate outcomes, rejection/conflict reason, protected prefix, original/final rank, exact-fallback status, and actually emitted candidate edges.

#### Scenario: Proposal is rejected
- **WHEN** a bridge or path fails validity, ambiguity, hub, confidence, conflict, or no-op gates
- **THEN** native trace records one explicit reason while retrieved edges remain unchanged

### Requirement: Exact fallback is auditable
Each method trace SHALL state whether the original Dense result was returned exactly.

#### Scenario: No effective promotion exists
- **WHEN** all proposals abstain or are no-ops
- **THEN** the trace marks exact fallback and contains no emitted edge

### Requirement: Diagnostics remain method-local
Trace changes SHALL NOT alter dataset records, prepared-data identity, shared metric columns, evaluation artifact roles, or aggregate output schemas.

#### Scenario: Trace detail changes
- **WHEN** native trace evidence is extended
- **THEN** existing dataset and evaluation cache identities remain unchanged
