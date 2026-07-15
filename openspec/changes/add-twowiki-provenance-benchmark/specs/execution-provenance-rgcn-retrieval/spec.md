## ADDED Requirements

### Requirement: Provenance graph tensorization preserves typed structure
The provenance R-GCN tensorizer SHALL encode all graph nodes, node types, directed relation types, reverse message relations, query anchor, candidate mask, and contracted output dependencies without projecting the graph through EvidenceGraph.

#### Scenario: Connector node participates in message passing
- **WHEN** a ToolOutput feeds a downstream ToolCall that returns another ToolOutput
- **THEN** all three nodes and both typed relations are present in the tensorized graph while only the two ToolOutputs are candidate-scored

### Requirement: R-GCN scores ToolOutput candidates only
`execution_provenance_rgcn_retriever` SHALL apply relation-specific message passing to the full provenance graph and SHALL return a complete ranking over request candidate ToolOutputs.

#### Scenario: Full ranking
- **WHEN** inference receives a valid provenance request and checkpoint
- **THEN** every candidate output receives one score, connector nodes receive no public ranking position, and the requested top-k is derived from the complete ranking

### Requirement: Logical dependency edges are scored without beam decoding
The method SHALL independently score contracted `ToolOutput -> ToolOutput` transitions that correspond to legal `feeds` plus `returns` typed paths, SHALL select the highest-scoring legal successor per selected source, and SHALL serialize those logical and native edges in the result trace without beam state or dynamic-oracle behavior.

#### Scenario: Select a learned successor
- **WHEN** the selected first-hop output has both gold and non-gold successor branches
- **THEN** the edge head scores both legal transitions, the selected successor is the highest-scoring eligible target, and the result maps back to the exact traversed typed edges

### Requirement: Training consumes sampled pairs and supervises nodes plus edges
Training SHALL compute candidate ranking loss from the materialized positive, easy-random, BM25-hard, and dense-hard pair records selected for each task, and SHALL combine it with class-balanced gold logical-edge supervision. Training MUST NOT expose a path-decoding loss, beam decoder, or dynamic oracle.

#### Scenario: Hard-negative pair changes candidate supervision
- **WHEN** a non-gold candidate is present in the graph but absent from the materialized pair artifact
- **THEN** that candidate contributes graph messages but does not contribute candidate BCE, while configured BM25/dense hard-negative pairs do contribute candidate BCE

### Requirement: Provenance checkpoints are isolated and self-describing
Provenance R-GCN checkpoints SHALL store a distinct schema version, node-type vocabulary, relation and binding-schema vocabulary, dense encoder metadata, model dimensions, and candidate/edge training-loss configuration, and MUST reject evidence-R-GCN, stale beam-era provenance, or incompatible provenance checkpoints.

#### Scenario: Load evidence checkpoint
- **WHEN** the provenance method receives a node-wise EvidenceGraph R-GCN checkpoint
- **THEN** loading fails before inference with an explicit checkpoint-family mismatch

### Requirement: Beam-era configuration is rejected
Training and inference configs and checkpoint payloads MUST NOT expose provenance R-GCN `beam_width`, `max_steps`, `path_loss_weight`, or dynamic-oracle fields.

#### Scenario: Load stale beam-era checkpoint
- **WHEN** inference receives a provenance checkpoint containing the retired beam-era schema
- **THEN** loading fails with an explicit unsupported provenance checkpoint schema error

### Requirement: Binding schema participates in message relation identity
For `feeds` edges, tensorization SHALL validate endpoint binding hashes and SHALL distinguish the configured binding field/parameter/kind schema from unrecognized binding schemas in the relation vocabulary.

#### Scenario: Tensorize evidence-to-context binding
- **WHEN** a valid `evidence -> context` semantic binding is tensorized
- **THEN** its forward and reverse messages use the configured binding-schema-aware relation IDs rather than a generic binding-blind `feeds` relation
