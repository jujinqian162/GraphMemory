## MODIFIED Requirements

### Requirement: Trainable graph model behavior remains equivalent
The system SHALL preserve relation vocab ordering, tensorization order and dtypes, public R-GCN method ids, request-authoritative graph input, dataset-neutral evidence labels, retrieved-subgraph construction, and ranked-result contracts while intentionally replacing independent one-shot R-GCN node ranking with selection-conditioned beam evidence decoding.

#### Scenario: Stable model boundaries with new decoding behavior
- **WHEN** focused trainable graph retriever tensorization, model, training, checkpoint, and retrieval tests run after the change
- **THEN** graph tensors and public request/result invariants SHALL remain stable
- **AND** R-GCN training state, logits, checkpoint metadata, and ranking expectations SHALL reflect the beam decoder rather than the retired one-shot scorer behavior

## ADDED Requirements

### Requirement: Encoded graph state is model-owned
The graph retriever model domain SHALL expose the R-GCN encoded question and node states to its beam decoder without adding model tensors to public retrieval or dataset contracts.

#### Scenario: Decoder consumes internal encoding
- **WHEN** beam training or inference scores a partial hypothesis
- **THEN** it SHALL consume an internal encoded-graph contract owned under `graph_memory.models.graph_retriever`
- **AND** public `GraphRankingRequest`, `EvidenceLabel`, and `RankedResult` fields SHALL NOT gain hidden-state tensors

### Requirement: Beam inference remains independent of training
Checkpoint-backed beam inference SHALL construct model and decoder runtime state through the shared model factory and MUST NOT import the training lifecycle.

#### Scenario: Beam inference import boundary
- **WHEN** graph retriever inference modules are inspected
- **THEN** they SHALL NOT import beam-training or R-GCN training modules
