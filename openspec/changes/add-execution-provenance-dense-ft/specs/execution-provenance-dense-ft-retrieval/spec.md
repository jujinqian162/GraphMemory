## ADDED Requirements

### Requirement: Execution-provenance Dense-FT training inputs
The system SHALL train the existing `dense_ft` method on execution-provenance datasets from dataset-owned `TextRankingRequest` candidates, label-side gold evidence IDs, and materialized text-negative pairs. It MUST NOT provide ExecutionProvenanceGraph topology, bindings, native traces, answer text, or gold dependency edges to the Dense-FT model.

#### Scenario: Build Dense-FT train payload
- **WHEN** a `twowiki_provenance` run reaches the `dense_ft` train stage
- **THEN** the payload contains flat question/ToolOutput text requests, aligned evidence labels, and train pairs but no EvidenceGraph or ExecutionProvenanceGraph input

### Requirement: Text-only negative sampling
The execution-provenance Dense-FT pair stage SHALL omit EvidenceGraph artifacts and SHALL emit an effective sampling configuration with `hard_graph_neighbor_per_positive=0`. Easy-random, BM25, and dense hard-negative settings SHALL otherwise retain their configured values.

#### Scenario: Plan provenance Dense-FT pairs
- **WHEN** a run selects `dense_ft` on `twowiki_provenance`
- **THEN** its pair invocation has no EvidenceGraph input or dependency and its stage config sets graph-neighbor negatives to zero

#### Scenario: Preserve evidence Dense-FT sampling
- **WHEN** a run selects `dense_ft` on an evidence-retrieval dataset
- **THEN** its pair invocation retains the configured EvidenceGraph input and graph-neighbor negative count

### Requirement: Flat Dense-FT retrieval lifecycle
The execution-provenance Dense-FT workflow SHALL use the existing `pairs -> train -> retrieve -> evaluate -> aggregate` lifecycle, save the existing model-directory artifact and metadata, and return a complete flat ranking over ToolOutput candidate IDs under method ID `dense_ft`. It SHALL NOT claim native-edge trace capability.

#### Scenario: Complete provenance Dense-FT smoke run
- **WHEN** a one-example CPU smoke run selects only `dense_ft` on `twowiki_provenance`
- **THEN** the run produces pair artifacts, `learned/dense_ft/checkpoints/best_model`, test predictions, evaluation metrics, and an aggregate row without scheduling an EvidenceGraph stage

### Requirement: Strict task-family execution
The Dense-FT retrieval builder MUST validate the concrete flat payload as execution-provenance family input before loading or running the model directory.

#### Scenario: Execute Dense-FT provenance retrieval
- **WHEN** the retrieve stage builds `dense_ft` from provenance text requests and an execution-provenance `FlatRetrievalBuildPayload`
- **THEN** Registry accepts the request family and preserves the existing strict model-directory metadata checks
