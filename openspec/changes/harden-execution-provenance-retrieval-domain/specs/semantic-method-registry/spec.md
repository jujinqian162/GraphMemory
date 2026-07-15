## ADDED Requirements

### Requirement: Dense runtime semantics are preserved
Registry builders for GraphRAG and execution-provenance retrieval SHALL preserve the configured model name, query prefix, passage prefix, batch size, normalization behavior, and injected encoder when constructing semantic scorers.

#### Scenario: Prefix-aware injected encoder
- **WHEN** a recording encoder is injected with non-empty query and passage prefixes
- **THEN** GraphRAG and provenance scorer calls contain the configured prefixes on the corresponding texts

### Requirement: Artifact metadata is authoritative
Workflow routing SHALL derive retrieval and evaluation evidence-graph requirements from `MethodDefinition.input_spec.required_artifact` and MUST NOT maintain a parallel hard-coded set of graph-backed method IDs.

#### Scenario: Plan a GraphRAG-only workflow
- **WHEN** GraphRAG declares `RequiredArtifact.NONE`
- **THEN** planner omits evidence-graph retrieval and evaluation dependencies without checking the GraphRAG method ID explicitly
