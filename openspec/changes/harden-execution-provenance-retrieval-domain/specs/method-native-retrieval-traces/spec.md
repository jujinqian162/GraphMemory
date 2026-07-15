## ADDED Requirements

### Requirement: Native traces use a closed typed union
Retrieval results SHALL represent method-native trace data as either a typed GraphRAG trace or a typed execution-provenance trace, and MUST NOT expose a shared list of arbitrary dictionaries.

#### Scenario: GraphRAG trace serialization
- **WHEN** GraphRAG returns linked entities, seeds, and traversed relations
- **THEN** result metadata serializes an `entity_search` trace with only the declared fields

#### Scenario: Provenance trace serialization
- **WHEN** provenance retrieval selects scored paths and traversed typed edges
- **THEN** result metadata serializes an `execution_provenance` trace with typed path and edge records

### Requirement: Native traces are validated
Ranked-result validation SHALL reject unknown native trace kinds, unknown fields, non-finite scores, invalid endpoint IDs, duplicate edge records, and path node references outside the request candidates or native graph context declared by the trace.

#### Scenario: Reject malformed provenance edge
- **WHEN** serialized native trace contains a provenance edge without a target or with a non-finite weight
- **THEN** ranked-result validation fails with a trace-specific contract error
