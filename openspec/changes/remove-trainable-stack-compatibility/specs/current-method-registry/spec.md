## ADDED Requirements

### Requirement: Complete method definitions
The registry SHALL expose one typed method definition for each retrieval method, including lifecycle, graph input source, graph config source, model source, encoder source, and train artifact descriptor.

#### Scenario: Inspect Dense-FT semantics
- **WHEN** workflow code requests the Dense-FT method definition
- **THEN** the definition identifies a trained model directory artifact and the configured sources for model and encoder data

#### Scenario: Inspect R-GCN semantics
- **WHEN** workflow code requests the R-GCN method definition
- **THEN** the definition identifies a trained checkpoint file artifact and the configured sources for graph, model, and encoder data

### Requirement: No compatibility projections
Callers MUST consume the current method registry directly and MUST NOT depend on registry projection modules, legacy catalog modules, builder identifiers, or capability boolean combinations.

#### Scenario: Enumerate methods
- **WHEN** experiment, tuning, validation, or workflow code enumerates retrieval methods
- **THEN** it obtains method definitions from the current registry without an intermediate compatibility view

### Requirement: Artifact shape validation
Workflow status validation SHALL distinguish file artifacts from directory artifacts using the method definition.

#### Scenario: Validate a model directory
- **WHEN** a Dense-FT train artifact path exists as a file rather than a directory
- **THEN** artifact validation fails
