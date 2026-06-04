## ADDED Requirements

### Requirement: Domain packages do not import root workflow integration ports
The system SHALL treat retained root modules as workflow integration ports, not as general internal dependencies for domain packages.

#### Scenario: Domain imports bypass root ports
- **WHEN** production imports under domain packages are scanned
- **THEN** they do not import `graph_memory.io`, `graph_memory.observability`, `graph_memory.training_config`, `graph_memory.experiment`, or `graph_memory.retrieval_registry` unless the import is explicitly allowed by the architecture test

#### Scenario: Root ports remain narrow
- **WHEN** retained root workflow integration port files are inspected
- **THEN** they only import approved owned implementation modules and do not grow new core logic

### Requirement: Durable docs describe the application boundary
The system SHALL update durable project documentation to show the application use-case layer and the tightened retrieval execution boundary.

#### Scenario: Architecture docs show the corrected layer order
- **WHEN** maintainers read the durable architecture and abstraction docs
- **THEN** they can identify the flow from scripts to application use cases to retrieval resolver/factory/execution without relying on the time-bound refactor plan

#### Scenario: Contract docs distinguish use-case request from method build request
- **WHEN** maintainers read retrieval contract docs
- **THEN** they can distinguish `RunRetrievalRequest`, `RetrievalMethodResolveRequest`, method-family build requests, and `RetrievalMethod` execution

### Requirement: Boundary tests cover the previous review gaps
The system SHALL include focused tests for the exact weaknesses found in the implementation review.

#### Scenario: Loose dense prefix scan covers execution and tuning
- **WHEN** architecture boundary tests scan retrieval modules
- **THEN** they fail if loose `query_prefix` or `passage_prefix` fields reappear in retrieval execution, resolver, factory, or tuning service internals outside dense-owned config/runtime construction

#### Scenario: Root-port import scan catches domain regressions
- **WHEN** architecture boundary tests scan domain packages
- **THEN** they fail if a domain package imports retained root workflow integration ports instead of owned implementation modules
