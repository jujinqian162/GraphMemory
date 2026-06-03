## ADDED Requirements

### Requirement: Durable docs describe the post-refactor package layout
The system SHALL update long-lived project documentation to point readers to the final domain-owned package paths and retained workflow integration ports.

#### Scenario: architecture docs reflect final module ownership
- **WHEN** maintainers read the durable architecture and abstraction documents
- **THEN** they can identify the final domain packages, dependency direction, and narrow workflow integration ports without relying on the time-bound refactor plan

### Requirement: Operations docs preserve CLI behavior while updating navigation
The system SHALL keep operational command guidance unchanged where CLI behavior is unchanged, while updating implementation navigation to the new package paths.

#### Scenario: command docs remain stable
- **WHEN** users follow the operation commands after the refactor
- **THEN** they use the same public CLI commands and method names as before the refactor

#### Scenario: handoff docs point to owned modules
- **WHEN** maintainers use implementation handoff or contract docs to locate code
- **THEN** old root module paths are not presented as the current implementation location
