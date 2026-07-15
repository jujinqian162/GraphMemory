## ADDED Requirements

### Requirement: Dataset raw source kind is explicit
Every dataset configuration SHALL declare whether its raw split source is a file or directory, and the resolved experiment configuration SHALL preserve that declaration.

#### Scenario: Existing file-backed dataset
- **WHEN** HotpotQA, 2WikiMultiHopQA, or MuSiQue configuration is resolved
- **THEN** each prepare invocation binds its raw source as a file artifact

#### Scenario: Directory-backed dataset
- **WHEN** TRAJECT-Bench configuration is resolved
- **THEN** each prepare invocation binds its raw source as a directory artifact

### Requirement: Stage status validates the declared raw artifact kind
The workflow planner and status inspector SHALL validate an external raw input against its declared artifact kind without hardcoded dataset-name branches.

#### Scenario: Missing directory source
- **WHEN** a directory-backed dataset source path does not exist as a directory
- **THEN** the prepare invocation reports the external dependency as missing or invalid rather than accepting a file at that path

#### Scenario: Plan identity includes source kind
- **WHEN** the declared raw source kind changes for an otherwise identical experiment configuration
- **THEN** the resolved plan and persisted input binding differ, invalidating stale stage identity

