## ADDED Requirements

### Requirement: Central validation surfaces are removed
The completed change SHALL delete `graph_memory/validation/` and `graph_memory/contracts/ranking.py`. Production and test code MUST NOT import either surface, and no compatibility package, re-export module, forwarding function, or deprecated alias SHALL preserve their APIs.

#### Scenario: Repository paths are inspected
- **WHEN** the migration is complete
- **THEN** neither `graph_memory/validation/` nor `graph_memory/contracts/ranking.py` exists

#### Scenario: Old import is attempted
- **WHEN** code imports `validate_ranked_results`, `validate_graphs`, dataset validators, pair validators, metric validators, model/checkpoint validators, or `RankedResult` from the retired modules
- **THEN** the import is absent from maintained code rather than redirected through a compatibility shim

### Requirement: Detached schema machinery is removed
Copied allowed-field constants, duplicated method/sample/edge allowlists, generic `_required_*` record readers, hand-written unknown-field rejection, hand-written Pydantic error formatting, trace-kind validation dispatch, trace-kind serialization dispatch, and `to_json_dict` methods that duplicate model fields SHALL be removed after their owning models are active.

#### Scenario: New trace member is added
- **WHEN** a future method adds a native trace kind
- **THEN** the contributor adds one closed union member and its custom validators without editing a parallel serializer branch and validation dispatch branch

#### Scenario: Model field is added
- **WHEN** a future scientific model gains a field
- **THEN** JSON dumping and parsing derive the field from the model and no copied allowed-field set is updated

### Requirement: Validation failures use owning boundary errors
Scientific schema and model-invariant failures SHALL surface as Pydantic `ValidationError` with field locations. Computation preconditions and numerical runtime failures SHALL use errors owned by their computation domain. The migration MUST NOT wrap every Pydantic failure in the retired `ContractValidationError` solely to preserve old exception text.

#### Scenario: Invalid artifact is parsed
- **WHEN** a model or aggregate invariant rejects an artifact
- **THEN** callers receive a Pydantic validation error identifying the model field or root invariant

#### Scenario: Algorithm precondition fails
- **WHEN** a runtime request such as non-positive `top_k` violates an algorithm precondition rather than an artifact schema
- **THEN** the owning retrieval API raises its direct domain/precondition error

### Requirement: Historical late-failure regression is permanently covered
The owning-boundary test suite SHALL include a regression proving that every registered retrieval method can be represented by the ranked-result method field without updating a second allowlist, and SHALL include immediate-failure tests for task-local ranking and trace defects.

#### Scenario: Registry method inventory is exercised
- **WHEN** tests iterate the active `RetrievalMethodId` values
- **THEN** the ranked-result model accepts every active value and rejects an unrelated value

#### Scenario: One result is malformed
- **WHEN** a fake retrieval method returns a malformed result for the first task
- **THEN** a focused test proves the second task is never invoked

### Requirement: Behavioral validation coverage is preserved without testing Pydantic mechanics exhaustively
Tests SHALL retain representative cases for each custom scientific invariant migrated from graphs, dataset records, provenance records, rankings/traces, train pairs, model/checkpoint configuration, metric suites, and cross-record joins. Tests SHOULD use parameterized owning-boundary cases and MUST NOT expand into exhaustive demonstrations that Pydantic itself rejects every ordinary type or extra field.

#### Scenario: Migrated invariant inventory is reviewed
- **WHEN** the old validation modules are removed
- **THEN** a migration matrix maps every public validator and every custom invariant family to an owning model and focused behavioral test

#### Scenario: Third-party behavior is sufficient
- **WHEN** rejection is fully provided by an ordinary Pydantic field type with no project-specific consequence
- **THEN** the compact suite may cover it through one representative model-boundary case instead of one test per field

### Requirement: Architecture guards prevent validation drift
Architecture tests SHALL prevent reintroduction of the retired package, ranking TypedDict contract, unchecked scientific JSON casts, upward imports from low-level models, and independent method/trace/field allowlists used only for validation.

#### Scenario: A contributor recreates a central validator module
- **WHEN** a new production import targets `graph_memory.validation` or a replacement central validation package
- **THEN** the architecture test fails and directs validation to the owning model

#### Scenario: A stage casts decoded JSON
- **WHEN** a stage directly casts a read payload to a scientific contract type
- **THEN** an AST-based boundary test fails

### Requirement: Documentation and OpenSpec targets reflect the new architecture
Maintained architecture and operations documentation SHALL remove `validation/` as a package boundary, identify domain-owned Pydantic models and typed artifact reads as the extension points, and explain the distinction between schema/model validation and tensor/algorithm runtime checks. New OpenSpec work MUST target owning models rather than the retired validator paths.

#### Scenario: Architecture package map is read
- **WHEN** a contributor consults maintained architecture documentation
- **THEN** it shows models under their owning datasets, graphs, retrieval, training, model, and evaluation domains and contains no maintained instruction to add `graph_memory/validation/*`

### Requirement: No legacy compatibility or retired-schema fallback is introduced
The migration SHALL update all maintained producers, consumers, tests, fixtures, and docs in one coordinated change. It MUST NOT retain dictionary-like Pydantic wrappers, mapping facades, old exception adapters, dual model/TypedDict aliases, or retired-schema conversion branches solely to reduce call-site edits.

#### Scenario: Existing dictionary-indexing caller is migrated
- **WHEN** a caller currently reads a contract with `record["field"]`
- **THEN** it is changed to consume the model API or an explicit JSON dump rather than receiving a mapping-compatibility model

### Requirement: Full engineering gates validate retirement
Completion SHALL require focused contract tests, the compact full pytest suite, Ruff, basedpyright at error level, compileall, Git diff checks, strict OpenSpec validation, and repository scans for retired imports and paths.

#### Scenario: Change is declared complete
- **WHEN** every implementation task is checked off
- **THEN** all declared engineering gates pass and the residual scans report no retired validation surface
