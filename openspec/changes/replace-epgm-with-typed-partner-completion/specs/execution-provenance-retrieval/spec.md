## MODIFIED Requirements

### Requirement: Default training-free provenance retrieval is typed partner completion
`execution_provenance_retriever` SHALL implement exactly one non-trained algorithm, typed partner completion, while preserving the `ExecutionProvenanceRankingRequest` input, the full candidate ranking output, the public registry id, and its non-trained status. The method config SHALL NOT expose a `variant` axis.

#### Scenario: Registry constructs the single algorithm
- **WHEN** the method config specifies `execution_provenance_retriever`
- **THEN** the registry builds the typed partner completion retriever with no variant selection

#### Scenario: Variant selection is rejected
- **WHEN** a config supplies a `variant` key for this method
- **THEN** config validation fails because the field no longer exists

#### Scenario: One config serves both graph families
- **WHEN** the same resolved method config is applied to a synthetic dependency graph and to a real agent-trace graph
- **THEN** both are retrieved by the same code path with the same parameter values

### Requirement: Frozen parameters participate in run identity
Every behavior-affecting traversal, relation-description, scoring, and acceptance parameter SHALL participate in the resolved ranking config and in the ranking cache identity. The EPGM cache identity prefix SHALL change so that artifacts produced by the removed variants cannot be reused, while identities of all other retrieval methods remain unchanged.

#### Scenario: Confidence threshold changes cache key
- **WHEN** otherwise identical runs use different `min_partner_confidence` values
- **THEN** they cannot reuse the same ranking artifact

#### Scenario: Other method caches survive
- **WHEN** the EPGM implementation version advances
- **THEN** BM25, Dense, GraphRAG, evidence-RGCN, and provenance-RGCN ranking identities are unaffected and their cached artifacts remain valid

## REMOVED Requirements

### Requirement: Legacy path mode is explicit
**Reason**: The `typed_beam` and `dependency_path` diagnostics are each measurably an identity function on one of the two target datasets — `dependency_path` returns byte-identical results to Dense on RQ3, and both encode dataset-specific hand-authored gates or priors. Retaining them as ablations would present overfitted variants as scientific comparisons.

**Migration**: Delete `configs/method/execution_provenance_retriever_dependency_path.yaml` and drop the `variant` field from the EPGM method config. Previously reported variant numbers remain reproducible from git history and the archived `results/*-twp-sd13-v4` artifacts.
