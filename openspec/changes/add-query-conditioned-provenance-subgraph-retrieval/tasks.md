## 1. Frozen Configuration and Algorithm Types

- [x] 1.1 Add the `ppr_steiner` default variant, frozen relation-description/version constants, and validated PPR/transition/selection parameters while disabling global `w_eff` multiplication for legacy typed-beam diagnostics.
- [x] 1.2 Add immutable typed records for relation affinity, normalized arcs, PPR state, connector paths, marginal selections, and the final selected subgraph.

## 2. Query-Conditioned Typed Diffusion

- [x] 2.1 Implement cached frozen-encoder relation embeddings and deterministic query-to-relation affinities without dataset or label access.
- [x] 2.2 Implement binding-aware forward/reverse arcs, source-local transition normalization, constant-weight identity, hub degree control, and deterministic ordering.
- [x] 2.3 Implement bounded deterministic Personalized PageRank with dangling-mass conservation, convergence diagnostics, and probability invariants.

## 3. Budgeted Connected-Subgraph Extraction

- [x] 3.1 Implement candidate prizes from Dense teleport relevance and PPR mass plus deterministic directed minimum-cost connector paths.
- [x] 3.2 Implement positive-marginal budgeted Steiner-style selection, connector accounting, evidence budget enforcement, stable tie-breaking, and full-list ranking projection.
- [x] 3.3 Implement selected-edge collapse with native orientation, duplicate suppression, and exact Dense fallback when no auditable candidate dependency is selected.

## 4. Contracts, Serialization, and Validation

- [x] 4.1 Add the closed `execution_provenance_subgraph` native trace dataclasses/models and include them in the native trace union.
- [x] 4.2 Serialize relation, transition, PPR, selection, connector, objective, emitted-edge, convergence, variant, and fallback records.
- [x] 4.3 Extend ranked-result validation for the new trace kind, source-row normalization, graph references, connectivity, direction, top-k budget, duplicates, finite values, and fallback consistency.

## 5. Registry and Experiment Integration

- [x] 5.1 Route `execution_provenance_retriever` default construction through `ppr_steiner` while preserving explicit legacy diagnostics and the public method id.
- [x] 5.2 Ensure resolved configs, Hydra YAML, CLI runner, MLflow variant tag, and Prefect ranking cache identity include every frozen behavior parameter and relation-description version.

## 6. Verification and Documentation

- [x] 6.1 Add unit tests for relation conditioning, source-local calibrated weights, constant-weight identity, invalid binding exclusion, PPR convergence/mass conservation, and deterministic repeatability.
- [x] 6.2 Add tiny exhaustive-oracle and domain tests for connector budgeting, connected selection, hub control, native orientation, fallback identity, and RQ2/RQ3-shaped graphs.
- [x] 6.3 Add serializer/validator/registry/cache round-trip tests for both the new default and legacy diagnostic variants.
- [x] 6.4 Update operations/contracts/design/paper-facing method descriptions and server run commands; mark quality experiments as pending user execution.
- [x] 6.5 Run focused tests, full pytest, Ruff, and basedpyright; document only pre-existing residual diagnostics.
