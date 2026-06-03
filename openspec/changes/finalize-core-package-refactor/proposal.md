## Why

The previous core package refactor changes have moved ownership into domain packages, but the repository still contains temporary root compatibility modules and lacks a final architecture guard for preventing old imports from returning. This change finishes the migration boundary so the package layout matches the approved design while preserving CLI, workflow, artifact, and retrieval/training behavior.

## What Changes

- Remove obsolete root and package compatibility paths that are no longer part of the approved workflow integration boundary.
- Delete `graph_memory.types` as the temporary aggregation layer and move its remaining owned records to their domain packages.
- Add architecture dependency tests that enforce package direction and approved root integration ports.
- Update durable docs to describe the stable post-refactor module map and workflow-facing ports.
- Run full validation, including type checking and a workflow behavior-equivalence run against the pre-refactor quick R-GCN workflow output.

## Capabilities

### New Capabilities
- `core-refactor-finalization`: Covers removal of old internal import paths, enforcement of package dependency boundaries, retained workflow integration ports, and final behavior-equivalence validation.
- `core-refactor-docs`: Covers durable documentation updates for the stabilized core package layout.

### Modified Capabilities

## Impact

- Affected code: `graph_memory/`, `scripts/`, `tests/`, `docs/`, and the OpenSpec change artifacts.
- Public CLI usage, workflow commands, retrieval method names, artifact schemas, checkpoint schema, ranking semantics, and training semantics remain unchanged.
- Internal imports from removed compatibility paths must be updated to domain-owned modules rather than supported through broad facades.
