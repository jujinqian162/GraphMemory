## Context

The repository currently collects 471 pytest cases from 69 test files and carries roughly 15,000 lines of test code. The suite grew through repeated feature additions and direct-cutover refactors; many tests now preserve implementation snapshots or deleted surfaces instead of externally meaningful behavior. At the same time, the repository contains scientifically sensitive logic whose regression coverage must remain explicit: leakage-safe projections, graph construction, ranking, evaluation, train-pair generation, R-GCN and beam training, checkpoint loading, and experiment execution.

The refactor must therefore reduce maintenance surface without using coverage percentage or raw assertion count as a proxy for confidence. The authoritative verification boundary is a compact behavioral suite plus representative workflow smoke execution.

## Goals / Non-Goals

**Goals:**

- Collect no more than 100 pytest cases.
- Materially reduce test file count and total test lines.
- Retain direct evidence for scientific correctness, leakage prevention, trainable-model behavior, and supported workflow connectivity.
- Consolidate repeated lower-layer checks into scenario tests at the owning boundary.
- Document an admission policy that prevents renewed test accumulation.

**Non-Goals:**

- Changing production behavior, configuration defaults, public method identifiers, or experiment outputs.
- Preserving line or branch coverage percentages.
- Keeping migration-history assertions merely because they currently pass.
- Replacing tests with large real-data or network-dependent runs.

## Decisions

### Classify tests by protected behavior

Every retained test must protect at least one of these categories: scientific calculation, leakage/data contract, algorithm/model behavior, artifact persistence, or supported workflow execution. Tests whose only subject is a literal default, deleted name, internal `__all__`, source spelling, or third-party validation mechanic are removed.

Alternative considered: retain all current tests and only group files. Rejected because it does not reduce collection or duplicated maintenance.

### Prefer owning-boundary scenarios

Related assertions will be combined at the smallest meaningful owner. For example, one dataset conversion scenario may assert IDs, visible fields, gold labels, and dependency edges together; one workflow scenario may assert composition, planning, stage execution, and output artifacts. This replaces repeated projection/config/CLI tests that exercise the same path through adjacent layers.

Alternative considered: preserve one test per method or field. Rejected because typed discriminated models and registry-driven iteration already cover structural variation.

### Keep custom invariants, not library mechanics

Custom cross-field and scientific constraints remain covered. Exhaustive checks that Pydantic rejects extra fields, `Literal` rejects arbitrary strings, `Field(gt=0)` rejects zero, or Hydra rejects an unknown override are removed or represented by one composition-boundary smoke scenario.

### Keep one generalized architecture guard

AST-based dependency-direction enforcement remains as a single data-driven architecture test. Tombstones for individually removed modules, source strings, compatibility names, and exact export lists are deleted.

### Verify behavior after reduction

The final gate is: collected-case/file/line budget check, compact pytest suite, Ruff, BasedPyright, compileall, OpenSpec strict validation, and representative smoke workflows where the suite does not already execute the same path.

## Risks / Trade-offs

- [Risk] Consolidation can make failures less localized. → Keep scenarios aligned to one owning boundary and use descriptive assertion messages where several invariants are checked together.
- [Risk] Removing tombstone tests may allow an old name to reappear. → Treat this as code-review and architecture-direction responsibility unless reintroduction changes supported behavior.
- [Risk] A test can appear redundant while protecting a subtle scientific invariant. → Retain metric, ranking, leakage, sampling, tensorization, loss, and checkpoint tests unless an equivalent owning-boundary scenario is demonstrated.
- [Risk] The 100-case budget can encourage oversized tests. → Count coherent behavior scenarios, not arbitrary bundles; require explicit replacement or justification for future additions.

## Migration Plan

1. Record the baseline inventory and classify obvious snapshot/tombstone/library-mechanic tests.
2. Delete those tests and obsolete helpers.
3. Consolidate duplicate dataset, retrieval, trainable, and workflow checks.
4. Run the compact suite repeatedly while preserving production code unchanged.
5. Update testing guidance with the admission rule and verification tiers.
6. Validate the final budgets and full gate stack.

Rollback is a normal Git revert of test-only changes; production behavior is not migrated.

## Open Questions

None. The user explicitly selected a hard target below 100 collected cases and requested immediate OpenSpec-driven implementation.
