## 1. Final Boundary Tests

- [x] 1.1 Add focused architecture/import tests that fail while `graph_memory.types` and old root compatibility paths remain importable
- [x] 1.2 Add architecture dependency tests for package direction and approved root workflow integration ports
- [x] 1.3 Verify the new tests fail for the expected pre-implementation reasons

## 2. Remove Temporary Compatibility Surfaces

- [x] 2.1 Move the remaining `graph_memory.types` records to domain-owned modules and update production/script imports
- [x] 2.2 Update tests to import domain-owned contracts and configs instead of removed compatibility paths
- [x] 2.3 Delete obsolete root compatibility files and old package directories after residual import searches are clean
- [x] 2.4 Verify focused tests and residual import searches for removed old paths

## 3. Durable Documentation

- [x] 3.1 Update architecture, abstraction, retrieval/model contract, testing, handoff, command, and docs index references to the final package layout
- [x] 3.2 Verify documentation no longer presents removed root paths as current implementation locations

## 4. Final Validation

- [x] 4.1 Run the full test suite
- [x] 4.2 Run basedpyright error-level type checking and fix all errors
- [x] 4.3 Run ruff and OpenSpec strict validation
- [x] 4.4 Run `rgcn_quick_train_after_refactor` with the same effective workflow configuration as `rgcn_quick_train`
- [x] 4.5 Compare behavior-bearing intermediate artifacts from `rgcn_quick_train_after_refactor` against `rgcn_quick_train` and document the result
