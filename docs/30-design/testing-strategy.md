# Testing strategy

Tests enforce domain separation and real workflow behavior.

- Registry tests lock the seven public IDs and both family matrices.
- Cross-input tests reject provenance requests at R-GCN and evidence requests at the provenance retriever.
- GraphRAG tests cover deterministic normalization, entity propagation, complete ranking, native entity trace, semantic fallback, and absence of EvidenceGraph dependencies.
- Provenance tests cover field binding, typed transitions, restricted seeds, invalidation, complete ranking, and actual-path traces without requiring Claim or Verification nodes.
- R-GCN tests cover node-wise BCE, full ranking, checkpoint schema, batching, and device propagation.
- Workflow tests run generated typed stage YAML and assert that flat/GraphRAG-only selections do not schedule EvidenceGraph construction.

Final verification runs targeted tests, the full test suite, Ruff, basedpyright, compileall, strict OpenSpec validation, symbol scans, and `git diff --check`.
