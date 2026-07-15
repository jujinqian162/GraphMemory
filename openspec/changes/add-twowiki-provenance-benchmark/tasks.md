## 1. Contract and regression tests

- [x] 1.1 Add failing dataset conversion, graph-invariant, projection, and leakage tests for `twowiki_provenance`
- [x] 1.2 Add failing TRAJECT capability tests that reject EvidenceGraph and execution-provenance projection
- [x] 1.3 Add failing stateless provenance tests for directed beam expansion, seed budget, `top_k`, binding, and logical traces
- [x] 1.4 Add failing Registry and model tests for the new provenance R-GCN request/checkpoint boundary

## 2. 2Wiki provenance conversion and adapter

- [x] 2.1 Add typed generated-raw records, parser, schema version, and cross-record validators under `graph_memory/datasets/twowiki_provenance`
- [x] 2.2 Implement deterministic 2Wiki gold-chain recovery, hard-negative branch construction, split policy, manifest, and statistics conversion service
- [x] 2.3 Add the one-time conversion CLI and a dataset prepare CLI/config using existing prepare-stage semantics
- [x] 2.4 Implement text, GraphRAG, ExecutionProvenance, and EvidenceEvaluation projectors with candidate/label separation
- [x] 2.5 Register dataset selection, validation, source/config models, and method compatibility without changing standard `twowiki`

## 3. TRAJECT semantic correction

- [x] 3.1 Remove catalog-connection EvidenceGraph sequential projection and native execution-provenance projection
- [x] 3.2 Restrict TRAJECT to honest flat/GraphRAG capability and add explicit unsupported-family errors
- [x] 3.3 Update TRAJECT tests and active documentation to describe offline tool-catalog retrieval only

## 4. Stateless execution provenance beam search

- [x] 4.1 Extend method config validation with beam width and effective seed-budget semantics
- [x] 4.2 Replace reverse BFS enumeration with legal directed incremental beam expansion and bounded pruning
- [x] 4.3 Implement semantic/path score fusion, binding-aware completeness, invalidation, and normalized length behavior
- [x] 4.4 Honor retrieval `top_k` and emit native typed paths plus contracted logical candidate edges
- [x] 4.5 Preserve builder/config behavior and pass focused provenance regressions

## 5. Provenance-native R-GCN

- [x] 5.1 Add provenance tensor records and tensorizer for all node/relation types, reverse message relations, query anchor, candidate mask, and logical transitions
- [x] 5.2 Add relation-aware provenance R-GCN encoder and ToolOutput-only ranking head
- [x] 5.3 Add an independent logical-edge head over contracted output transitions
- [x] 5.4 Add pair-materialized candidate loss, class-balanced edge loss, batching, trainer, dev evaluation, and checkpoint round-trip with family validation
- [x] 5.5 Add non-beam inference returning full rankings and independently selected typed/logical edge traces

## 6. Registry and existing-workflow integration

- [x] 6.1 Register `execution_provenance_rgcn_retriever`, its concrete build payload, config, capabilities, and method config YAML
- [x] 6.2 Route provenance pair building and training through existing pair/train stage types without scheduling EvidenceGraph construction
- [x] 6.3 Route dataset requests and checkpoints through existing retrieve/evaluate/aggregate stages and preserve numeric path metrics
- [x] 6.4 Add a small `twowiki_provenance` experiment config and planner/status tests proving the existing DAG is reused

## 7. Verification and documentation

- [x] 7.1 Generate a fixture/pilot conversion twice and verify byte-identical output, complete gold paths, non-isolation, and forbidden-field absence
- [x] 7.2 Run focused dataset, TRAJECT, provenance search, provenance R-GCN, Registry, and workflow tests
- [x] 7.3 Run formatter, linter/type-checker gates and strict OpenSpec validation
- [x] 7.4 Update operations/config documentation with conversion, prepare, smoke-run, metric, and synthetic-benchmark disclosure commands

## 8. Retire TRAJECT-Bench

- [x] 8.1 Remove the TRAJECT dataset package, validator, prepare script, dataset config, and local raw artifact tree
- [x] 8.2 Remove TRAJECT routing and the now-redundant directory-source configuration contract
- [x] 8.3 Remove TRAJECT-specific tests, active documentation, and the obsolete standalone OpenSpec change
- [x] 8.4 Run dataset enumeration, workflow, full test, static-analysis, and strict OpenSpec verification gates

## 9. Review-driven semantic graph and non-beam R-GCN correction

- [x] 9.1 Add failing tests for BM25/dense semantic successor edges, ambiguous-chain rejection, endpoint-verified bindings, consumed hard-negative pairs, and absence of provenance R-GCN beam/oracle fields
- [x] 9.2 Replace shuffled-ring graph construction with deterministic BM25/dense/hybrid successor scoring, fixed semantic out-degree, gold recall fallback, and auditable manifest/statistics fields
- [x] 9.3 Reject duplicated or ambiguous evidence-to-support mappings before conversion and extend validation for endpoint-consistent bindings
- [x] 9.4 Route provenance pair construction through real easy/BM25/dense samplers and make candidate BCE consume the materialized pair subset
- [x] 9.5 Remove provenance R-GCN decoder, dynamic oracle, beam/max-step/path-loss config and checkpoint fields; replace inference with independent learned logical-edge selection
- [x] 9.6 Implement endpoint-validated binding consistency in stateless search and binding-schema-aware provenance R-GCN relation IDs
- [x] 9.7 Update configs/docs and run focused tests, full pytest, Ruff, BasedPyright, strict OpenSpec validation, and diff checks
