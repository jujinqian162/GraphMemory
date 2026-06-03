## 1. Batch 5 - Retrieval Core, Flat Methods, Resolver, and Factory

- [x] 1.1 Add focused architecture/import tests that fail while `RetrievalBuildContext` and old root retrieval/rerank/tuning modules remain
- [x] 1.2 Create retrieval package modules for contracts, requests, resolver, factory, execution result assembly, flat BM25/dense methods, and seed signals
- [x] 1.3 Replace `RetrievalBuildContext` with method-family build requests and dense runtime/config composition
- [x] 1.4 Update scripts and tests to use the new retrieval package public entries without changing CLI parser contracts or public method names
- [x] 1.5 Verify BM25, fake-dense, graph-rerank-through-factory, trainable factory delegation, and parser contract focused tests

## 2. Batch 6 - Graph Rerank and Tuning Domains

- [x] 2.1 Move graph-rerank config parsing, scoring components, candidate expansion, normalization, debug helpers, and method adapter into retrieval graph-rerank modules
- [x] 2.2 Move graph-rerank grid generation, objective scoring, best-config selection, initial-score-cache usage, and tuning service into retrieval tuning modules
- [x] 2.3 Update scripts and tests to import graph-rerank and tuning behavior from owned retrieval modules
- [x] 2.4 Remove old root retrieval/rerank/rerank_config/tuning modules after residual import searches are clean
- [x] 2.5 Run focused retrieval/tuning tests, full pytest, type checking at error level, and strict OpenSpec validation
