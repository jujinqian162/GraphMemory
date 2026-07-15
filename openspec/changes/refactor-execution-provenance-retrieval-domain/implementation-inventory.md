# Implementation Inventory

> 根约束：[`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](../../../docs/10-plans/execution-provenance-retrieval-domain-plan.md)。本清单记录实施前活动面，用于删除和迁移核对，不构成兼容承诺。

## Baseline snapshot

- Memory Stream symbols: 64 files across active code/config/scripts/tests/docs and historical OpenSpec; production ownership集中于 dataset projectors/selection、experiment config/planning/stage models、Registry/builders、retrieval requests/method/tuning、retrieve stage、importance validation 与 CLI/config。
- Old graph rerank symbols: 76 files; production ownership集中于 Registry/builders、`retrieval/methods/graph_rerank`、tuning、stage dispatch、config/search space 和 operations docs。
- Ambiguous evidence graph names: 82 files; production ownership覆盖 contracts、graph construction/views/index/statistics、三类 dataset projectors、evaluation、R-GCN tensorization/training/inference、training pairs、stages 与 scripts。
- Beam/frontier/oracle symbols: 47 files; production ownership集中于 R-GCN decoder/oracle/beam loss、neural model、training/inference/dev evaluation、config/validation、trainer wiring 和三份专用 tests。

## Migration ownership

| Area | Current owners | Target action |
|---|---|---|
| Memory Stream | `retrieval/methods/memory_stream`, `retrieval/tuning/memory_stream*`, temporal request/projectors, Registry/builders, experiment/stage/validation, CLI/config/tests | 全量删除；旧 method/config unsupported；不保留 alias、re-export 或 artifact loader |
| Evidence graph | `contracts/graphs.py`, `graphs/*`, dataset projectors/selection, evaluation, R-GCN, training pairs, stages/scripts | 机械迁移为 EvidenceGraph/EvidenceGraphBuildRequest/EvidenceGraphRankingRequest，保持传统数据语义 |
| R-GCN | `models/graph_retriever/*`, method configs, model validation, trainers/tests | 以 `036d12d^` node-wise behavior 为参考，手工保留后续 batching/device/request-first 修复；删除 beam stack |
| GraphRAG | old `graph_rerank` package/tuning/config; historical `feature/fast-graph-rag` implementation | 删除旧 heuristic rerank；按当前架构移植 deterministic entity index、PPR、projection 和 independent trace |
| Semantic Registry | `registry/methods.py`, `registry/retrieval.py`, `registry/retrieval_builders.py`, experiment dispatch | 七个 ID；显式 request/family/artifact/capability；具体 payload；builder 预执行校验 |
| Execution provenance | none in current production code | 新增独立 graph contract、request、typed path method 和 builder/config；不新增 dataset adapter |
| Workflow | experiment planning/stage models、stages、scripts、Hydra configs | EvidenceGraph stage 只服务 R-GCN/training dependency；GraphRAG/provenance 不触发 |

## Reference snapshots

- Node-wise R-GCN behavior reference: direct parent `036d12d^`; do not revert whole commits because later batching、device propagation、request-first、Registry 和 checkpoint fixes must remain.
- Entity-search reference: `feature/fast-graph-rag` contains `retrieval/methods/fast_graphrag/{nlp,index,pagerank,scoring,method}.py` and associated tests; port algorithms only, not stale wiring or method IDs.

## Deletion verification scope

Final active code/config/script scan includes `graph_memory/`, `configs/`, `scripts/` and current behavioral tests. Historical plans and completed archived OpenSpec changes may mention retired concepts, but they must not be imported, configured or advertised as current capability. The incomplete active `openspec/changes/add-memory-stream-retrieval` directory is deleted rather than archived.
