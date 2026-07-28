# Validation migration matrix

This matrix records the retired validation sources, their Pydantic owners, consumption boundary, and focused regression coverage. Private `_required_*`, record/list/map readers, copied field sets, and hand serializers map to the same owner as their public caller and are not retained as runtime APIs.

| Retired source / functions | Authoritative owner | Boundary / consumer | Focused coverage |
|---|---|---|---|
| `validation/common.py`: `validate_no_label_fields`, `_walk_forbidden_fields` | `contracts/model.py::reject_label_fields`, called only by owning ranking/graph models | dataset ranking records, graph models | dataset structure, provenance dataset, Pydantic domain tests |
| `validation/common.py`: task alignment, record readers, strict scalars, uniqueness, string sequences | closed `DomainModel` fields plus dataset/result/graph/pair/evaluation aggregates | every project-owned artifact parse | Pydantic domain, dataset, retrieval, graph, pair, evaluation tests |
| `validation/graphs.py`: `validate_graphs`, expected-item helpers, node/edge validators | `graphs/contracts.py`: node union, `GraphEdge`, `EvidenceGraph`, `EvidenceGraphBatch`; graph ranking request aggregate | graph publication/consumption and tensorization | phase-1 structures, R-GCN tensorization/batching, architecture tests |
| `validation/ranking.py`: `validate_ranked_results`, candidate helpers, subgraph/metadata checks | `retrieval/results.py`: `RankedResult`, `RetrievedSubgraph`, `RankedResultEnvelope`, `RankedResultBatch` | per-task retrieval execution then final batch | retrieval hardening, phase-1/phase-2 retrieval, invalid-first-result regression |
| `validation/ranking.py`: entity/GraphRAG/local/provenance/query-conditioned trace branches and `_native_trace_record` | discriminated trace/member models in `retrieval/contracts.py`; graph values in `graphs/provenance/contracts.py` | method result construction and ranked-result envelope | execution-provenance domain and retrieval hardening tests |
| `validation/tasks.py`: HotpotQA ranking/label validators | `datasets/hotpotqa/records.py`: ranking, label, combined, prepared split | parser/converter/prepare/projectors | phase-1 data/evaluation/dense-ft tests |
| `validation/tasks.py`: 2Wiki ranking/label validators | `datasets/twowiki/records.py` | parser/converter/prepare/projectors | cross-dataset workflow tests |
| `validation/tasks.py`: MuSiQue ranking/label validators | `datasets/musique/records.py` | parser/converter/prepare/projectors | cross-dataset workflow tests |
| `validation/twowiki_provenance.py`: all ranking, label, candidate, graph, construction identity, confidence, binding, and gold-path validators | `datasets/twowiki_provenance/records.py`, shared `ExecutionProvenanceGraph`, and `ProvenanceGraphConstructionConfig` | conversion, transform, prepare, retrieval projection | provenance dataset, parallel conversion, transform, provenance R-GCN tests |
| `validation/training_pairs.py`: pair/config/summary validators and candidate/label/graph helpers | `training_pairs/config.py`, `training_pairs/contracts.py::TrainPairRecord/TrainPairBuildSummary/TrainPairDataset` | pair builders, artifact stage, all trainers' preflight payloads | phase-2 pair, dense-ft, provenance R-GCN, fail-fast payload tests |
| `validation/model.py`: graph-retriever feature/model/training/checkpoint validators | `models/graph_retriever/config/records.py` and checkpoint envelope | config construction and checkpoint load/save | R-GCN model, retrieval, batching and checkpoint tests |
| provenance checkpoint `_validate_payload` | provenance config and checkpoint envelope models | checkpoint load/save before model state loading | provenance R-GCN checkpoint tests |
| `validation/metrics.py`: generic/evidence metric row validators | `evaluation/contracts.py::MetricRow/PerTaskMetricRow/FailureCase`, aliases, and suite row adapter | evaluation before selection/publication; typed stage result | phase-1 evaluation, per-task, output/delivery and tracking tests |
| Dense-FT dataclass/TypeAdapter validation | dense-ft Pydantic data/run/selection/metadata models | trainer preflight and metadata publication | dense-ft data and workflow tests |

## Historical drift regression

Commits `3942114` and `1b5b98d` changed active retrieval-method behavior without one authoritative result schema. `RetrievalMethodId` now lives in `retrieval/methods/ids.py`, and `RankedResult.method` references it directly. `tests/test_pydantic_domain_contracts.py` iterates every enum member, so adding a method cannot require editing a copied validation allowlist.

## Runtime assertions deliberately retained

Tensor shape/device/index assertions in graph batching, tensorization, and framework state loading remain algorithm/framework-owned. They do not duplicate artifact field declarations or JSON validation and therefore stay outside Pydantic.
