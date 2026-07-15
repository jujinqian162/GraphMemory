## Why

TRAJECT-Bench exposes a catalog and a one-dimensional gold call sequence rather than a native execution graph, so projecting catalog links into execution-provenance edges creates misleading structure and gives graph retrieval little valid signal. A separately named, leakage-audited benchmark derived once from labeled 2Wiki evidence chains can provide a controlled branching execution graph in which typed traversal and a provenance-native R-GCN have a real, measurable role.

## What Changes

- Add a deterministic one-time converter that produces `twowiki_provenance` raw train/dev/test artifacts, manifests, and graph statistics from labeled 2Wiki source data.
- Add a dataset-owned `twowiki_provenance` adapter that parses the generated raw schema and projects it to text ranking, GraphRAG, execution-provenance ranking, and evidence-label requests.
- Construct a typed Task/Agent/ToolCall/ToolOutput graph whose gold evidence chain is present as a valid `feeds`/`returns` path and whose remaining branches are deterministic BM25, dense, or hybrid semantic successor matches using the same public node/edge schema without answer, gold-flag, ordering, or topology leakage.
- **BREAKING**: remove the TRAJECT-Bench adapter, dataset configuration, preparation path, validation, tests, documentation, and local raw artifacts because the source cannot exercise execution-provenance graph retrieval.
- Replace exhaustive post-hoc provenance path enumeration with directed typed beam search, configurable semantic seeds, incremental pruning, binding-aware path scoring, honored `top_k`, and logical candidate-edge traces for path evaluation.
- Add `execution_provenance_rgcn_retriever`, a trainable provenance-native R-GCN that encodes all typed graph nodes and binding schemas, consumes materialized BM25/dense hard-negative pairs, and independently scores ToolOutput candidates plus contracted output-to-output dependency transitions without a beam decoder.
- Reuse the existing prepare/pairs/train/retrieve/evaluate/aggregate lifecycle and stage types; add only dataset, method, configuration, Registry, and planner wiring needed to route the new dataset and method.
- Preserve the existing evidence-only Dense and Dense-FT R-GCN methods and their node-wise checkpoint contract.

## Capabilities

### New Capabilities
- `twowiki-provenance-dataset`: Deterministic 2Wiki-derived raw conversion, typed branching graph schema, leakage validation, request projection, and workflow-ready dataset artifacts.
- `execution-provenance-rgcn-retrieval`: Provenance-native graph tensorization, relation-aware R-GCN encoding, binding-schema-aware relations, pair-supervised ToolOutput candidate scoring, logical-edge scoring, checkpointing, and typed/logical edge traces.

### Modified Capabilities
- `execution-provenance-retrieval`: Require directed incremental beam traversal, configurable seed breadth, binding-aware scoring, exact `top_k`, and native plus contracted logical path traces.
- `semantic-method-registry`: Add the provenance-native R-GCN method, payload, trainable capabilities, and execution-provenance family compatibility while keeping evidence R-GCN inputs separate.
- `retrieval-workflow-matrix`: Extend the execution-provenance method matrix and route the new trainable method through existing lifecycle stages without adding a workflow stage type.

## Impact

The change affects `graph_memory/datasets`, provenance graph/request contracts and validators, stateless provenance search, a new provenance-R-GCN model package, Registry definitions/builders, train-pair and train/retrieve stage dispatch, experiment configuration models, dataset/method compatibility, scripts, tests, and operations documentation. TRAJECT-Bench is no longer a supported dataset. Generated 2Wiki-derived raw data and model checkpoints are new schemas with no compatibility promise; existing 2Wiki, HotpotQA, MuSiQue, GraphRAG, Dense, and evidence-R-GCN artifacts remain unchanged.
