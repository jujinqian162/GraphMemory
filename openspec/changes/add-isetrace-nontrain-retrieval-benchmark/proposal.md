## Why

ISETrace now has revision-pinned canonical trajectories, query-independent provenance graphs, deterministic motifs, and a 100-query natural-language pilot, but it remains outside the experiment workflow. The repository therefore cannot compare flat retrieval, content-derived graph retrieval, and training-free execution-provenance retrieval on the same native Agent trajectory candidates.

This increment should add the non-training benchmark path before any provenance retriever training. It must preserve the M1/M2 leakage boundary: graph topology is built from trajectories only, while queries and labels remain separate evaluation artifacts.

## What Changes

- Add a test-only ISETrace benchmark dataset adapter that joins query-authoring JSONL with a revision-pinned trajectory split and ranks every `execution.tool_output` in the referenced trajectory.
- Add explicit pilot review and label policies. Unreviewed examples may be used only when configured; directional queries can evaluate answer events while complete-chain/contributing-source queries evaluate complete support.
- Integrate ISETrace with the existing content-addressed prepare, graph, ranking, evaluation, Prefect, Hydra, and tracking workflow without invoking any training stage.
- Reuse the existing BM25 and frozen Dense implementations unchanged.
- Extend the maintained non-LLM GraphRAG baseline with a title-free shared-entity fallback for trajectory outputs, while forbidding access to native provenance edges.
- Add a deterministic frozen-Dense-seeded `provenance_path` method that derives query-independent logical output dependencies from `data.feeds` and artifact write/read lifecycles, performs bounded bidirectional schema-gated completion, and emits auditable logical dependency edges.
- Keep the current 100-query file pilot-only until manual review is complete. No paper result is claimed by this change.

## Capabilities

### New Capabilities

- `isetrace-nontrain-benchmark`: Natural or template provenance queries can be evaluated over native ISETrace ToolOutput candidate sets with content-addressed query and trajectory inputs.
- `training-free-provenance-path-retrieval`: Frozen Dense seeds can be expanded through query-independent logical provenance dependencies without task-specific training or label access.

### Modified Capabilities

- `entity-search-graphrag`: When candidates have no document titles, GraphRAG can construct bounded shared-entity groups from candidate text and still abstains to exact Dense output when no bridge is accepted.
- `retrieval-workflow-matrix`: The single-method workflow supports a test-only execution-provenance dataset and non-training methods without requiring train/dev inputs.
- `method-native-retrieval-traces`: The closed trace union includes deterministic provenance-path proposals, interventions, and exact fallback.

## Impact

- Dataset/runtime code under `graph_memory/datasets/isetrace/`, `graph_memory/stages/`, `graph_memory/experiment/`, and `configs/dataset/`.
- Provenance dependency projection under `graph_memory/graphs/provenance/`.
- Retrieval request, method, registry, result, and trace contracts under `graph_memory/retrieval/` and `graph_memory/registry/`.
- Focused adapter, GraphRAG, provenance-path, workflow, and evaluation tests plus maintained operations/contracts documentation.
- No Dense fine-tuning, graph-neural training, checkpoint migration, LLM call, query generation, or paper metric update.
