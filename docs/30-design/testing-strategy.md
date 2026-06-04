# Testing Strategy

Date: 2026-05-20

Status: Working reference.

## Goal

Tests should protect the scientific correctness of the evidence-tracing pipeline across supported phases. The aim is not high coverage for its own sake, but confidence that data conversion, graph construction, retrieval, reranking, trainable retrieval, tuning, and evaluation obey their contracts.

## Principles

- Test behavior at the smallest meaningful boundary.
- Prefer tiny deterministic fixtures over large data files.
- Keep core algorithm tests independent of CLI, filesystem, and network.
- Treat schema violations, leakage risks, split overlap, invalid graph references, and metric mismatches as test-worthy failures.
- Avoid tests that depend on downloading models or datasets at test time.
- Use integration smoke tests sparingly to prove the pipeline connects.

## Test Layers

| Layer | Test style | Purpose |
|---|---|---|
| Data contract tests | Unit tests with small dict fixtures | Validate required fields, forbidden fields, ID consistency, and fail-fast behavior. |
| Conversion tests | Unit tests with tiny raw HotpotQA examples | Ensure supporting facts map to `m*` node IDs and labels stay out of inputs. |
| Text/entity tests | Unit tests | Ensure stopwords, content tokens, lexical scoring, and heuristic entities are deterministic. |
| Graph tests | Unit tests with tiny task inputs | Ensure edge types, weights, limits, and no-label graph construction. |
| Retriever tests | Unit tests with tiny tasks | Ensure all retrievers output complete rankings in a shared shape. |
| Score-pipeline tests | Unit tests with tiny tasks and artificial scores | Ensure method construction, score normalization, graph requirements, and component composition work. |
| Reranker tests | Unit tests with artificial scores and graphs | Ensure graph propagation, score normalization, and graph helper compatibility work. |
| Train pair tests | Unit tests with tiny labels, rankings, and graphs | Ensure positive and negative training samples obey the artifact contract. |
| Tensorization tests | Unit tests with tiny graphs | Ensure message edges, relation IDs, edge weights, and ablation filters are deterministic. |
| Trainable model tests | Unit tests with fake embeddings and tiny tensors | Ensure model forward/backward, checkpoint metadata, and batch contracts work without real encoders. |
| Tuning tests | Unit tests with synthetic metric rows | Ensure objective and tie-breaks are deterministic. |
| Evaluation tests | Unit tests with tiny predictions/labels/graphs | Ensure metrics and connectivity use the correct artifacts. |
| CLI smoke tests | Small filesystem tests | Ensure scripts parse arguments and write expected artifacts. |
| Architecture dependency tests | AST import scans | Ensure domain packages keep approved dependency direction, root workflow integration ports stay one-way, and removed compatibility paths do not return. |
| End-to-end smoke test | Tiny synthetic pipeline | Ensure converter -> graph -> retrieval -> evaluation can run together. |

## Recommended Test Files

The Phase 1 plan already names the main test files. Keep that shape:

```text
tests/
  test_phase1_real_data_structures.py
  test_phase1_real_graphs.py
  test_phase1_real_retrieval.py
  test_phase1_real_evaluation.py
```

Possible additions if the implementation needs them:

```text
tests/test_phase1_real_validation.py
tests/test_phase1_real_reproducibility.py
tests/test_phase1_real_cli_smoke.py
tests/test_phase2_rgcn_pairs.py
tests/test_phase2_rgcn_tensorize.py
tests/test_phase2_rgcn_model.py
tests/test_phase2_rgcn_training.py
tests/test_phase2_rgcn_retrieval.py
tests/test_core_refactor_final_boundaries.py
```

Do not add many tiny test files too early. Start with the planned files and split only when a file becomes hard to navigate.

## Fixture Strategy

Use small hand-written fixtures:

- One HotpotQA-style raw example with two documents and two gold supporting facts.
- One task input with 4-6 memory items.
- One graph with `q`, several `m*` nodes, and each Phase 1 edge type.
- One prediction record with a complete ranking.
- One labels record with two gold nodes.

Fixture rules:

- Fixture records should be readable inline in tests.
- Large JSON fixture files are only allowed when testing CLI file I/O.
- Fixtures should use realistic field names from the data contract.
- Tests should include at least one negative fixture per major artifact type.

## What Must Be Tested Before Trusting Results

### Data Conversion

Must test:

- `task_id` is derived from the raw HotpotQA `_id`, not sampled position.
- `sentence_id` is local to document title.
- `position` is flattened across all memory items.
- `m{position}` node IDs are stable.
- supporting facts map from `(title, sentence_id)` to node IDs.
- input artifacts exclude gold fields.
- labels contain gold fields.

### Split Sampling

Must test:

- same seed gives same split.
- different offset gives disjoint slices when configured that way.
- negative count/offset raises.
- offset + count beyond available examples raises.

### Graph Construction

Must test:

- graph contains exactly one `q` node.
- graph contains all input memory nodes.
- graph edges only reference existing nodes.
- sequential edges connect adjacent sentences in the same source.
- query overlap edges originate from `q`.
- entity/bridge edges are created on controlled examples.
- graph output contains no gold fields.

### Retrieval

Must test:

- every method returns every memory node exactly once.
- ranking scores are finite.
- output schema is shared across methods.
- dense test skips clearly if the model is unavailable locally.
- graph methods require graph inputs and graph rerank config.
- flat methods do not require graph inputs.
- `application.run_retrieval` owns complete use-case orchestration.
- `retrieval.execution.service.run_retrieval` executes an already-built retrieval method and does not accept loose dense/checkpoint parameters.

### Score Pipeline

Must test:

- flat methods can run as one-component score pipelines.
- graph methods combine baseline and graph components with configured weights.
- graph candidate expansion gates graph component contributions.
- pipeline graph-method output matches the graph rerank helper on controlled artificial scores.
- retrieved graph-method subgraphs use induced top-k graph edges.

### Reranking

Must test:

- score normalization handles equal scores.
- bridge/entity neighbors can promote connected candidates.
- all original memory nodes remain in the final ranking.
- score components add up consistently when debug is enabled.
- `lambda_path` stays inert for HotpotQA Phase 1 unless explicitly implemented.

### Evaluation

Must test:

- Recall@k, Evidence F1@k, Full Support@k, and MRR.
- Connected Evidence Recall uses the shared graph, not method-emitted edges.
- Query-Evidence Connectivity includes `q + top_k` graph reasoning.
- prediction/label/graph task ID mismatch raises.
- missing gold node references raise.

### Trainable Retrieval

Must test:

- train pairs contain no `q` samples
- positives exactly match gold evidence nodes
- negatives never include gold evidence nodes
- hard negative sampling is deterministic for a fixed seed
- graph tensorization has stable relation vocab and message edge direction
- disabled edge types are filtered before message edge expansion
- `GraphBatch` and `TrainingBatch` shapes match contract expectations
- model forward returns one logit per supervised sample
- BCE loss can backpropagate through at least one trainable parameter
- checkpoint metadata validates before inference
- trainable retrieval emits the standard ranked result shape and does not read labels

## Network And Model Policy

Tests must not require network access.

Dense retrieval tests should choose one of these patterns:

- Use a tiny fake encoder object for most tests.
- Use a locally cached sentence-transformers model only when available.
- Skip with a clear reason when a real model is not cached.

Do not download models during ordinary test runs.

## Test Command Tiers

Recommended tiers:

```text
uv run pytest tests -q
```

Full local verification.

```text
uv run pytest tests/test_phase1_real_data_structures.py tests/test_phase1_real_graphs.py -q
```

Fast contract/conversion/graph checks.

```text
uv run pytest tests/test_phase1_real_retrieval.py -q
```

Retrieval and rerank checks. May skip real dense model tests if unavailable.

## Test Naming

Test names should state behavior:

```text
test_supporting_facts_map_title_sentence_to_node_ids
test_graph_builds_typed_edges_without_label_fields
test_bm25_and_dense_emit_same_ranked_schema
test_graph_rerank_uses_bridge_to_promote_connected_evidence
test_full_support_and_connected_evidence_use_top_k_nodes_on_shared_graph
```

Avoid vague names such as:

```text
test_prepare
test_graph
test_eval
```

## What Not To Test Heavily

- Exact dense model ranking quality on real data.
- Large-scale runtime performance.
- End-to-end paper-quality metrics.
- Unimplemented future baselines.
- Cached artifacts.

Those are experiment verification concerns, not unit test targets.

## Implementation Readiness Criteria

Before trusting an experiment run:

- All unit tests pass.
- CLI smoke tests write expected artifact files.
- Dense tests either pass with a cached model or skip with a clear reason.
- No retrieval or graph artifact contains label-only fields.
- Evaluation tests prove labels are read from label artifacts only.
- Trainable runs validate train pairs, tensor batches, and checkpoint metadata before training or inference.
- Retrieval-only trainable inference tests prove labels and train pairs are not read.
- Architecture dependency tests prove root compatibility paths stay removed and workflow integration ports stay narrow.
- Architecture dependency tests prove domain packages import owned implementation modules rather than retained root workflow integration ports.
