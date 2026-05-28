# Validation Strategy

Date: 2026-05-20

Status: Working reference.

## Goal

Validation protects experiment correctness. Invalid artifacts, inconsistent joins, hidden label leakage, impossible scores, and malformed configs should stop the run before they produce misleading metrics.

## Principles

- Validate at script boundaries.
- Fail fast with clear error messages.
- Do not repair artifacts silently.
- Do not infer missing fields.
- Keep validators deterministic and side-effect free.
- Treat validation as part of the scientific contract, not as user-experience polish.

## Boundary Rule

Validation should run:

```text
read artifact
  -> validate input contract
  -> run domain logic
  -> validate output contract
  -> write artifact
```

Evaluation and tuning should also validate joins before computing anything:

```text
predictions + labels + graphs
  -> exact task_id join validation
  -> metric computation
```

## Recommended Validators

| Validator | Purpose |
|---|---|
| `validate_memory_task_inputs(records)` | Input-visible task records are complete and label-free. |
| `validate_memory_task_labels(records, inputs_by_task_id)` | Gold labels match known task and node IDs. |
| `validate_graphs(graphs, inputs_by_task_id)` | Graph nodes/edges are complete, endpoint-safe, and label-free. |
| `validate_ranked_results(predictions, inputs_by_task_id)` | Rankings are complete, finite, duplicate-free, and method-consistent. |
| `validate_train_pairs(records, inputs_by_task_id, labels_by_task_id, graphs_by_task_id)` | Train pairs match task, label, and graph artifacts without invalid negatives. |
| `validate_train_pair_build_summary(summary)` | Negative sampling summary matches the documented artifact shape. |
| `validate_graph_rerank_config(config)` | Rerank parameters are finite and valid. |
| `validate_negative_sampling_config(config)` | Pair-builder sampling counts and hard-pool settings are valid. |
| `validate_trainable_model_config(config)` | Trainable model dimensions and semantic config fields are present and valid. |
| `validate_trainable_training_config(config)` | Train loop config fields are finite and reproducible. |
| `validate_trainable_checkpoint_metadata(checkpoint)` | Checkpoint metadata can reconstruct the model and matches the requested method. |
| `validate_metric_rows(rows)` | Metric rows have expected columns and valid value ranges. |
| `validate_task_id_alignment(...)` | Artifacts match exactly where exact joins are required. |
| `validate_no_label_fields(records)` | Input-visible artifacts contain no forbidden gold fields. |

Validator return style:

```text
validate_xxx(...) -> None
```

Validators raise exceptions. They do not return cleaned data.

Public validators accept `object` at the boundary and perform their own runtime narrowing. Call sites should pass
loaded JSON artifacts or domain-typed artifacts directly; they should not cast just to satisfy static type checkers.
Any zero-copy cast needed for `TypedDict` or invariant container handling belongs inside `validation.py`, after the
validator has checked that the value has the expected list/map/object shape.

## Error Message Style

Validation errors should include:

- artifact or validator name
- field name or invariant
- `task_id` when available
- offending node ID or method when available
- expected condition
- observed value or concise reason

Good:

```text
Invalid graph: task_id=hotpot_000123 edge target=m52 does not exist in nodes.
```

Good:

```text
Invalid ranked results: task_id=hotpot_000007 method=bm25 ranked_nodes contains duplicate node_id=m3.
```

Avoid:

```text
bad graph
invalid data
key error
```

## Exception Types

Phase 1 can start with standard exceptions:

- `ValueError` for invalid content or invariant violations.
- `KeyError` only when a missing key genuinely escapes validation, but validators should usually raise `ValueError` with context.
- `FileNotFoundError` for missing required paths.

```python
class ContractValidationError(ValueError):
    pass
```

Recommendation:

- Use one project-level `ContractValidationError(ValueError)` for artifact contract violations.
- Do not create many custom exception classes in Phase 1.
- Use standard exceptions for non-contract problems such as missing files.

## Unknown Fields

Default rule:

- Unknown top-level fields should raise unless they are inside an explicit `metadata` or `debug` object.

Reason:

- Silent unknown fields can hide accidental leakage or typoed output names.

Allowed extension containers:

- `metadata`
- `debug`

Rule:

- Retrieval and graph construction should still reject label-only fields even inside extension containers if they reveal gold evidence.

## Label Leakage Validation

Forbidden fields in input-visible artifacts:

- `gold_answer`
- `gold_evidence_nodes`
- `gold_dependency_edges`
- `supporting_facts`
- `is_gold`
- `is_gold_evidence`
- `is_gold_edge`

Must check:

- `*_memory_tasks.input.json`
- `*_graphs.json`
- retrieval debug artifacts generated before evaluation

Evaluation artifacts may contain gold labels because evaluation is label-aware.

## Artifact-Specific Invariants

### Memory Task Inputs

Validate:

- unique `task_id`
- non-empty `memory_items`
- unique memory item IDs within task
- `id == m{position}` for HotpotQA Phase 1
- positions are contiguous from `0`
- required fields exist
- forbidden fields are absent

### Labels

Validate:

- labels match known task IDs
- each label record has at least one gold evidence node for HotpotQA Phase 1
- every gold node exists in that task input
- no duplicate gold nodes
- `gold_dependency_edges` is empty for HotpotQA Phase 1 unless a dependency-labeled dataset is explicitly used

### Graphs

Validate:

- graph task IDs match task inputs
- exactly one `q` node
- all task memory nodes exist in graph nodes
- edge endpoints exist
- edge types are allowed
- edge weights are finite and non-negative
- graph contains no label-only fields

### Ranked Results

Validate:

- predictions match known task IDs
- method name is supported
- `ranked_nodes` includes every memory node exactly once
- no duplicate node IDs
- scores are finite
- ranking is sorted descending by score
- retrieved subgraph nodes are in top-k or documented as query node `q`
- retrieved subgraph edges reference known graph nodes

### Metrics

Validate:

- required columns exist
- numeric metrics are in `[0.0, 1.0]`
- latency is non-negative
- HotpotQA-only `Path Recall@10` and `Edge Recall@10` are `N/A`

### Train Pairs

Validate:

- pair task IDs exist in input, label, and graph artifacts
- pair node IDs are memory nodes, never `q`
- positive rows exactly come from gold evidence nodes
- negative rows never include gold evidence nodes
- `sample_type="positive"` only appears with `label=1`
- negative sample types only appear with `label=0`
- duplicate `(task_id, node_id, sample_type)` rows are invalid

### Trainable Configs And Checkpoints

Validate:

- feature names and relation vocab order are explicit
- config values needed for tensor dimensions are present
- checkpoint `method_name` matches the requested retrieval method
- checkpoint metadata contains model and training config
- loading fails before model construction when metadata is incomplete

## What Validation Should Not Do

Validators should not:

- fill default config values
- convert combined artifacts into input/label artifacts
- drop unknown fields
- remove duplicate nodes
- sort rankings
- coerce invalid scores
- guess missing labels
- fallback to another file

If transformation is needed, implement it as a named conversion step and test it separately.

## Testing Validation

Every validator should have:

- one valid fixture test
- at least one invalid fixture test
- an assertion that the error message includes the relevant `task_id` when available

Critical negative tests:

- label field appears in input artifact
- graph edge references missing node
- duplicate ranked node ID
- prediction task ID missing from labels
- split request exceeds available examples
- NaN or infinite score appears

## Extension Decisions

- Unknown fields in core artifact sections should fail.
- `metadata` and `debug` may contain extension fields, but input-visible artifacts still must not contain label-only information anywhere.
