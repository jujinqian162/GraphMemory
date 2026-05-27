# Contract Documentation

Status: Maintained project-level navigation.

This directory contains the canonical contracts for artifacts, retrieval behavior, model inputs, configs, and validation boundaries. Phase plans may link here, but should not duplicate field definitions.

## Canonical Documents

| Document | Owns | Does not own |
|---|---|---|
| `data-contracts.md` | Disk artifacts, JSON/CSV schemas, artifact producers/consumers, ID rules, validation boundaries. | Tensor shapes, model internals, training loop mechanics. |
| `retrieval-contracts.md` | Public retrieval methods, registry metadata, seed retrieval signals, ranking behavior. | Metric formulas, checkpoint contents, file IO. |
| `model-contracts.md` | Trainable model configs, `GraphBatch`, `TrainingBatch`, relation vocab, checkpoints, training evaluation contract. | Raw dataset conversion, CLI command examples. |

Historical phase-specific contract files remain available for provenance:

- `phase1-data-contracts.md`
- `phase2-trainable-retriever-contracts.md`

When a rule becomes stable, maintain it in the canonical project-level document above. Phase-specific files should only preserve historical context or link to the maintained source of truth.

## Bilingual Type Documentation Rule

Every concrete project type that defines a stable data contract must include a complete bilingual Python triple-quoted docstring.

This applies to:

- `TypedDict` artifact records.
- frozen dataclass configs and internal records.
- concrete model, tensorizer, feature builder, sampler, registry, and retriever classes.
- public `Protocol` interfaces that define a replaceable behavior boundary.

Each docstring must explain:

- the type purpose in English and Chinese.
- every field or public method in English and Chinese.
- tensor index semantics when relevant.
- whether the type belongs to disk artifact, library-core state, or model input.

Required style:

```python
class TrainPairRecord(TypedDict):
    """
    One training pair artifact row for a query-node supervision example.
    一个 query-node 监督样本对应的训练 pair artifact 行。

    Fields / 字段:
    - task_id: Task join key matching memory task, label, and graph artifacts.
      task_id：任务 join key，必须匹配 memory task、label 和 graph artifact。
    - node_id: Memory node id being supervised; must not be the question node `q`.
      node_id：被监督的 memory node id；不能是问题节点 `q`。
    - label: Binary evidence label, where 1 means gold evidence and 0 means sampled negative.
      label：二分类 evidence 标签，1 表示 gold evidence，0 表示采样负例。
    - sample_type: Sampling source used to create this row.
      sample_type：生成该样本行时使用的采样来源。
    """

    task_id: TaskId
    node_id: NodeId
    label: Literal[0, 1]
    sample_type: TrainPairSampleType
```

Inline comments may clarify implementation details, but they are not a substitute for complete field documentation.

## Maintenance Rules

- Keep one canonical home for each contract. Other docs should link instead of copying schema details.
- If a phase plan needs a new field, first update the canonical contract, then link the plan to the changed section.
- Unknown fields should fail validation unless the owning contract explicitly allows `metadata` or `debug`.
- Validators must fail fast and must not repair, sort, drop, or infer data.
- Scripts own file IO. Library-core modules receive parsed records, config objects, or tensors.
