# Memory Stream Implementation Plan

## Goal

实现可复现的 `memory_stream` Phase 2 baseline：读取已经完成的人工
importance 标注，经严格匹配和 task 内等级秩归一化后，与 dense relevance
和 position-derived pseudo-recency 组合排序。

## Data Preparation

Importance 不再由仓库中的 LLM runtime 生成。原始人工标注作为不可变 legacy
输入，由以下命令转换：

```powershell
uv run python scripts/data/clean_importance.py
```

默认输入：

```text
data/hotpotqa/processed/dev_memory_tasks.input.json
data/hotpotqa/processed/memory_stream/dev.first_1000.gpt-5.4-mini.importance.json
```

默认输出：

```text
data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json
data/hotpotqa/processed/memory_stream/dev.first_1000.importance.cleaning_summary.json
```

清洗程序验证 canonical dev 前 1000 个 task 的顺序、task id、content digest
和精确 node coverage。每个 task 内按唯一原始分数等级等距映射到整数
`1..10`；ties 保持不变，恒定 task 映射为 `5`。清洗报告记录输入输出
SHA-256、legacy 来源元数据、分布和异常统计。

## Artifact Contract

```json
{
  "schema_version": 1,
  "method": "memory_stream",
  "tasks": [
    {
      "task_id": "hotpot_x",
      "content_digest": "<sha256>",
      "scores": {
        "m0": 10,
        "m1": 1
      }
    }
  ]
}
```

Producer 对 selected canonical prefix 要求 task 数量、顺序、digest 和 node
coverage 完全一致。Consumer 允许 artifact 是 workflow task 的超集，但必须
按 task id 选择、拒绝重复记录，并重新验证 digest 和 node coverage。

## Retrieval

```python
relevance_raw = dense_score_by_node_id[item["id"]]
age_steps = max_position - item["position"]
recency_raw = recency_decay ** age_steps
importance_raw = float(task_importance["scores"][item["id"]])
```

每个 signal 在 task 内独立 min-max normalization；常量 signal 映射为
`0.0`。最终分数：

```python
score = (
    relevance_weight * relevance[node_id]
    + recency_weight * recency[node_id]
    + importance_weight * importance[node_id]
)
```

按 `(-score, node_id)` 排序。默认权重均为 `1.0`，
`recency_decay=0.99`。

## Remaining Tasks

- [x] 精简 importance artifact contract 和严格 validator。
- [x] 人工标注 legacy artifact 清洗、归一化和 summary。
- [x] 删除本地 LLM annotation、prompt、cache、runtime 和 config。
- [ ] 实现 Memory Stream ranking method。
- [ ] 注册 retrieval settings、builder 和 workflow。
- [ ] 将 normalized importance path 记录到 retrieval provenance。
- [ ] 完成 focused、integration 和 smoke verification。
