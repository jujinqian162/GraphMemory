# dense_rgcn_graph_retriever/ablations.json

对应配置文件：`configs/training/dense_rgcn_graph_retriever/ablations.json`

该文件只记录可发现的 R-GCN variant 和 experiment config 示例，不再保存完整训练超参。

完整训练配置唯一来源：

```text
configs/training/dense_rgcn_graph_retriever/base.json
```

runner 会先解析 `base.json` 的 `defaults + profiles.<profile>`，再叠加 `scripts/workflow/registry.py` 中注册的最小 variant override，并把最终配置写入：

```text
runs/<experiment>/ablations/dense_rgcn_graph_retriever/<variant>/effective_training_config.json
```

## Variant 列表

- `full_rgcn`：直接 alias 主实验，不重复训练。
- `wo_bridge`：训练和推理时隐藏 `bridge` 边。
- `wo_entity_overlap`：训练和推理时隐藏 `entity_overlap` 边。
- `wo_sequential`：训练和推理时隐藏 `sequential` 边。
- `wo_query_overlap`：训练和推理时隐藏 `query_overlap` 边。
- `wo_graph`：使用 identity graph encoder。
- `wo_edge_type`：共享 relation transform。
- `wo_edge_weight`：使用 uniform edge weight。
- `wo_seed_score`：移除 seed-score 特征。
- `wo_hard_negatives`：关闭 BM25、dense 和 graph-neighbor hard negatives。

`random_edges` 尚未实现，因为它会改变 graph construction，不属于当前 model/pair override 范围。

## 启用方式

在 experiment config 中开启：

```json
{
  "enable_ablation": true,
  "ablation_variants": {
    "dense_rgcn_graph_retriever": [
      "wo_bridge",
      "wo_entity_overlap",
      "wo_hard_negatives"
    ]
  }
}
```

不写 `ablation_variants` 时，runner 会展开已注册的完整 suite。
