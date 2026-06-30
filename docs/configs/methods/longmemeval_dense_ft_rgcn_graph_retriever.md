# longmemeval_dense_ft_rgcn_graph_retriever.json

对应配置文件：`configs/methods/longmemeval_dense_ft_rgcn_graph_retriever.json`

这是 LongMemEval V1 retrieval workflow 使用的 Dense-FT-seeded R-GCN trainable graph retriever 方法配置。它保持 public method id 为 `dense_ft_rgcn_graph_retriever`，但面向 LongMemEval 的长候选列表降低 `full` 和 `cloud-full` 的图训练 batch，并把训练 epoch 提高到 15。

## 与 dense_ft_rgcn_graph_retriever.json 的差异

- `method` 仍固定为 `dense_ft_rgcn_graph_retriever`，所以 workflow、manifest、artifact 目录和检索方法名不变。
- 训练时仍由 workflow 自动依赖 `dense_ft` 的 best model directory，并把该目录作为 R-GCN encoder checkpoint。
- `full` 和 `cloud-full` 使用 `train.trainer.batch_size: 8`、`epochs: 15`。
- `cloud-quick` 使用 `train.trainer.batch_size: 8`，避免云端 quick run 意外使用更大的图 batch。
- LongMemEval active experiment config 通过 `method_configs.dense_ft_rgcn_graph_retriever` 引用本文件；HotpotQA 和 2Wiki 继续使用通用 `configs/methods/dense_ft_rgcn_graph_retriever.json`。

## 使用位置

```json
"method_configs": {
  "dense_ft_rgcn_graph_retriever": "configs/methods/longmemeval_dense_ft_rgcn_graph_retriever.json"
}
```
