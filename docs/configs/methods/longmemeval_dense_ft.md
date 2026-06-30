# longmemeval_dense_ft.json

对应配置文件：`configs/methods/longmemeval_dense_ft.json`

这是 LongMemEval V1 retrieval workflow 使用的 Dense-FT 方法配置。它保持 public method id 为 `dense_ft`，但面向 LongMemEval 的长 conversation turn 文本降低 `full` 和 `cloud-full` 的 SentenceTransformers 训练 batch，避免把 `query + positive + negative` 三段长文本按过大的 mini-batch 送入反向传播。

## 与 dense_ft.json 的差异

- `method` 仍固定为 `dense_ft`，所以 workflow、manifest、artifact 目录和检索方法名不变。
- `quick` 和 `smoke` profile 与通用 Dense-FT 配置保持同等语义。
- `full` 和 `cloud-full` 使用 `train_batch_size: 8`、`eval_batch_size: 32`、`epochs: 2`。
- LongMemEval active experiment config 通过 `method_configs.dense_ft` 引用本文件；HotpotQA 和 2Wiki 继续使用通用 `configs/methods/dense_ft.json`。

## 使用位置

```json
"method_configs": {
  "dense_ft": "configs/methods/longmemeval_dense_ft.json"
}
```
