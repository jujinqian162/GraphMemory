# dense_rgcn_graph_retriever.json

对应配置文件：`configs/methods/dense_rgcn_graph_retriever.json`

这是 R-GCN trainable graph retriever 的当前方法配置。它不是旧训练配置 schema，没有版本字段、默认值容器或旧字段别名。结构校验只接受当前字段；旧配置、旧 checkpoint 和旧 run artifact 需要删除后重跑。

## 使用位置

experiment config 通过 `method_configs` 引用本文件：

```json
"method_configs": {
  "dense_rgcn_graph_retriever": "configs/methods/dense_rgcn_graph_retriever.json"
}
```

workflow 初始化时会按 `default_profile` 或 experiment `--profile` 合并 `profiles.<name>`，然后写入：

```text
runs/<experiment>/learned/dense_rgcn_graph_retriever/effective_method_config.json
```

同时 workflow 会编译完整的 pair/train/retrieve/evaluate stage config。低层脚本只接受这些 stage config：

```text
python scripts/build_train_pairs.py --config runs/<experiment>/config/stages/pairs.dense_rgcn_graph_retriever.json
python scripts/train_method.py --config runs/<experiment>/config/stages/train.dense_rgcn_graph_retriever.json
python scripts/run_retrieval.py --config runs/<experiment>/config/stages/retrieve.dense_rgcn_graph_retriever.json
```

## 字段

- `method`: 固定为 `dense_rgcn_graph_retriever`。
- `default_profile`: 未指定 profile 时使用的配置覆盖名。
- `encoder`: 冻结 Sentence-Transformers encoder 的模型、query/passage 前缀和文本 batch size。该信息会写入 R-GCN checkpoint metadata，检索时从 checkpoint provenance 恢复。
- `pairs`: 训练 pair 采样配置，包括 easy random、BM25 hard、dense hard、graph-neighbor hard negative 的数量。
- `train.model`: R-GCN 模型结构，包含 `hidden_dim`、`num_layers`、`dropout` 和 `ablation`。
- `train.trainer`: 训练循环参数，包含 optimizer、learning rate、batch size、epoch、device、随机种子和 pos weight 开关。
- `train.selection`: best checkpoint 选择语义。当前 R-GCN 训练仍使用 dev composite 指标。
- `train.reporting`: 训练报告开关。
- `profiles`: 对上述字段的深度覆盖。profile 名必须与 experiment `--profile` 对齐。

## 产物

训练输出是 checkpoint 文件：

```text
runs/<experiment>/learned/dense_rgcn_graph_retriever/checkpoints/best.pt
```

checkpoint 当前契约不含版本字段。检索 summary 中的 checkpoint、device 和 encoder 信息来自实际 builder provenance，不由脚本层从 CLI 默认值推断。
