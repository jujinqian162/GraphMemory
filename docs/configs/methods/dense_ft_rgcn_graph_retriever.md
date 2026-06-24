# dense_ft_rgcn_graph_retriever.json

对应配置文件：`configs/methods/dense_ft_rgcn_graph_retriever.json`

这是 Dense-FT-seeded R-GCN trainable graph retriever 的当前方法配置。它复用 R-GCN 的 pair、model、trainer、selection 和 reporting 字段，但训练时由 workflow 自动依赖 `dense_ft` 的 best model directory，并把该目录作为 R-GCN encoder checkpoint。它不是旧训练配置 schema，没有版本字段、默认值容器或旧字段别名。

## 使用位置

experiment config 通过 `method_configs` 引用本文件：

```json
"method_configs": {
  "dense_ft_rgcn_graph_retriever": "configs/methods/dense_ft_rgcn_graph_retriever.json"
}
```

workflow 初始化时会按 `default_profile` 或 experiment `--profile` 合并 `profiles.<name>`，然后写入：

```text
runs/<experiment>/learned/dense_ft_rgcn_graph_retriever/effective_method_config.json
```

选择该方法时，workflow 会把 `dense_ft` 作为训练依赖一起编译 pair/train stage，但不会把依赖方法加入用户选择的 retrieve/evaluate 输出。低层 stage config 入口是：

```text
python scripts/build_train_pairs.py --config runs/<experiment>/config/stages/pairs.dense_ft.json
python scripts/train_method.py --config runs/<experiment>/config/stages/train.dense_ft.json
python scripts/build_train_pairs.py --config runs/<experiment>/config/stages/pairs.dense_ft_rgcn_graph_retriever.json
python scripts/train_method.py --config runs/<experiment>/config/stages/train.dense_ft_rgcn_graph_retriever.json
python scripts/run_retrieval.py --config runs/<experiment>/config/stages/retrieve.dense_ft_rgcn_graph_retriever.json
```

## 字段

- `method`: 固定为 `dense_ft_rgcn_graph_retriever`。
- `default_profile`: 未指定 profile 时使用的配置覆盖名。
- `encoder`: Dense-FT seed 不存在时的回退 encoder 配置；正常 workflow 训练时，R-GCN 会从 `dense_ft` 的 metadata 恢复 model path、query/passage prefix 和 batch size。
- `pairs`: 训练 pair 采样配置，包括 easy random、BM25 hard、dense hard、graph-neighbor hard negative 的数量。
- `train.model`: R-GCN 模型结构，包含 `hidden_dim`、`num_layers`、`dropout` 和 `ablation`。
- `train.trainer`: 训练循环参数，包含 optimizer、learning rate、batch size、epoch、device、随机种子和 pos weight 开关。
- `train.selection`: best checkpoint 选择语义。当前 R-GCN 训练仍使用 dev composite 指标。
- `train.reporting`: 训练报告开关。
- `profiles`: 对上述字段的深度覆盖。profile 名必须与 experiment `--profile` 对齐。

## 产物

Dense-FT seed 产物是依赖方法的模型目录：

```text
runs/<experiment>/learned/dense_ft/checkpoints/best_model/
```

本方法训练输出是独立 R-GCN checkpoint 文件：

```text
runs/<experiment>/learned/dense_ft_rgcn_graph_retriever/checkpoints/best.pt
```

checkpoint metadata 的 method 必须是 `dense_ft_rgcn_graph_retriever`。检索 summary 中的 checkpoint、device 和 encoder 信息来自实际 builder provenance，不由脚本层从 CLI 默认值推断。