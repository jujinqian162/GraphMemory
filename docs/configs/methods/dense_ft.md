# dense_ft.json

对应配置文件：`configs/methods/dense_ft.json`

这是 Dense-FT trainable retriever 的当前方法配置。它和 R-GCN 共用 `method_configs` 入口，没有版本字段、默认值容器或旧训练配置兼容读取。旧 Dense-FT 模型目录如果仍带版本字段 metadata，应删除并重训。

## 使用位置

experiment config 通过 `method_configs` 引用本文件：

```json
"method_configs": {
  "dense_ft": "configs/methods/dense_ft.json"
}
```

workflow 会写入 run-local effective method config，并生成 pairs/train/retrieve/evaluate stage config。低层训练入口是：

```text
python scripts/train_method.py --config runs/<experiment>/config/stages/train.dense_ft.json
```

## 字段

- `method`: 固定为 `dense_ft`。
- `default_profile`: 未指定 profile 时使用的配置覆盖名。
- `encoder`: SentenceTransformer 基础模型、本地模型目录、query/passage 前缀和编码 batch size。训练完成后这些值写入模型目录 metadata，检索时从 metadata 恢复。
- `pairs`: 生成 `train_pairs.json` 的负采样配置。当前默认关闭 hard-dense，避免 pair 阶段额外加载 dense encoder。
- `train.data.hard_negatives_per_positive`: 每个正例最多送入 Dense-FT 训练的负例数，优先顺序为 hard-dense、hard-bm25、graph-neighbor、easy-random。
- `train.trainer`: SentenceTransformers 2.7 `fit()` 参数，包括训练/评估 batch size、epoch、learning rate、warmup、梯度裁剪、device 和 AMP。
- `train.selection`: 选择最佳模型的 evaluator 指标，当前默认 `eval_dev_cos_sim_map@100`。
- `profiles`: 对上述字段的深度覆盖。`smoke` 使用 CPU，其余 profile 默认继承 CUDA device。

## 产物

Dense-FT 的 checkpoint 角色表示模型目录，不是 `.pt` 文件：

```text
runs/<experiment>/learned/dense_ft/checkpoints/best_model/
  dense_ft_model_config.json
  modules.json
  ...
```

`dense_ft_model_config.json` 是当前 strict metadata 契约，不含版本字段。检索 summary 中的 model dir、device 和 encoder 信息来自实际 builder provenance。
