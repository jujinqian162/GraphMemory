# Dense-FT 训练配置

配置文件：`configs/training/dense_ft/base.json`

## 配置语义

`defaults` 是所有 profile 的基线；`profiles.<name>` 只写相对 `defaults` 的覆盖值。默认设备已经是 `trainer.device: "cuda"`，因此 `quick`、`full`、`cloud-quick` 和 `cloud-full` 不重复声明 CUDA。`smoke` 使用 CPU，才显式覆盖为 `"cpu"`。

## 字段

- `encoder.model`：SentenceTransformer 基础模型或本地模型目录。
- `encoder.query_prefix` / `passage_prefix`：训练和检索共用的文本前缀。
- `encoder.batch_size`：检索编码批大小，并写入模型目录中的 `dense_ft_model_config.json`。
- `pair_sampling`：生成 `train_pairs.json` 时使用的负样本配置；当前默认关闭 hard-dense，避免 pair 阶段额外加载 dense encoder。
- `data.hard_negatives_per_positive`：每个正例最多选取的负例数，优先级固定为 hard-dense、hard-bm25、graph-neighbor、easy-random。
- `trainer.*`：SentenceTransformers 训练参数，包括 batch size、epoch、学习率、warmup、梯度裁剪、精度模式和 checkpoint 保留数。
- `selection.best_metric`：最终记录和模型选择使用的 dev 指标。

## 产物

workflow 的 checkpoint 角色对 dense-ft 表示 SentenceTransformer 模型目录：

```text
learned/dense_ft/checkpoints/best_model/
  dense_ft_model_config.json
  modules.json
  ...
```

它不是 R-GCN 使用的 `.pt` checkpoint。检索命令只需传入该目录，模型路径、前缀和编码批大小由 metadata 恢复。
