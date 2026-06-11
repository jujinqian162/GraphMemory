# dense_rgcn_graph_retriever/base.json

对应配置文件：`configs/training/dense_rgcn_graph_retriever/base.json`

这个文件是 `dense_rgcn_graph_retriever` 的默认训练配置。它取代低层训练脚本上的大量 CLI 超参，是正常 experiment runner 路径下 R-GCN 训练行为的主要控制入口。

它控制三件事：

1. `pairs` stage 如何采样训练 pair。
2. `train` stage 如何构建 encoder、R-GCN 模型和训练循环。
3. `retrieve` stage 对 trainable retriever 使用什么 device 做推理。

## 使用位置

experiment config 通过下面字段引用本文件：

```json
"training_configs": {
  "dense_rgcn_graph_retriever": "configs/training/dense_rgcn_graph_retriever/base.json"
}
```

experiment runner 会把本文件解析为 run-local effective training config：

```text
runs/<experiment_name>/learned/dense_rgcn_graph_retriever/effective_training_config.json
```

之后 `pairs` 和 `train` stage 使用的是这个 run-local resolved config，而不是源文件本身。

## Profile 合并规则

本文件采用：

```text
defaults + profiles.<selected_profile>
```

的深度合并规则。

例如 experiment runner 使用 `--profile quick` 时，会读取：

```text
defaults
profiles.quick
```

并让 `profiles.quick` 覆盖 `defaults` 中相同路径的字段。

如果不显式传 profile，则使用：

```json
"default_profile": "quick"
```

## 顶层字段

### `schema_version`

配置 schema 版本。

当前值：

```json
1
```

用途：

- 标识当前配置文件结构版本。
- 为后续 schema 升级预留。

当前代码只要求它存在或默认补为 `1`，还没有复杂版本迁移逻辑。

### `method`

这个 training config 对应的 trainable method。

当前唯一有效值：

```json
"dense_rgcn_graph_retriever"
```

experiment runner 会校验 training config 中的 `method` 必须和 selected method 匹配。

### `default_profile`

没有显式选择 training profile 时使用的 profile。

当前值：

```json
"quick"
```

正常 experiment runner 中，experiment `--profile` 会映射到 training config 同名 profile。

### `defaults`

默认训练参数。所有 profile 都会先继承这里的值。

不要把 `defaults` 理解为“只给 quick 用”。如果某个 profile 没有覆盖字段，就会继续使用 `defaults` 的值。

### `profiles`

profile 覆盖项。

当前 profile：

- `smoke`：最小流程测试。
- `quick`：快速训练。
- `full`：较大训练规模。
- `cloud-quick`：服务器快速训练，使用更大的 batch size。
- `cloud-full`：服务器较完整训练，使用更大的 batch size 和 full profile 的模型规模。

每个 profile 可以只写它想覆盖的字段；没有写的字段会继承 `defaults`。

experiment runner 不做 profile 映射。`--profile cloud-quick` 会同时查找 experiment config 的 `profiles.cloud-quick` 和本 training config 的 `profiles.cloud-quick`。

## `encoder`

控制冻结 text encoder 的模型、文本前缀和 encoder 内部文本 mini-batch。

### `encoder.model`

Sentence-Transformers 模型名或本地模型路径。

当前值：

```json
"models/intfloat-e5-base-v2"
```

用途：

- 训练时生成 query 和 memory text embedding。
- checkpoint 的 `model_config.encoder_model` 会记录该值。
- 推理重建模型时也需要与 checkpoint 中的 encoder 维度匹配。

可填值：

- Hugging Face / Sentence-Transformers 模型 ID 或本地模型目录，例如 `models/intfloat-e5-base-v2`。
- 本地模型目录，例如 `models/intfloat-e5-base-v2`。

### `encoder.query_prefix`

编码 query 前拼接的文本前缀。

当前值：

```json
"query: "
```

E5 系列模型通常需要 query/passsage 前缀。如果更换 encoder，需要确认该模型是否仍需要这些前缀。

### `encoder.passage_prefix`

编码 memory text 前拼接的文本前缀。

当前值：

```json
"passage: "
```

### `encoder.batch_size`

每次传给 Sentence-Transformers 的文本 mini-batch 大小。当前默认值：

```json
64
```

该值控制 query/passage 编码时的 GPU mini-batch，不是每个训练 batch 包含的 task graph 数。跨任务 flatten 后，encoder 仍使用这里的值切分文本；增大它通常能提高 GPU 利用率，但也会增加显存占用，过大时可能 OOM。

## `model`

控制 R-GCN trainable model 结构。

### `model.hidden_dim`

R-GCN 和 scorer 使用的隐藏维度。

当前默认值：

```json
128
```

影响：

- 模型参数量。
- checkpoint 中的模型重建配置。
- 训练显存和速度。

常见设置：

- `32`：smoke/debug。
- `128`：quick 默认。
- `256`：full 默认。

### `model.num_layers`

R-GCN 层数。

当前默认值：

```json
2
```

可填值：

- 正整数：使用对应层数的 R-GCN。
- `0`：不做图消息传递，用于 `wo_graph` ablation。

### `model.dropout`

Dropout 概率。

当前默认值：

```json
0.1
```

约束：

- 应为 `0.0` 到 `1.0` 之间的数值。

### `model.ablation`

结构消融名称。

当前默认值：

```json
"full_rgcn"
```

当前支持值：

- `full_rgcn`：完整 R-GCN。
- `wo_graph`：不使用图消息传递；通常同时设置 `num_layers: 0`。
- `wo_edge_type`：不区分 edge type，使用 shared relation transform。
- `wo_bridge`：禁用 bridge edge。
- `wo_edge_weight`：不使用 artifact edge weight，改用 uniform edge weight。
- `wo_seed_score`：移除 seed score 相关 node/scorer 特征。

这个字段会写入 checkpoint，并影响推理时模型结构重建。

## `optimization`

控制训练循环。

### `optimization.optimizer`

优化器名称。

当前唯一有效值：

```json
"AdamW"
```

虽然字段是字符串，但当前训练实现实际只使用 `torch.optim.AdamW`。填其他值会导致验证失败或行为不受支持。

### `optimization.epochs`

训练 epoch 数。

当前默认值：

```json
5
```

profile 覆盖：

- `smoke`: `1`
- `quick`: `5`
- `full`: `10`

### `optimization.batch_size`

每个训练 batch 中包含多少个 task graph。

它同时决定一次 graph feature bulk 请求最多聚合多少个 task，但不覆盖 `encoder.batch_size`。例如：

```json
{
  "encoder": {"batch_size": 64},
  "optimization": {"batch_size": 8}
}
```

表示一次构建 8 个 task graph，并由 Sentence-Transformers 按每 64 条文本执行内部 mini-batch。

当前默认值：

```json
8
```

修改 quick profile 的 batch size：

```json
"profiles": {
  "quick": {
    "optimization": {
      "batch_size": 16
    }
  }
}
```

修改所有 profile 的默认 batch size：

```json
"defaults": {
  "optimization": {
    "batch_size": 16
  }
}
```

如果某个 profile 自己覆盖了 `batch_size`，它不会继承新的 default 值。

### `optimization.learning_rate`

AdamW learning rate。

当前默认值：

```json
0.0001
```

### `optimization.max_grad_norm`

梯度裁剪的最大 norm。

当前默认值：

```json
1.0
```

用途：

- 训练时调用 gradient clipping，避免梯度爆炸。

### `optimization.random_seed`

训练循环随机种子。

当前默认值：

```json
13
```

用途：

- 控制训练中可控随机性。
- 注意它不等同于 `pair_sampling.random_seed`；后者控制训练 pair 采样。

### `optimization.pos_weight`

是否启用 BCE positive class weighting。

当前默认值：

```json
true
```

含义：

- `true`：根据 train pairs 中负例/正例比例设置 `pos_weight`。
- `false`：不使用 positive weighting。

如果训练 pair 中没有正例，启用 `pos_weight` 会 fail-fast。

### `optimization.device`

训练和 trainable retrieval 推理使用的 torch device 字符串。

当前默认值：

```json
"cuda"
```

常见值：

- `cuda`
- `cuda:0`
- `cpu`

如果写 `cuda` 但当前环境没有可用 CUDA，训练会失败。这个字段不会自动 fallback 到 CPU。

## `pair_sampling`

控制 `pairs` stage 的负采样。`scripts/build_train_pairs.py --config ...` 会读取这一节。

### `pair_sampling.random_seed`

pair sampling 随机种子。

当前默认值：

```json
13
```

### `pair_sampling.easy_random_per_positive`

每个正例采样多少个 easy random negative。

当前默认值：

```json
2
```

### `pair_sampling.hard_bm25_per_positive`

每个正例采样多少个 BM25 hard negative。

当前默认值：

```json
2
```

### `pair_sampling.hard_dense_per_positive`

每个正例采样多少个 dense hard negative。

当前默认值：

```json
0
```

注意：启用 dense hard negative 需要当前环境能加载配置中的 dense encoder。`full` profile 当前覆盖为 `2`。

### `pair_sampling.hard_graph_neighbor_per_positive`

每个正例采样多少个 graph-neighbor hard negative。

当前默认值：

```json
1
```

### `pair_sampling.hard_pool_size`

hard retriever negative 的候选池大小。

当前默认值：

```json
30
```

约束：

- 所有采样数量必须是非负整数。
- 当 hard retriever negative 数量大于 0 时，`hard_pool_size` 必须为正整数。

## `selection`

训练中选择 best checkpoint 的指标设置。

当前字段：

```json
"selection": {
  "best_metric": "dev_composite",
  "higher_is_better": true
}
```

当前接线状态：

- 字段会进入 resolved training config 和 manifest。
- 当前训练代码的 best metric 仍硬编码为 dev composite：

```text
0.50 * Full Support@5 + 0.30 * Recall@5 + 0.20 * MRR
```

因此目前不要把 `best_metric` 改成其他值来期待训练逻辑自动变化。这个 section 更接近已记录但尚未完全参数化的配置接口。

## `reporting`

训练报告相关设置。

当前字段：

```json
"reporting": {
  "render_training_curves": true
}
```

当前接线状态：

- 字段会进入 resolved training config 和 manifest。
- 当前 experiment runner 还没有自动根据这个字段渲染训练曲线。
- 如果需要曲线图，仍需要使用已有 plotting/report 脚本或手动生成。

## 推荐修改方式

### 改 quick batch size

修改：

```json
"profiles": {
  "quick": {
    "optimization": {
      "batch_size": 16
    }
  }
}
```

### 改 full 训练 epoch

修改：

```json
"profiles": {
  "full": {
    "optimization": {
      "epochs": 20
    }
  }
}
```

### 临时改 CPU 训练

修改当前 profile：

```json
"profiles": {
  "quick": {
    "optimization": {
      "device": "cpu"
    }
  }
}
```

### 改成本地 encoder

修改：

```json
"defaults": {
  "encoder": {
    "model": "models/intfloat-e5-base-v2"
  }
}
```

同时确认 experiment config 中 dense baseline 的 `defaults.dense_encoder` 是否也应该同步修改。
