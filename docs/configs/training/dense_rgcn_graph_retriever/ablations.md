# dense_rgcn_graph_retriever/ablations.json

对应配置文件：`configs/training/dense_rgcn_graph_retriever/ablations.json`

这个文件是 R-GCN 消融实验的 training config 模板。它和 `base.json` 的结构基本一致，但额外增加了 `ablation_overrides`，集中列出不同 ablation 应覆盖的 model 字段。

## 当前接线状态

需要注意：当前 experiment runner 已经能读取普通 training config 的 `defaults + profiles.<profile>`，但还没有自动展开 `ablation_overrides` 为多组 run。

因此当前这个文件适合用作：

- 手动选择某个 ablation 配置的模板。
- 后续实现 ablation runner 时的约定基础。
- 记录每个 ablation 应该覆盖哪些字段。

如果现在直接把 experiment config 的 `training_configs.dense_rgcn_graph_retriever` 指向本文件，runner 会正常解析 `defaults` 和 `profiles`，但不会自动遍历 `ablation_overrides`。

## 和 base.json 相同的字段

以下字段含义与 `base.json` 一致：

- `schema_version`
- `method`
- `default_profile`
- `defaults.encoder`
- `defaults.model`
- `defaults.optimization`
- `defaults.pair_sampling`
- `defaults.selection`
- `defaults.reporting`
- `profiles.smoke`
- `profiles.quick`
- `profiles.full`

详细字段解释见：

```text
docs/configs/training/dense_rgcn_graph_retriever/base.md
```

## `ablation_overrides`

`ablation_overrides` 是一个按 ablation 名称组织的覆盖表。

当前字段：

```json
"ablation_overrides": {
  "full_rgcn": {},
  "wo_bridge": {
    "model": {
      "ablation": "wo_bridge"
    }
  },
  "wo_edge_type": {
    "model": {
      "ablation": "wo_edge_type"
    }
  },
  "wo_edge_weight": {
    "model": {
      "ablation": "wo_edge_weight"
    }
  },
  "wo_graph": {
    "model": {
      "ablation": "wo_graph",
      "num_layers": 0
    }
  },
  "wo_seed_score": {
    "model": {
      "ablation": "wo_seed_score"
    }
  }
}
```

### `full_rgcn`

完整 R-GCN，不覆盖任何字段。

等价于使用 `defaults.model.ablation = "full_rgcn"`。

### `wo_bridge`

禁用 bridge edge 的消融。

覆盖：

```json
"model": {
  "ablation": "wo_bridge"
}
```

### `wo_edge_type`

不区分 edge type 的消融。

覆盖：

```json
"model": {
  "ablation": "wo_edge_type"
}
```

### `wo_edge_weight`

不使用 artifact edge weight 的消融。

覆盖：

```json
"model": {
  "ablation": "wo_edge_weight"
}
```

### `wo_graph`

不使用图消息传递的消融。

覆盖：

```json
"model": {
  "ablation": "wo_graph",
  "num_layers": 0
}
```

`num_layers: 0` 很重要，它让模型结构进入 identity graph encoder 路径。

### `wo_seed_score`

移除 seed score 相关特征的消融。

覆盖：

```json
"model": {
  "ablation": "wo_seed_score"
}
```

## 当前手动使用方式

如果现在要跑某个 ablation，推荐复制本文件或复制 `base.json` 后手动改：

```json
"defaults": {
  "model": {
    "ablation": "wo_seed_score"
  }
}
```

或者只改某个 profile：

```json
"profiles": {
  "quick": {
    "model": {
      "ablation": "wo_seed_score"
    }
  }
}
```

为了避免同一 method 输出互相覆盖，建议为不同 ablation 使用不同 run name。

## 后续如果实现自动 ablation runner

建议规则是：

```text
resolved_config = defaults + profiles.<profile> + ablation_overrides.<ablation>
```

并把 ablation 名称写入 run name 或 artifact path，避免覆盖：

```text
runs/<experiment_name>_<ablation>/
```
