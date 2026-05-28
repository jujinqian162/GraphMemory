# graph_rerank.json

对应配置文件：`configs/search_spaces/graph_rerank.json`

这个文件定义 graph rerank 的 tuning search space。它不用于 R-GCN trainable retriever 训练，而是用于 `bm25_graph_rerank` 和 `dense_graph_rerank` 的 `tune` stage。

## 使用位置

experiment config 通过下面字段引用本文件：

```json
"search_spaces": {
  "graph_rerank": "configs/search_spaces/graph_rerank.json"
}
```

`tune` stage 会读取它，并枚举各个 list 字段形成 grid search candidate。每个 candidate 会在 dev split 上评估，然后选出一个 run-local tuned config：

```text
runs/<experiment_name>/tuned/<method>.dev_selected.json
```

`retrieve` stage 使用的是 tuned config，而不是 search space 原文件。

## 字段说明

### `lambda_init`

初始 seed retrieval 分数的权重。

当前候选：

```json
[1.0]
```

含义：

- 值越大，rerank 越保留 BM25/dense 的原始排序信号。
- 当前只搜索 `1.0`，表示不调这个维度。

### `lambda_query`

query-overlap graph signal 的权重候选。

当前候选：

```json
[0.0, 0.05, 0.1, 0.2]
```

含义：

- 控制 query 与 memory node 直接 overlap 边对 rerank 的影响。
- `0.0` 表示禁用该信号。

### `lambda_neighbor`

memory-memory neighbor graph signal 的权重候选。

当前候选：

```json
[0.0, 0.05, 0.1, 0.2, 0.4]
```

含义：

- 控制从 seed node 向邻居传播的图信号强度。
- 通常是 graph rerank 的主要调参项之一。

### `lambda_bridge`

bridge edge graph signal 的权重候选。

当前候选：

```json
[0.0, 0.05, 0.1, 0.2]
```

含义：

- 控制 bridge 边对 evidence rerank 的影响。
- `0.0` 表示禁用 bridge signal。

### `lambda_path`

path signal 的权重候选。

当前候选：

```json
[0.0]
```

含义：

- 为 path-based graph signal 预留。
- 当前只使用 `0.0`，相当于不启用 path signal。

### `max_hops`

图传播最大 hop 数候选。

当前候选：

```json
[1, 2]
```

含义：

- `1`：只考虑一跳邻居。
- `2`：允许更远的二跳传播。

值越大，图传播范围越大，但也更容易引入噪声。

### `seed_top_s`

参与图传播的 seed retrieval top-s 候选。

当前候选：

```json
[20, 30]
```

含义：

- 只从初始 BM25/dense 排名前 `s` 的 node 中取 seed signal。
- 值越大，图传播覆盖更多初始候选，也可能带入更多低质量 seed。

### `neighbor_type_weights`

不同 memory-memory edge type 的固定权重。

当前值：

```json
"neighbor_type_weights": {
  "bridge": 1.0,
  "entity_overlap": 0.7,
  "sequential": 0.3
}
```

字段含义：

- `bridge`：bridge edge 权重。
- `entity_overlap`：entity overlap edge 权重。
- `sequential`：同一 source 内顺序相邻 edge 权重。

注意：

- 当前配置使用 `neighbor_type_weights`。
- 旧字段名 `type_weights` 已废弃；如果旧 tuned config 还在用 `type_weights`，应先转换再运行。

## 搜索规模

当前 grid size 为：

```text
len(lambda_init)
* len(lambda_query)
* len(lambda_neighbor)
* len(lambda_bridge)
* len(lambda_path)
* len(max_hops)
* len(seed_top_s)

= 1 * 4 * 5 * 4 * 1 * 2 * 2
= 320 candidates
```

每个 method 会在 dev split 上评估这些 candidates。`dense_graph_rerank` 需要 dense encoder 可用。

## 常见修改方式

### 缩小 quick tuning 搜索

例如只测更小范围：

```json
"lambda_query": [0.0, 0.1],
"lambda_neighbor": [0.0, 0.2],
"lambda_bridge": [0.0, 0.1],
"max_hops": [1],
"seed_top_s": [20]
```

### 固定某个参数

把候选 list 改成单元素即可：

```json
"max_hops": [1]
```

### 调整 edge type 权重

修改：

```json
"neighbor_type_weights": {
  "bridge": 1.0,
  "entity_overlap": 0.5,
  "sequential": 0.2
}
```

这个字段不是 list，不参与 grid 枚举；它会应用到所有 candidate。
