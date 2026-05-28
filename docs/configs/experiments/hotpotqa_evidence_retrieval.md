# hotpotqa_evidence_retrieval.json

对应配置文件：`configs/experiments/hotpotqa_evidence_retrieval.json`

这个文件是 HotpotQA evidence retrieval 的默认 experiment runner 配置。它决定一次实验使用什么数据集、哪些 split、哪些 retrieval method、图构建参数、graph-rerank tuning 搜索空间，以及 trainable R-GCN retriever 应引用哪个 training config。

它不直接保存 R-GCN 的 batch size、epochs、hidden dim 等训练超参。训练细节由 `training_configs.dense_rgcn_graph_retriever` 指向的 training config 管理。

## 使用位置

- `scripts/experiment.py init ... --config configs/experiments/hotpotqa_evidence_retrieval.json`
- `scripts/experiment.py run ... --config configs/experiments/hotpotqa_evidence_retrieval.json`
- `graph_memory.experiment.load_experiment_config`
- `graph_memory.experiment.build_effective_config`

初始化 run 后，resolved config 会写入：

```text
runs/<experiment_name>/config/effective_config.json
runs/<experiment_name>/manifest.json
```

之后同名 run 的 `plan` / `run` 会优先读 manifest 中已经冻结的配置。修改源 config 文件不会自动改变已有 run；需要换 run name 或重新 init。

## 顶层字段

### `recipe`

实验 recipe 名称。

当前值：

```json
"hotpotqa_evidence_retrieval"
```

用途：

- 标识这个 experiment profile 的实验意图。
- 写入 manifest，方便之后区分 run 类型。

当前代码不会根据不同 `recipe` 分派不同 pipeline；它主要是记录和复现元数据。

### `dataset`

数据集名称。

当前值：

```json
"hotpotqa"
```

用途：

- 写入 effective config 和 manifest。
- 帮助人读 run summary 时判断数据来源。

### `task`

任务名称。

当前值：

```json
"evidence_retrieval"
```

用途：

- 标识当前任务是 evidence retrieval。
- 当前代码主要把它作为元数据保存。

### `raw`

原始数据路径映射。

当前字段：

```json
"raw": {
  "dev": "data/hotpotqa/raw/dev.json",
  "train": "data/hotpotqa/raw/train.json"
}
```

字段含义：

- `train`：HotpotQA train 原始文件路径。
- `dev`：HotpotQA dev 原始文件路径。

这些路径由 `prepare` stage 使用。路径应相对 repo root，除非你明确传入绝对路径。

### `split_sources`

每个实验 split 从哪个 raw split 读取数据。

当前字段：

```json
"split_sources": {
  "train": "train",
  "dev": "dev",
  "test": "dev"
}
```

字段含义：

- `train`：训练 split 使用 `raw.train`。
- `dev`：验证 / tuning split 使用 `raw.dev`。
- `test`：测试 split 使用 `raw.dev`，通过 offset 与 dev 区分。

注意：当前 HotpotQA 配置没有单独的 raw test label，因此 test 默认从 dev 中切出一段。

### `split_offsets`

每个 split 从对应 raw 文件的第几个有效样本开始取。

当前字段：

```json
"split_offsets": {
  "train": 0,
  "dev": 0,
  "test": 500
}
```

字段含义：

- `train`: train split 从 train raw 的第 0 个有效样本开始。
- `dev`: dev split 从 dev raw 的第 0 个有效样本开始。
- `test`: test split 从 dev raw 的第 500 个有效样本开始。

这样可以让 dev 和 test 尽量不重叠。修改 offset 会改变数据切片，已有 run 不会自动更新。

### `defaults`

experiment runner 的默认参数。

当前字段：

```json
"defaults": {
  "dense_encoder": "intfloat/e5-base-v2",
  "passage_prefix": "passage: ",
  "query_prefix": "query: ",
  "seed": 13,
  "top_k": 10
}
```

字段含义：

- `dense_encoder`：dense retrieval 和 graph-rerank seed dense retrieval 使用的 Sentence-Transformers 模型名或本地模型路径。
- `query_prefix`：dense encoder 编码 query 前拼接的前缀。
- `passage_prefix`：dense encoder 编码 memory passage 前拼接的前缀。
- `seed`：prepare split sampling 使用的随机种子。
- `top_k`：retrieval 输出和 evaluation 使用的默认 top-k。

注意：R-GCN training 自己的 encoder 配置在 training config 中；这里的 `dense_encoder` 不等同于 R-GCN 训练 encoder，除非两个 config 手动保持一致。

### `profiles`

不同运行规模的 profile。experiment runner 的 `--profile` 会选择这里的一个 profile，并覆盖 `defaults` 中由 runner 构造出的 split size。

当前 profile：

- `smoke`：1 train / 1 dev / 1 test，用于最小流程检查。
- `quick`：100 train / 100 dev / 100 test，用于快速实验。
- `full`：5000 train / 500 dev / 1000 test，用于较大规模实验。

每个 profile 字段：

- `train_examples`：prepare train split 时最多取多少样本。
- `dev_examples`：prepare dev split 时最多取多少样本。
- `test_examples`：prepare test split 时最多取多少样本。

### `graph`

图构建参数，由 `graphs` stage 传给 `scripts/build_graphs.py`。

当前字段：

```json
"graph": {
  "max_bridge_edges": 50,
  "max_entity_neighbors": 10,
  "max_query_overlap": 20,
  "use_spacy": false
}
```

字段含义：

- `max_query_overlap`：最多保留多少条 query-overlap 边。
- `max_entity_neighbors`：每个 memory node 最多连接多少个 entity-overlap neighbor。
- `max_bridge_edges`：最多保留多少条 bridge 边。
- `use_spacy`：是否使用 spaCy 做 entity extraction。

约束：

- 三个 `max_*` 字段应为正整数。
- `use_spacy` 为布尔值。设为 `true` 前需要确认当前环境安装并可加载 spaCy 相关模型。

### `methods`

默认参与实验的 retrieval method 列表。

当前可用值：

- `bm25`
- `dense`
- `bm25_graph_rerank`
- `dense_graph_rerank`
- `dense_rgcn_graph_retriever`

用途：

- `init` 时如果没有传 `--methods`，runner 使用这里的完整列表。
- 如果传了 `--methods`，会用 CLI 选择的 method 子集。

### `search_spaces`

tuning search-space 配置路径。

当前字段：

```json
"search_spaces": {
  "graph_rerank": "configs/search_spaces/graph_rerank.json"
}
```

字段含义：

- `graph_rerank`：`tune` stage 为 `bm25_graph_rerank` 和 `dense_graph_rerank` 使用的 grid search 配置。

### `training_configs`

trainable method 到 training config 的映射。

当前字段：

```json
"training_configs": {
  "dense_rgcn_graph_retriever": "configs/training/dense_rgcn_graph_retriever/base.json"
}
```

字段含义：

- key 必须是 trainable retrieval method 名称。
- value 是该 method 的 training config 路径。

experiment runner 会按当前 experiment `--profile` 解析 training config 中同名 profile。例如 experiment profile 为 `quick` 时，会解析 training config 的 `profiles.quick`。

## 常见修改位置

### 改 quick 运行规模

修改：

```json
"profiles": {
  "quick": {
    "train_examples": 200,
    "dev_examples": 100,
    "test_examples": 100
  }
}
```

### 改 dense baseline 模型

修改：

```json
"defaults": {
  "dense_encoder": "models/intfloat-e5-base-v2"
}
```

如果使用本地模型路径，需要确认路径在运行机器上存在。

### 改 R-GCN training config

修改：

```json
"training_configs": {
  "dense_rgcn_graph_retriever": "configs/training/dense_rgcn_graph_retriever/base.json"
}
```

训练超参本身不要写在 experiment config 里，应写在对应 training config。
