# hotpoqa_dev_full.json

对应配置文件：`configs/experiments/hotpoqa_dev_full.json`

这个文件是 HotpotQA evidence retrieval experiment config 的 dev-full 变体。它和 `hotpotqa_evidence_retrieval.json` 基本相同，但额外提供了 `profiles.dev`，用于把 HotpotQA dev 中 offset 之后的大部分样本作为 test 区间。

文件名中的 `hotpoqa` 少了一个 `t`，这是当前仓库里的实际文件名。引用时必须使用当前路径：

```text
configs/experiments/hotpoqa_dev_full.json
```

## 主要用途

适合在没有官方 test label 的情况下，用 HotpotQA dev 文件切出更大的评估区间：

- `dev` profile 的 `dev_examples` 为 500。
- `dev` profile 的 `test_examples` 为 6869。
- `cloud-full` profile 的 `test_examples` 也为 6869，同时使用更大的 train split，适合服务器训练后做完整 dev held-out 评估。
- `test` split 仍从 `raw.dev` 读取，但使用 `split_offsets.test = 500`。

因此这个配置表达的是：

```text
dev:  raw dev offset 0,   取 500 条
test: raw dev offset 500, 取 6869 条
```

## 和默认配置的区别

相比 `configs/experiments/hotpotqa_evidence_retrieval.json`，此文件额外有：

```json
"profiles": {
  "cloud-quick": {
    "dev_examples": 500,
    "test_examples": 1000,
    "train_examples": 1000
  },
  "cloud-full": {
    "dev_examples": 500,
    "test_examples": 6869,
    "train_examples": 90447
  },
  "dev": {
    "dev_examples": 500,
    "test_examples": 6869,
    "train_examples": 1
  }
}
```

字段含义：

- `train_examples: 1`：dev-full 评估场景通常不用于完整训练，只保留极小 train split 让 prepare/graph 流程结构完整。
- `dev_examples: 500`：前 500 条 dev 样本用于 dev/tune。
- `test_examples: 6869`：从 offset 500 后取更多 dev 样本作为 test-like 评估集。
- `cloud-quick`：服务器快速训练 profile，使用 1000 条 train、500 条 dev、1000 条 test-like held-out 样本。
- `cloud-full`：服务器较完整训练 profile，使用 90447 条 train、500 条 dev、6869 条 test-like held-out 样本。

## 其他字段

除 `profiles.dev` 外，其余字段含义与 `hotpotqa_evidence_retrieval.json` 一致：

- `recipe`
- `dataset`
- `task`
- `raw`
- `split_sources`
- `split_offsets`
- `defaults`
- `profiles.smoke`
- `profiles.quick`
- `profiles.full`
- `profiles.cloud-quick`
- `profiles.cloud-full`
- `graph`
- `methods`
- `search_spaces`
- `training_configs`

详细解释见：

```text
docs/configs/experiments/hotpotqa_evidence_retrieval.md
```

## 使用注意

- 这个配置依赖 HotpotQA dev 文件中有足够多的有效样本。
- `test` 不是官方 blind test，而是从 labeled dev 切出的 held-out 区间。
- 如果修改 `split_offsets.test`，需要重新确认 dev/test 是否重叠。
- 已经 init 的 run 会使用 manifest 中冻结的配置；改源文件不会改变旧 run。
