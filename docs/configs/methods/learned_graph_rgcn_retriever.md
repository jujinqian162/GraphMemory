# learned_graph_rgcn_retriever.json

对应配置文件：`configs/methods/learned_graph_rgcn_retriever.json`

这是 learned graph R-GCN trainable retriever 的当前方法配置。它是独立 public method，不是 `dense_rgcn_graph_retriever` 的开关。workflow 会为它构建 method-local proposal graph，并自动训练/复用 `dense_ft` 的 best model directory 作为 R-GCN seed encoder，然后用 learned edge gate 缩放 R-GCN message edge，并用 rank、edge、sparse 三项 loss 训练。

## 使用位置

experiment config 通过 `method_configs` 引用本文件：

```json
"method_configs": {
  "learned_graph_rgcn_retriever": "configs/methods/learned_graph_rgcn_retriever.json"
}
```

workflow 初始化后会写入：

```text
runs/<experiment>/learned/learned_graph_rgcn_retriever/effective_method_config.json
```

低层脚本仍只消费 generated stage config：

```text
python scripts/build_proposal_graphs.py --dataset hotpotqa --method learned_graph_rgcn_retriever ...
python scripts/build_train_pairs.py --config runs/<experiment>/config/stages/pairs.dense_ft.json
python scripts/train_method.py --config runs/<experiment>/config/stages/train.dense_ft.json
python scripts/build_train_pairs.py --config runs/<experiment>/config/stages/pairs.learned_graph_rgcn_retriever.json
python scripts/train_method.py --config runs/<experiment>/config/stages/train.learned_graph_rgcn_retriever.json
python scripts/run_retrieval.py --config runs/<experiment>/config/stages/retrieve.learned_graph_rgcn_retriever.json
```

只选择 `learned_graph_rgcn_retriever` 时，`dense_ft` 只作为 pair/train 依赖出现；除非用户显式选择 `dense_ft`，否则不会生成 `dense_ft` 的 retrieve/evaluate 输出行。

## 字段

- `method`: 固定为 `learned_graph_rgcn_retriever`。
- `encoder`: method config 中的基础 Sentence-Transformers encoder 设置。正常 workflow 训练时，R-GCN 的有效 encoder 会被 `dense_ft` best model directory 覆盖，并写入 checkpoint metadata；该字段仍用于当前配置解析和 hard-dense pair sampling 的基础 encoder 设置。
- `proposal_graph`: proposal graph 的召回上限，默认 `max_query_overlap=80`、`max_entity_neighbors=30`、`max_bridge_edges=200`、`use_spacy=false`。它复用现有 graph builder 规则，但输出到该 method 自己的 proposal graph artifact。
- `pairs`: 复用 train-pair 采样配置。输入 graph 为 proposal graph，不是共享 `graphs` artifact。
- `train.loss`: 三项 loss 权重，默认 `rank_loss_weight=1.0`、`edge_loss_weight=0.2`、`sparse_loss_weight=0.05`。
- `train.model`: R-GCN 模型结构配置。
- `train.trainer`: 训练循环参数。
- `train.selection` 和 `train.reporting`: best checkpoint 选择与报告开关。
- `profiles`: 对上述字段的 profile 覆盖。profile 名必须与 experiment `--profile` 对齐；`dev` 使用和 `smoke` 相同的小训练设置，用于 canonical HotpotQA dev-heldout profile。

## Label Boundary

`gold_dependency_edges` 只作为 edge loss 的 label-side supervision 和 path metrics 标签使用。proposal graph 构建、检索输入 graph tensor、test-time retrieval 都不能把 gold dependency edge 直接插入图里。

HotpotQA 通常没有 dependency edge labels，因此 edge loss sample count 可以为 0；rank loss 和 sparse gate penalty 仍然有效。2Wiki 的 dependency labels 只有在 proposal graph 已经包含同端点 candidate edge 时才会标正例。

## 产物

proposal graph artifact 路径形如：

```text
runs/<experiment>/graphs/train.learned_graph_rgcn_retriever.graphs.json
```

checkpoint 路径形如：

```text
runs/<experiment>/learned/learned_graph_rgcn_retriever/checkpoints/best.pt
```

checkpoint metadata 的 method 必须是 `learned_graph_rgcn_retriever`。用 `dense_rgcn_graph_retriever` checkpoint 加载该 method，或反向加载，都应在 retrieval 前失败。
