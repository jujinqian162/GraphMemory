# Config Documentation

本目录记录 `configs/` 下每个可编辑配置文件的中文说明。目标是让配置文件不再只能靠猜字段名使用，而是每个 JSON 文件都有一个一一对应的说明文件。

## 命名规则

说明文件尽量镜像 `configs/` 的目录结构：

```text
configs/experiments/hotpotqa_evidence_retrieval.json
docs/configs/experiments/hotpotqa_evidence_retrieval.md

configs/methods/dense_rgcn_graph_retriever.json
docs/configs/methods/dense_rgcn_graph_retriever.md

configs/methods/dense_ft_rgcn_graph_retriever.json
docs/configs/methods/dense_ft_rgcn_graph_retriever.md
```

如果新增一个正式配置文件，应同时新增对应的 `docs/configs/.../*.md`，并在本页登记。

## 当前配置说明

| Config | Documentation | 用途 |
|---|---|---|
| `configs/experiments/hotpotqa_evidence_retrieval.json` | `experiments/hotpotqa_evidence_retrieval.md` | HotpotQA evidence retrieval 的默认 experiment runner 配置。 |
| `configs/experiments/hotpoqa_dev_full.json` | `experiments/hotpoqa_dev_full.md` | 使用完整 dev 作为 test 区间的 HotpotQA dev-full 变体配置。 |
| `configs/experiments/hotpotqa_rgcn_ablation_selected.json` | `experiments/hotpotqa_rgcn_ablation_selected.md` | 服务器上运行选定 R-GCN variants 的 ablation 配置。 |
| `configs/methods/dense_rgcn_graph_retriever.json` | `methods/dense_rgcn_graph_retriever.md` | R-GCN trainable graph retriever 的当前方法配置。 |
| `configs/methods/dense_ft.json` | `methods/dense_ft.md` | Dense-FT trainable retriever 的当前方法配置。 |
| `configs/methods/dense_ft_rgcn_graph_retriever.json` | `methods/dense_ft_rgcn_graph_retriever.md` | 使用 Dense-FT checkpoint 作为 encoder seed 的 R-GCN trainable graph retriever 当前方法配置。 |
| `configs/search_spaces/graph_rerank.json` | `search_spaces/graph_rerank.md` | BM25/dense graph rerank 的 tuning search space。 |

## 维护原则

- 说明文档写“这个 config 文件怎么填”，不要复制整段运行命令。
- 如果字段已经在配置中存在但当前代码没有完全消费，必须明确标注“当前未完全接线”。
- 如果字段有可选值，必须列出当前代码实际支持的值。
- 如果字段会进入 run manifest 或 checkpoint，必须说明它影响复现或推理。
- 如果字段只是 debug 或未来预留，不要写成正式实验结果的一部分。
