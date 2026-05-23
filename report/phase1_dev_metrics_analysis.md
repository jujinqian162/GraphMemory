# Phase 1 Dev Metrics Analysis

## 1. 概述

本报告分析 Phase 1 HotpotQA evidence-tracing 实验在 7000+ 条 dev 数据上的阶段性结果。当前实验比较四类方法：

- `bm25`
- `bm25_graph_rerank`
- `dense`
- `dense_graph_rerank`

Phase 1 的核心目标不是回答生成，而是评估检索方法是否能够从候选 memory sentences 中找回完整 supporting evidence，并进一步观察这些证据在构造图中的连通情况。因此，除传统的 Recall、F1、MRR 之外，`Full Support`、`Connected Evidence Recall` 和 `Query-Evidence Connectivity` 是更贴近本阶段目标的指标。

整体来看，图重排带来了稳定但有限的提升。提升最明显的位置出现在 BM25 基础上的完整证据恢复和连通证据恢复；在 dense baseline 上，图重排仍有小幅增益，但增益幅度较小。这说明当前图重排更像是证据补全与连通性增强模块，而不是显著改善 first-hit ranking 的模块。

## 2. 指标说明

主结果表包含以下指标：

- `Recall@k`：top-k 中召回的 gold evidence nodes 比例。
- `Evidence F1@k`：基于 top-k 的 evidence precision 和 recall 的调和平均。
- `Full Support@k`：top-k 是否包含一个样本的全部 gold evidence nodes。该指标比 Recall 更严格，适合衡量多跳证据是否找全。
- `MRR`：第一个 gold evidence node 的 reciprocal rank，反映最早命中证据的位置。

路径与连通性结果包含以下指标：

- `Connected Evidence Recall@k`：top-k 包含全部 gold evidence nodes，并且这些 gold evidence nodes 在 top-k induced graph 中连通。
- `Query-Evidence Connectivity@10`：top-10 包含全部 gold evidence nodes，并且这些证据从 query node `q` 可达。
- `Path Recall@10` 和 `Edge Recall@10`：当前 HotpotQA Phase 1 没有显式 gold dependency path / gold dependency edge 标注，因此按设计输出为 `N/A`。

效率表包含以下指标：

- `Retrieval Latency / Query`：平均单 query 检索耗时，按当前实现语义可理解为 ms/query。
- `Memory Size`：平均 memory node 数量。
- `Avg Retrieved Nodes`：平均返回子图中的节点数量。
- `Avg Retrieved Edges`：平均返回子图中的边数量。
- `Index Build Time` 和 `Graph Construction Time`：当前结果中为 `0.0`，更适合解释为本次汇总未记录对应构建耗时，而不应解释为真实成本为零。

## 3. 主结果分析

### 3.1 主表结果

| Method | Recall@2 | Recall@5 | Recall@10 | Evidence F1@5 | Evidence F1@10 | Full Support@5 | Full Support@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.4738 | 0.6788 | 0.8127 | 0.4309 | 0.3101 | 0.3750 | 0.6004 | 0.8270 |
| bm25_graph_rerank | 0.4818 | 0.7194 | 0.8610 | 0.4569 | 0.3295 | 0.4490 | 0.7011 | 0.8222 |
| dense | 0.5448 | 0.7697 | 0.8887 | 0.4882 | 0.3404 | 0.5320 | 0.7540 | 0.8530 |
| dense_graph_rerank | 0.5389 | 0.7777 | 0.8946 | 0.4938 | 0.3429 | 0.5497 | 0.7706 | 0.8524 |

从主结果看，`dense_graph_rerank` 在 `Recall@5`、`Recall@10`、`Evidence F1@5`、`Evidence F1@10`、`Full Support@5` 和 `Full Support@10` 上取得最高结果；`dense` 在 `Recall@2` 和 `MRR` 上最高。

这表明 dense baseline 已经能够较好地将至少一个 gold evidence node 排在前面，而图重排的主要收益集中在 top-5/top-10 范围内补齐更多 supporting evidence。换言之，图重排对完整证据恢复更有帮助，但没有明显改善第一个证据命中的排序位置。

### 3.2 BM25 加图的变化

相对于 `bm25`，`bm25_graph_rerank` 的提升如下：

| Metric | Absolute Change |
|---|---:|
| Recall@2 | +0.0079 |
| Recall@5 | +0.0406 |
| Recall@10 | +0.0483 |
| Evidence F1@5 | +0.0261 |
| Evidence F1@10 | +0.0194 |
| Full Support@5 | +0.0740 |
| Full Support@10 | +0.1007 |
| MRR | -0.0048 |

BM25 加图后，`Full Support@10` 从 0.6004 提升到 0.7011，绝对提升约 10.07 个百分点。这是主表中最明显的收益之一，说明图重排能够帮助 BM25 补齐原本没有进入 top-k 的 supporting evidence。

同时，`MRR` 从 0.8270 小幅下降到 0.8222。这说明图重排并不总是让第一个 gold evidence 更靠前；它更可能是在 top-k 内调整证据集合，使更多相关证据同时进入候选范围。

### 3.3 Dense 加图的变化

相对于 `dense`，`dense_graph_rerank` 的变化如下：

| Metric | Absolute Change |
|---|---:|
| Recall@2 | -0.0058 |
| Recall@5 | +0.0080 |
| Recall@10 | +0.0059 |
| Evidence F1@5 | +0.0056 |
| Evidence F1@10 | +0.0025 |
| Full Support@5 | +0.0178 |
| Full Support@10 | +0.0166 |
| MRR | -0.0006 |

Dense baseline 本身已经较强，因此图重排的增益明显小于 BM25 场景。`Full Support@10` 从 0.7540 提升到 0.7706，绝对提升约 1.66 个百分点；`Recall@10` 从 0.8887 提升到 0.8946，绝对提升约 0.59 个百分点。

这些提升幅度有限，但方向基本一致：除 `Recall@2` 和 `MRR` 外，主要 top-k evidence retrieval 指标均有小幅提升。这说明在强 dense 检索器之上，当前图重排仍能提供一定补充价值，但其边际收益不大。

## 4. 连通性结果分析

### 4.1 路径与连通性结果

| Method | Connected Evidence Recall@5 | Connected Evidence Recall@10 | Query-Evidence Connectivity@10 | Path Recall@10 | Edge Recall@10 |
|---|---:|---:|---:|---|---|
| bm25 | 0.2632 | 0.4565 | 0.5948 | N/A | N/A |
| bm25_graph_rerank | 0.3520 | 0.5721 | 0.6988 | N/A | N/A |
| dense | 0.3773 | 0.5946 | 0.7473 | N/A | N/A |
| dense_graph_rerank | 0.4065 | 0.6221 | 0.7669 | N/A | N/A |

连通性结果进一步说明图重排的主要作用在于提高证据集合的结构完整性。`dense_graph_rerank` 在三个可用连通性指标上均取得最高结果：

- `Connected Evidence Recall@5 = 0.4065`
- `Connected Evidence Recall@10 = 0.6221`
- `Query-Evidence Connectivity@10 = 0.7669`

这些指标比普通 Recall 更贴近 evidence-tracing 目标。普通 Recall 只要求 gold evidence 出现在 top-k 中，而连通性指标进一步要求这些 evidence 在图结构中形成可解释的连接关系。

### 4.2 BM25 加图的连通性提升

相对于 `bm25`，`bm25_graph_rerank` 的连通性提升如下：

| Metric | Absolute Change |
|---|---:|
| Connected Evidence Recall@5 | +0.0888 |
| Connected Evidence Recall@10 | +0.1156 |
| Query-Evidence Connectivity@10 | +0.1039 |

这是当前结果中最强的图结构证据。BM25 加图后，不仅 top-k 中包含更多完整证据，而且这些证据在构造图中的连通性也明显改善。

不过，该结果也应谨慎表述。`Connected Evidence Recall@10` 提升到 0.5721，说明仍有相当比例样本没有形成完整连通证据恢复。因此当前方法可以被认为改善了证据连通性，但还不能说明其已经充分解决多跳证据路径恢复问题。

### 4.3 Dense 加图的连通性提升

相对于 `dense`，`dense_graph_rerank` 的连通性变化如下：

| Metric | Absolute Change |
|---|---:|
| Connected Evidence Recall@5 | +0.0291 |
| Connected Evidence Recall@10 | +0.0275 |
| Query-Evidence Connectivity@10 | +0.0197 |

在 dense baseline 上，图重排仍然提升了连通性，但提升幅度较小。该现象与主表一致：dense baseline 已经较强，图模块主要提供边际补充。

值得注意的是，`Query-Evidence Connectivity@10` 高于 `Connected Evidence Recall@10`。这符合指标定义：前者要求 query node 能到达 gold evidence nodes，后者更强调 gold evidence nodes 在 top-k induced graph 中互相连通。因此，前者通常更容易满足。

## 5. 效率分析

### 5.1 效率结果

| Method | Index Build Time | Graph Construction Time | Retrieval Latency / Query | Memory Size | Avg Retrieved Nodes | Avg Retrieved Edges |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.0 | 0.0 | 0.6734 | 41.3408 | 9.9856 | 0.0000 |
| bm25_graph_rerank | 0.0 | 0.0 | 1.1508 | 41.3408 | 9.9856 | 34.2372 |
| dense | 0.0 | 0.0 | 36.6765 | 41.3408 | 9.9856 | 0.0000 |
| dense_graph_rerank | 0.0 | 0.0 | 37.5994 | 41.3408 | 9.9856 | 35.2995 |

表中的 `Index Build Time` 和 `Graph Construction Time` 均为 0.0，主要反映当前指标汇总没有从运行元数据中填充索引构建和图构建耗时；因此该表可用于比较 retrieval 阶段的平均查询延迟，但不应被解读为索引或图构建没有成本。

BM25 加图后，平均延迟从 0.6734 ms/query 增加到 1.1508 ms/query，绝对增加约 0.4774 ms/query。相对增幅较高，但这是因为 BM25 本身延迟很低；从绝对耗时看，图重排开销仍然较小。

Dense 加图后，平均延迟从 36.6765 ms/query 增加到 37.5994 ms/query，绝对增加约 0.9229 ms/query，相对增加约 2.52%。这说明在 dense retrieval 场景中，主要开销仍来自 dense encoding / scoring，图重排的额外耗时较有限。

### 5.2 Retrieved subgraph 规模

Flat 方法的 `Avg Retrieved Edges` 为 0，这是因为它们不返回图边。Graph rerank 方法平均返回约 34 到 35 条边：

- `bm25_graph_rerank`: 34.2372
- `dense_graph_rerank`: 35.2995

这说明 graph rerank 输出的不只是节点排序，还包含 top-k induced subgraph。该信息也为解释具体检索案例提供了可观察的结构线索。

同时，平均 35 条边对于约 10 个返回节点来说并不算非常稀疏。因此，当前连通性结果应理解为在已有图构建规则下的阶段性结构恢复表现，而不是对具体边类型贡献的单独归因。

## 6. 综合判断

当前结果支持以下判断：

1. 图重排对 BM25 的帮助较明显。  
   BM25 加图后，`Full Support@10` 提升约 10.07 个百分点，`Connected Evidence Recall@10` 提升约 11.56 个百分点。这说明图结构对于弥补 sparse retrieval 的多跳证据缺失有较明显作用。

2. 图重排对 dense 的帮助存在但幅度有限。  
   Dense 加图后，`Full Support@10` 提升约 1.66 个百分点，`Connected Evidence Recall@10` 提升约 2.75 个百分点。考虑到 dense baseline 本身已经较强，这一结果说明图重排有一定补充作用，但不是数量级上的提升。

3. 图重排主要改善完整证据恢复和证据连通性，而不是 first-hit ranking。  
   `Recall@2` 和 `MRR` 在 dense 加图后略降或基本持平。这表明当前方法更适合作为 evidence set completion / connectivity enhancement，而不是单纯的 first evidence reranker。

4. 效率开销总体可接受。  
   Dense 场景下图重排额外增加约 0.92 ms/query，BM25 场景下额外增加约 0.48 ms/query。相对于 Full Support 和连通性指标上的提升，该开销处于可接受范围。

5. 当前结果仍属于 Phase 1 阶段性验证。  
   当前数据为 dev 结果，且图重排参数可能经过 dev tuning；因此该结果反映的是 dev setting 下的阶段性表现。同时，HotpotQA Phase 1 没有显式 gold dependency paths，因此当前结果不能被表述为严格的 gold path recovery 提升。
