# Provenance R-GCN Candidate Loss 版本与后续评估计划

> 状态：v2 batching/runtime 修复已实现，尚未完成正式 loss 对照实验  
> 实现规范：[`unify-rgcn-dataloader-batching`](../../openspec/changes/unify-rgcn-dataloader-batching/)  
> 当前保留版本：`provenance-candidate-loss-v2`  
> 当前候选损失：task-balanced pairwise logistic  
> 固定辅助损失：class-balanced logical-edge BCE  
> 适用方法：`execution_provenance_rgcn_retriever`

## 1. 本文档锁定的决定

当前版本暂时保留 feature 分支已经实现的 pairwise candidate loss，并将其候选损失协议标记为：

```text
provenance-candidate-loss-v2
```

在 pairwise 的正式训练结果出来之前，不因为实现复杂度或论文当前文字仍写着 BCE 而提前切回 BCE。未来根据同一套数据、负例、模型、推理和选择协议下的受控实验，再决定是否尝试或切换到带正类权重的 BCE、混合损失或 listwise 损失。

这里的 `v2` 是 **candidate-loss protocol version tag**，不是 Git tag，也不得与下列版本混淆：

- `twowiki_provenance` raw dataset schema v3；
- provenance R-GCN checkpoint schema v4；
- experiment artifact 或 Prefect task implementation version。

后续配置、checkpoint 和实验记录若增加显式字段，应分别保存：

```text
candidate_loss_protocol = provenance-candidate-loss-v2
candidate_loss_type = task_balanced_pairwise_logistic
```

本次 DataLoader/disconnected-union 迁移改变训练 runtime 与 checkpoint schema，但不改变 candidate loss 数学定义、创建 Git tag 或补写正式实验结果。

## 2. Git 历史结论

### 2.1 v1：Provenance R-GCN 最初使用 candidate BCE

当前 Provenance R-GCN 首次出现在提交：

```text
fe2a8b4  feat: add semantic execution provenance retrieval
```

其 candidate loss 是未加正类权重的 BCE：

```python
candidate_loss = F.binary_cross_entropy_with_logits(
    output.candidate_logits[pair_indices],
    output.candidate_logits.new_tensor(pair_targets),
)
```

同时保留 class-balanced logical-edge BCE：

```text
L = candidate_loss_weight * L_candidate_bce
  + edge_loss_weight * L_edge_bce
```

本文将这一历史协议记为：

```text
provenance-candidate-loss-v1
```

本地 `main` 的 `8fedf11` 和远端 `origin/main` 的 `9273f56` 都仍属于这一 BCE 路径。`origin/main` 的论文提交从 `8fedf11` 分叉，因此 `docs/raw/exp.tex` 写 candidate BCE 与 main 代码一致。

### 2.2 v2：ablation-validity 分支显式用 pairwise 替换 BCE

提交：

```text
9cb8ddf  feat: harden provenance R-GCN ablation validity (schema v3)
```

在 `feature/provenance-rgcn-ablation-validity` 分支中显式完成了：

```text
candidate BCE
    -> task-balanced pairwise logistic candidate loss
```

当前公式是：

\[
\mathcal{L}_{\mathrm{cand}}^{(t)}
=
\frac{1}{|P_t||N_t|}
\sum_{p\in P_t}\sum_{n\in N_t}
\log\left(1+\exp\left(-(s_p-s_n)\right)\right),
\]

其中每个 task 的所有已物化 positive 与所有去重后的 selected negative 两两比较。当前代码等价于：

```python
margins = (
    candidate_logits[positive_indices].unsqueeze(1)
    - candidate_logits[negative_indices].unsqueeze(0)
)
candidate_loss = F.softplus(-margins).mean()
```

edge loss 未被替换，仍为带 `pos_weight` 的 logical-edge BCE：

\[
\mathcal{L}
=
1.0\,\mathcal{L}_{\mathrm{candidate\ pairwise}}
+
0.5\,\mathcal{L}_{\mathrm{edge\ BCE}}.
\]

checkpoint 已记录：

```text
candidate_loss_type = task_balanced_pairwise_logistic
```

### 2.3 v2 不是 Provenance R-GCN 的原始必要组成

Git 历史证明 pairwise 是后续 hardening 决策，而不是最初方法定义。更早的 `b3b3f14 feat: add pairwise RGCN ranking loss` 针对的是已经移除/演进的另一条 `learned_graph_rgcn_retriever` 路径；它保留 BCE 并额外加入 pairwise，不能被视为当前 Provenance R-GCN 的初始实现。

## 3. 为什么 v2 引入 pairwise

`improve-provenance-rgcn-ablation-validity` 的设计背景包括：

1. generic BM25/dense hard negatives 曾将 candidate-pair 数从 `444,072` 增加到 `1,036,168`，但没有带来稳定的 node-ranking 收益；
2. 旧 pair builder 允许同一 candidate 因不同 `sample_type` 被重复计入监督；
3. 旧 candidate BCE 没有正类权重，在每个 task 只有少量 positive、较多 selected negative 时更偏向负类校准；
4. 最终推理使用 candidate logit 排序，pairwise 直接优化 `score(positive) > score(negative)`，比绝对二分类阈值更贴近排序目标；
5. v2 同时引入 provenance-successor / predecessor negatives，并要求每个 task 内按 candidate 去重，因此 pairwise 可以直接比较 gold output 与结构上真正竞争的 output。

需要保留一个边界：旧训练循环本来就是逐 task 计算一个 scalar loss 后进行梯度累积，所以“外层 task 等权”并非完全由 pairwise 首次带来。v2 的主要变化是 task 内正负关系的直接排序、正负比较的对称性，以及与 candidate 去重和 provenance-native hard negatives 的配套。

## 4. 为什么目前不能断言 pairwise 优于 BCE

`main` 与 `feature/provenance-rgcn-ablation-validity` 不能构成 loss ablation，因为 feature 同时改变了：

- raw dataset / checkpoint schema；
- gold-spine 与 matched semantic branch 图构造；
- feed edge 权重校准；
- provenance successor / predecessor negatives；
- candidate 级去重和 hardness precedence；
- structured candidate promotion 与 edge abstention；
- checkpoint selection objective；
- Edge Precision / Recall / F1 评价协议。

因此，即使 feature 结果优于 main，也不能把提升归因于 pairwise。当前 feature OpenSpec 的正式三 seed full-profile matrix 仍未完成，且尚无只改变 candidate loss 的受控结果。

当前决策“保留 v2”表示先完成并观察 pairwise 协议，而不是宣称它已经被证明优于 BCE。

## 5. 当前 v2 的正式语义

### 5.1 Candidate supervision

- 每个 gold ToolOutput 都必须作为 positive；
- negative 来自已物化、去重后的 provenance successor、provenance predecessor、dense、BM25 和 easy samplers；
- 每个 positive 与每个 selected negative 比较；
- comparison 在 task 内求平均；
- task loss 在 optimizer batch 内等权平均；
- 未进入 pair artifact 的 candidate 可以参与图消息传播，但不进入 candidate loss。

### 5.2 Edge supervision

edge head 继续使用 class-balanced BCE。切换 candidate loss 时不得顺便删除或改变 edge loss，因为 edge scorer、structured promotion 和 dependency-edge 输出都依赖该监督。

### 5.3 已实现的 DataLoader 与尾批归一化语义

`unify-rgcn-dataloader-batching` 已删除 `batch_size` 的旧伪批处理含义。`per_device_graph_batch_size` 现在只表示一次 disconnected-union 多图 forward、backward 和 optimizer step 中的 task graph 数量。

每个 DataLoader batch 的 candidate pairwise 与 edge BCE 仍先在 task 内计算，然后对该 batch 的实际 task 数做等权平均。最后一个不足配置图数的 batch 直接按其实际 task 数归一化，不跨 DataLoader batch 保留梯度。

实现和测试已经验证：

- disconnected union 不包含跨 task message edge；
- candidate/query/transition ownership 由 offsets 和显式 query indices 约束；
- pairwise comparison 只发生在同一 task 内；
- edge `pos_weight` 逐 task 计算，不在图 batch 全局重算；
- dropout 关闭时 batched 与 separate forward/ranking 在浮点容差内一致；
- 每个 DataLoader batch 恰好产生一次 optimizer/global step；
- candidate-loss protocol tag 仍为 `provenance-candidate-loss-v2`。

## 6. 计划中的 loss 版本

### 6.1 v1：历史 unweighted BCE，不再作为默认方案

标签：

```text
provenance-candidate-loss-v1
```

定义：

\[
\mathcal{L}_{t}
=
\operatorname{mean}_{i\in P_t\cup N_t}
\operatorname{BCEWithLogits}(s_i,y_i).
\]

状态：历史基线。其 main 结果不能直接与 v2 结果解释为 loss 差异，因为数据、图、采样、推理和选择协议不同。

### 6.2 v2：当前保留的 task-balanced pairwise logistic

标签：

```text
provenance-candidate-loss-v2
```

定义：每个 task 内所有 positive-negative comparison 的 logistic loss 均值，随后对 task 求平均；edge loss 保持不变。

状态：**当前锁定并保留，优先完成正式结果。**

### 6.3 v3 候选：normalized task-balanced BCE with positive weight

拟定标签：

```text
provenance-candidate-loss-v3
```

该版本尚未启用。推荐不是简单使用全局 BCE，而是在每个 task 内根据真实 selected pairs 计算：

\[
w_t=\frac{|N_t|}{|P_t|}.
\]

然后使用按有效权重归一化的 BCE：

\[
\mathcal{L}_{t}
=
\frac{
 w_t\sum_{p\in P_t}\operatorname{softplus}(-s_p)
 +\sum_{n\in N_t}\operatorname{softplus}(s_n)
}{w_t|P_t|+|N_t|}.
\]

采用显式 denominator，而不是直接依赖 `BCEWithLogitsLoss(pos_weight=..., reduction="mean")`，可以让不同正负比例 task 的 loss scale 更稳定。外层仍对 task 等权平均，edge BCE 不变。

目标：评估绝对二分类校准与正类平衡能否在更简单的实现下达到或超过 v2。

### 6.4 v4 候选：weighted BCE + pairwise hybrid

拟定标签：

```text
provenance-candidate-loss-v4
```

定义候选：

\[
\mathcal{L}_{\mathrm{cand}}
=
\lambda_{\mathrm{bce}}\mathcal{L}_{\mathrm{weighted\ BCE}}
+
\lambda_{\mathrm{pair}}\mathcal{L}_{\mathrm{pairwise}}.
\]

用途：同时保留 absolute calibration 和 relative ranking。该思路与历史提交 `b3b3f14` 的方向相似，但若启用必须重新定义适用于当前 provenance v3 数据和 pair contract 的权重，不得直接复制旧方法参数。

优先级低于 v3，只有在 v2 排序较强但 calibration/threshold 较差，或 v3 calibration 较好但排序退化时才考虑。

### 6.5 v5 候选：task-local listwise / sampled softmax

拟定标签：

```text
provenance-candidate-loss-v5
```

状态：探索性、低优先级。该版本把每个 task 的 selected candidate 视为一个 ranking list，使用 multi-positive listwise objective 或 sampled softmax。只有 v2/v3/v4 都暴露出明确限制时再设计；当前不得为追求复杂性提前实现。

## 7. 未来 loss 对照实验必须满足的控制条件

任何 `v2 vs v3` 或后续比较必须只改变 candidate-loss protocol。以下内容必须完全一致：

- v3 raw dataset digest 与 prepared split；
- train/dev/test task IDs；
- provenance graph construction identity；
- pair artifact 和每个 task 的 positive/negative candidate 集合；
- candidate 去重与 sample-type precedence；
- encoder、模型初始化、R-GCN 层数和 dropout；
- edge BCE、edge loss weight 与 edge labels；
- structured inference、threshold 和 promotion policy；
- optimizer、学习率、epoch budget、梯度裁剪；
- true graph batch 与尾批归一化语义；
- dev checkpoint selection objective；
- random seeds。

禁止用 main 的 v1/BCE 结果和 feature 的 v2/pairwise 结果直接判断 loss 优劣。

建议至少使用 seeds：

```text
13, 17, 29
```

优先观察：

1. Dev Full Support@5；
2. Dev MRR；
3. Dev Edge F1@10；
4. Test Full Support@5 / MRR / Path@10 / Edge F1@10；
5. 收敛 epoch、loss variance 和 seed variance；
6. positive-negative margin 分布；
7. candidate probability calibration（仅 BCE/hybrid 有直接解释价值）。

loss 版本选择必须基于预先声明的 dev 规则和多 seed 稳定性，不得因为单个 test cell 更高而事后切换协议。

## 8. 论文与报告约束

只要正式结果使用 v2，方法描述必须写成：

```text
Task-balanced pairwise logistic candidate loss
+ positive-reweighted dependency-edge BCE
```

不得继续写成 candidate BCE。当前 `provenance_full` runtime 使用真实的 device-local graph batch `8`；每个 DataLoader batch 都执行一次 optimizer step。报告必须列出：

```text
per-device graph batch size
actual tasks per optimizer step（含短尾批）
```

pairwise loss 目前是训练协议，不是已经通过独立消融验证的论文核心创新。论文核心仍应集中在 typed execution-provenance representation、relation-aware message passing、ToolOutput ranking 与 dependency recovery。只有完成严格的 v2/v3 loss 对照后，才能单独主张某种 candidate loss 的贡献。

## 9. 当前执行顺序

1. **已完成**：保留并标记 `provenance-candidate-loss-v2`；
2. **已完成**：修复梯度累积尾组归一化并补充短尾组测试；
3. **已完成**：实施真正多图 batch 并证明其保持 v2 task-local comparison 语义；
4. 完成 checkpoint schema v4 下 v2 pairwise 的正式训练和多 seed 结果；
5. 根据 v2 的效果、稳定性和 calibration 再决定是否启用 v3 weighted BCE；
6. 只有 v2/v3 各有明显互补优缺点时才考虑 v4 hybrid；
7. v5 listwise 暂不进入实现计划。

在完成第 4 步之前，默认 loss 不再变化。