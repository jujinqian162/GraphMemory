# 实验部分中文草稿

> 状态：第一版论文实验章节草稿，正文已按论文叙事组织。

## 1. 实验设置

### 1.1 研究问题

实验围绕以下三个研究问题展开：

- **RQ1：显式图结构能否提高多跳证据检索的完整性？** 我们考察图增强方法能否在有限的 top-$k$ 检索预算下找回更多支持证据，并提高完整证据集合的覆盖率。
- **RQ2：关系感知的图编码能否进一步改善证据路径恢复？** 我们比较基于启发式图重排、冻结稠密编码器的 R-GCN，以及经过任务内微调的稠密编码器与 R-GCN 组合，分析结构建模与语义表示的互补作用。
- **RQ3：面向执行依赖的 provenance graph 是否比普通文本检索和实体图检索更适合模拟多智能体轨迹？** 我们在由 2WikiMultiHopQA 改造得到的执行轨迹检索任务上，对比稀疏检索、轻量非 LLM GraphRAG 与 Execution-Provenance Graph Memory（EPGM）。

### 1.2 数据集与划分

第一组实验关注通用的 evidence-path retrieval，使用 HotpotQA、2WikiMultiHopQA 和 MuSiQue。三个数据集均保留独立的训练数据，并从带标签的开发集前部划出 500 个样本用于模型选择，其余样本仅用于最终测试。第二组实验使用合成的 2Wiki execution-provenance benchmark。该数据集将可恢复的两条支持证据映射为带类型的工具调用、工具输出和数据流依赖，并在结构相近的非 gold 分支中检索正确的证据输出及其依赖关系。

| 数据集 | 训练样本 | 模型选择样本 | 测试样本 | 主要评测目标 |
|---|---:|---:|---:|---|
| HotpotQA | 90,025 | 500 | 6,869 | 证据集合与图连通性 |
| 2WikiMultiHopQA | 167,454 | 500 | 12,076 | 证据集合与依赖路径 |
| MuSiQue | 19,938 | 500 | 1,917 | 证据集合与多跳路径 |
| Synthetic 2Wiki Provenance | 74,012 | 500 | 2,851 | 模拟执行轨迹中的证据输出与依赖恢复 |

Synthetic 2Wiki Provenance 是一个受控的合成检索任务，而不是真实的多智能体执行日志。它的作用是隔离并检验 execution-level dependency 是否能为检索提供额外信息，而不是替代真实 agent benchmark。

### 1.3 对比方法

在 evidence-path retrieval 实验中，我们比较以下方法：

- **BM25**：基于词项匹配的稀疏检索基线。
- **Dense**：使用预训练稠密编码器计算问题与候选证据的语义相似度。
- **BM25-Graph / Dense-Graph**：以 BM25 或 Dense 为初始分数，在 evidence graph 上进行启发式图重排。
- **Dense R-GCN**：以冻结的 Dense 表示为节点特征，使用关系图卷积网络编码证据图。
- **Dense-FT**：在任务训练数据上微调的稠密检索器。
- **Dense-FT R-GCN**：以 Dense-FT 表示初始化 R-GCN，在语义表示和关系结构上联合建模。该方法是 evidence-path retrieval 部分的主要完整模型。

在模拟 execution-provenance 实验中，当前结果比较三类代表性方法：

- **BM25** 表示不使用稠密语义表示和图结构的稀疏基线。
- **FastGraphRAG-style** 是一个轻量、非 LLM 的实体图检索基线。它先把候选文本切为私有小 text units，以确定性的 noun/structured phrase 抽取构建实体共现图，采用 frequency-scaled positive PMI 和固定剪枝，再结合精确/词法 query linking 与冻结 Dense 实体种子执行 Personalized PageRank，最后把图分数投影并软融合回共享候选证据。该实现忠实复现 FastGraphRAG 的低成本 NLP 共现图检索原则，但不声称复现依赖 LLM 社区摘要和生成阶段的完整 GraphRAG 产品。
- **EPGM** 在类型化 execution-provenance graph 上进行有界路径搜索，将语义相关性、依赖完整性、字段绑定和 grounding 信号组合为路径得分，并返回实际参与检索的 provenance path。

后续将加入可训练的 **EPGM R-GCN**。由于该方法的完整测试结果尚未产生，本稿不填入估计值，也不据此作任何结论。

### 1.4 评测指标

我们使用 Recall@$k$ 衡量前 $k$ 个结果覆盖 gold evidence 的比例，使用 Evidence F1@$k$ 综合衡量检索精度与召回率。Full Support@$k$ 仅在全部 gold evidence 均出现在前 $k$ 个结果时记为 1，因此比普通 Recall 更直接地反映多跳问题是否获得完整支持。MRR 衡量第一个 gold evidence 的排序位置。

对于具有图标签的数据集，我们进一步报告 Connected Evidence Recall 和 Query-Evidence Connectivity。Path Recall@10 衡量检索子图是否包含连接 gold evidence 的有效路径，Edge Recall@10 衡量 gold dependency edge 的直接恢复比例。HotpotQA 不提供有方向的 gold dependency path，因此只报告连通性指标；2WikiMultiHopQA 和 MuSiQue 报告路径与边指标。

## 2. Evidence-Path Retrieval

### 2.1 图编码在三个多跳问答数据集上稳定提高完整证据恢复率

表 1 给出 HotpotQA 的主要结果。Dense-FT R-GCN 在 Recall@5、Recall@10、Full Support@5、Full Support@10 和 MRR 上均取得最佳结果。相较于 Dense-FT，引入关系图编码后 Full Support@5 从 68.55% 提升至 80.08%，Full Support@10 从 86.68% 提升至 93.29%。这表明，即使稠密编码器已经经过任务内微调，显式证据关系仍能帮助模型补全多跳问题所需的支持集合。

**表 1：HotpotQA evidence retrieval 结果（%）**

| 方法 | Recall@5 | Recall@10 | Full Support@5 | Full Support@10 | MRR | Connected Recall@5 | Connected Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 67.88 | 81.27 | 37.50 | 60.04 | 82.70 | 26.32 | 45.65 |
| Dense | 76.98 | 88.86 | 53.20 | 75.38 | 85.30 | 37.73 | 59.44 |
| BM25-Graph | 71.94 | 86.10 | 44.90 | 70.11 | 82.22 | 35.20 | 57.21 |
| Dense-Graph | 77.78 | 89.47 | 55.03 | 77.06 | 85.22 | 40.70 | 62.21 |
| Dense R-GCN | 88.67 | 95.97 | 76.12 | 91.21 | 91.44 | 57.69 | 73.58 |
| Dense-FT | 85.22 | 94.32 | 68.55 | 86.68 | 88.47 | 49.18 | 69.24 |
| **Dense-FT R-GCN** | **90.95** | **97.01** | **80.08** | **93.29** | **92.99** | **59.80** | **75.50** |

2WikiMultiHopQA 上的提升更加明显。如表 2 所示，Dense-FT R-GCN 的 Full Support@5 达到 94.44%，较 Dense-FT 高 10.72 个百分点；Full Support@10 达到 98.28%。在结构指标上，Dense-FT R-GCN 的 Path Recall@10 为 93.70%，Edge Recall@10 为 88.75%，分别比 Dense-Graph 高 35.58 和 31.53 个百分点。Dense R-GCN 同样显著高于未训练的 Dense-Graph，说明主要增益并不只来自稠密编码器微调，而与关系感知的图传播直接相关。

**表 2：2WikiMultiHopQA evidence-path retrieval 结果（%）**

| 方法 | Recall@5 | Recall@10 | Full Support@5 | Full Support@10 | MRR | Path Recall@10 | Edge Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 52.33 | 67.74 | 18.46 | 36.24 | 65.93 | -- | -- |
| Dense | 67.76 | 79.53 | 36.28 | 55.86 | 86.33 | -- | -- |
| BM25-Graph | 55.00 | 73.46 | 22.71 | 47.21 | 62.43 | 38.39 | 38.19 |
| Dense-Graph | 68.07 | 81.53 | 37.61 | 60.46 | 87.06 | 58.13 | 57.22 |
| Dense R-GCN | 94.50 | 97.92 | 89.58 | 96.04 | 96.09 | 90.75 | 86.68 |
| Dense-FT | 92.60 | 98.20 | 83.71 | 96.48 | 93.62 | -- | -- |
| **Dense-FT R-GCN** | **97.07** | **99.10** | **94.44** | **98.28** | **97.78** | **93.70** | **88.75** |

MuSiQue 包含更复杂的组合问题，其结果呈现相同趋势。Dense-FT R-GCN 在 Recall@5、Recall@10、Full Support@5 和 Full Support@10 上均取得最佳表现，其中 Full Support@5 从 Dense-FT 的 58.42% 提升至 63.43%。Path Recall@10 达到 81.32%，比 Dense-Graph 高 13.93 个百分点。Dense-FT 的 MRR 比 Dense-FT R-GCN 高 0.12 个百分点，说明图编码的主要收益仍然体现在完整证据集合与路径恢复，而不是把第一个相关证据尽可能提前。

**表 3：MuSiQue evidence-path retrieval 结果（%）**

| 方法 | Recall@5 | Recall@10 | Full Support@5 | Full Support@10 | MRR | Path Recall@10 | Edge Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 55.20 | 70.18 | 18.83 | 37.92 | 75.44 | -- | -- |
| Dense | 68.66 | 85.07 | 37.09 | 64.21 | 85.78 | -- | -- |
| BM25-Graph | 60.48 | 77.09 | 29.32 | 52.95 | 75.93 | 51.02 | 53.41 |
| Dense-Graph | 71.18 | 87.20 | 44.34 | 69.54 | 84.29 | 67.40 | 68.80 |
| Dense R-GCN | 79.17 | 93.81 | 56.29 | 83.93 | 86.73 | 79.97 | 78.95 |
| Dense-FT | 80.79 | 93.75 | 58.42 | 83.36 | **91.48** | -- | -- |
| **Dense-FT R-GCN** | **84.33** | **94.85** | **63.43** | **86.28** | 91.36 | **81.32** | **79.45** |

### 2.2 结构增益主要体现在完整支持和路径恢复

三个数据集共同显示，单纯提高语义相似度并不能稳定保证多跳证据的完整性。Dense-FT 在普通 Recall 上已经较强，但将其作为 R-GCN 的语义初始化后，Full Support 和路径指标仍进一步提高。该现象在 2WikiMultiHopQA 上最显著：Dense-FT R-GCN 的 Recall@10 与 Dense-FT 只相差不到 1 个百分点，而 Full Support@5 提高超过 10 个百分点。这说明图编码并非简单找回更多孤立相关句子，而是更倾向于将属于同一推理链的证据共同提升到有限检索预算内。

启发式图重排也能改善部分指标，但提升幅度和跨数据集稳定性弱于 R-GCN。BM25-Graph 在 2WikiMultiHopQA 上降低了 Recall@2 和 MRR，而 Dense-Graph 在三个数据集上的 Full Support 和路径指标通常有所改善。这一差异说明图结构本身具有价值，但固定规则难以同时平衡语义相关性、关系类型和不同问题结构；关系感知模型能够从训练数据中学习更合适的传播方式。

### 2.3 消融实验表明 bridge edge 对路径恢复最为关键

为进一步分析关系图中各结构的作用，我们在 2WikiMultiHopQA 的同一测试集（12,076 个样本）上对 Dense R-GCN 和 Dense-FT R-GCN 分别进行消融。表 4 报告完整模型及四个核心结构变体。该消融实验使用独立训练运行，因此数值只在表内与各自的 `full R-GCN` 比较，不与表 2 的主实验数值直接作差。

**表 4：2WikiMultiHopQA R-GCN 核心结构消融（%）**

| 语义编码器 | 图结构变体 | Recall@5 | Full Support@5 | Connected Recall@10 | Path Recall@10 |
|---|---|---:|---:|---:|---:|
| Dense | full R-GCN | **94.13** | **88.79** | **70.41** | **90.39** |
| Dense | w/o bridge | 93.02 | 86.95 | 69.96 | 71.82 |
| Dense | w/o entity overlap | 93.89 | 88.49 | 69.48 | 86.62 |
| Dense | w/o graph | 88.66 | 78.71 | 68.69 | 82.81 |
| Dense | w/o edge type | 92.44 | 86.20 | 69.39 | 88.34 |
| Dense-FT | full R-GCN | **97.29** | **94.77** | 73.18 | **94.33** |
| Dense-FT | w/o bridge | 96.84 | 94.00 | 73.32 | 74.79 |
| Dense-FT | w/o entity overlap | 96.77 | 93.89 | 72.35 | 89.45 |
| Dense-FT | w/o graph | 95.24 | 90.88 | **73.71** | 92.59 |
| Dense-FT | w/o edge type | 96.08 | 92.82 | 72.37 | 92.66 |

最显著且跨编码器一致的现象来自 bridge edge。去掉 bridge 后，Dense R-GCN 和 Dense-FT R-GCN 的 Path Recall@10 分别下降 18.57 和 19.55 个百分点，而 Recall@5 仅下降 1.12 和 0.45 个百分点。这种差异表明，bridge edge 的主要作用不是增加孤立相关证据的数量，而是把分散的支持证据连接成可恢复的多跳路径。去掉 entity-overlap edge 也使 Path Recall@10 分别下降 3.77 和 4.88 个百分点，进一步说明跨证据实体连接对路径构造有直接作用。

完全移除图传播对 Dense R-GCN 的影响最大：Full Support@5 从 88.79% 降至 78.71%，Path Recall@10 从 90.39% 降至 82.81%。在 Dense-FT 初始化下，相同消融的降幅缩小至 3.89 和 1.74 个百分点，说明任务内微调的语义表示能够补偿一部分结构缺失，但仍不能完全替代图结构对完整支持集合的建模。去掉关系类型后，两个模型的 Full Support@5 分别下降 2.58 和 1.95 个百分点，也支持关系感知传播优于将所有边视为同一种连接。

并非所有辅助设计都产生一致增益。完整消融结果中，去掉 edge weight 或 hard negatives 在冻结 Dense 编码器上反而改善了部分指标，而在 Dense-FT 编码器上仅造成小幅下降；query-overlap edge 的影响也随编码器而变化。因此，我们只把 bridge、整体图传播、实体重合边和关系类型作为当前数据稳定支持的结构贡献，并将其余模块视为需要进一步调参与多随机种子验证的实现选择。

## 3. 模拟 Agent Trajectory Retrieval

### 3.1 Execution-level provenance 比实体关系传播更适合恢复执行依赖

表 5 聚焦三种具有代表性的检索范式：词项匹配、轻量实体图传播和 execution-provenance path retrieval。EPGM 在所有列出的证据排序指标上均优于 BM25 和 FastGraphRAG-style baseline。与 FastGraphRAG-style 相比，EPGM 的 Recall@5 从 58.00% 提升至 63.87%，Full Support@5 从 19.89% 提升至 32.83%，相对提升约 65.1%；Full Support@10 从 50.30% 提升至 57.21%。MRR 同样提高 6.05 个百分点。

**表 5：Synthetic 2Wiki execution-provenance retrieval 结果**

| 方法 | Recall@2 | Recall@5 | Recall@10 | Full Support@5 | Full Support@10 | MRR | Path Recall@10 | Edge Recall@10 | 延迟（ms/query） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 28.32 | 44.05 | 58.59 | 7.33 | 23.68 | 55.16 | -- | -- | **0.60** |
| FastGraphRAG-style (non-LLM) | 40.27 | 58.00 | 74.92 | 19.89 | 50.30 | 71.86 | -- | -- | 105.53 |
| **EPGM** | **43.21** | **63.87** | **78.52** | **32.83** | **57.21** | **77.91** | **28.83** | **28.73** | 41.10 |

FastGraphRAG-style baseline 通过实体重合和实体关系传播增强语义检索，因此明显优于 BM25；但其图结构描述的是文本中的实体关联，而不是工具输出如何进入后续调用。EPGM 直接沿 `ToolOutput -> ToolCall -> ToolOutput` 的字段绑定与数据流关系扩展，因此更容易同时找回属于同一执行依赖的两个证据输出。这一差异在 Full Support@5 上最明显，说明 provenance 结构的主要作用不是只提高单个相关输出的排名，而是在较小上下文预算内补全支持链。

EPGM 的 Path Recall@10 和 Edge Recall@10 分别为 28.83% 和 28.73%。这说明非训练版本已经能够恢复一部分执行依赖，但绝对覆盖率仍有较大提升空间。该结果也为后续 EPGM R-GCN 提供了明确目标：在保留候选证据召回的同时，提高正确依赖边和完整路径的覆盖率。

### 3.2 EPGM 在当前实现下兼顾了结构收益与检索开销

EPGM 的平均检索延迟为 41.10 ms/query，与 FastGraphRAG-style 的 105.53 ms/query 相比降低约 61.0%。FastGraphRAG-style 需要对实体进行额外编码并迭代执行 Personalized PageRank，而 EPGM 使用有界 beam search，只扩展类型和方向满足 provenance contract 的路径。BM25 仍然是最快的方法，但其 Full Support@5 只有 7.33%，表明极低延迟伴随着明显的完整证据损失。

这里的效率结论仅覆盖 retrieval stage，不包含原始数据生成、图构造或模型训练成本。完整系统成本将在 EPGM R-GCN 结果产生后统一补充。

## 4. 稳健性与当前局限

2WikiMultiHopQA 上的 R-GCN 消融已经把 evidence-path retrieval 的主要结构增益定位到 bridge edge、整体图传播和关系类型，但这些结论不能直接替代 execution-provenance retriever 自身的消融。对于非训练 EPGM，后续最有价值的实验包括去除字段绑定分数、去除 provenance completeness 分数，以及打乱 `feeds` 依赖边；Dense 可作为完全不使用 provenance path 的语义对照。由于当前合成数据不包含真实的 evidence revision，invalidation penalty 的消融应留到带修订事件的 agent trajectory 数据上进行。

本实验仍有三个主要局限。第一，Synthetic 2Wiki Provenance 的执行结构由问答证据链确定性转换得到，并不等价于真实多智能体系统产生的异构日志。第二，当前结果来自一个固定划分，尚未报告多随机种子方差或逐样本 bootstrap 置信区间。第三，现有实验只评估检索和路径恢复，不能直接支持 unsupported-claim reduction 或 downstream impact tracing 的结论。因此，本节结果应解释为对 execution-provenance retrieval 的受控验证，而不是对完整多智能体记忆系统的最终评估。

## 5. 小结

两组实验从互补角度支持显式结构化记忆。传统多跳问答实验表明，R-GCN 在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上稳定提高 Full Support，并在具有依赖标签的数据集上显著改善 Path Recall 和 Edge Recall；消融实验进一步确认 bridge edge 对恢复完整多跳路径尤其关键。模拟 execution-provenance 实验表明，与稀疏检索和非 LLM entity-search GraphRAG 相比，面向执行依赖的路径检索更有利于在有限 top-$k$ 预算内恢复完整证据输出及其依赖关系。后续的 EPGM R-GCN 将检验关系感知训练能否把这一优势扩展到更高的路径覆盖率。
