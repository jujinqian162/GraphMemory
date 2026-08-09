以下是对aaai投稿截止那天的实验状态概括（之前的实验，若不在意可以跳过）：

当时实验最核心的问题是没有任何数据集能同时提供Agent Trajectory, memory search query和相应的label(answer id)。实验分出三个部分RQ1-3。

RQ1是传统问答检索数据集和baseline(不提供Agent Trajectory)，我适配了hotpotqa、2wikimultihop和musique三个数据集，h和bm25, dense, fastgraphrag, rgcn 这些baseline。RQ1的问题就是数据集和Agent没啥关系，唯一作用就是强调图结构携带更多信息，对记忆检索有用。

RQ2是转换2wikimultihop数据集模拟Agent Trajectory，将原本的多跳链条强行改造成了Tool Call，Tool Output链条，这个部分唯一的能力就是在数据集足够大（能训练）的情况下提供“看着像Agent Trajectory”的数据，但是缺点就是毕竟2wikimultihop数据集不是Agent Trajectory，用程序模拟它非常勉强。而且graph的edge和node类型都很少，RGCN也难以发挥出高的水平。

RQ3则是真正的Agent Trajectory。RQ3的目标是尽可能贴合论文，而论文中要求了很多node和edge，比如claim、decision、verification等等，问题是这些结构即使是在不提供query和label的数据集中也几乎没有，原因是claim、decision、verification这些东西在程序看来都是字符串，无法分辨并构造相应的图结构。于是我写了个侵入式的skill，要求Agent使用multiAgent策略并提供一些tool如record_claim， record_decision 并强制要求Agent在完成任务的过程中使用这个skill，这样就有了完全贴合论文表述的数据，这个部分的缺点是数据量太小，都是我自己跑的一些任务，也无法训练RGCN。

---
以下是目前的进度和计划

我现在的想法是，还是必须引入真实的Agent Trajectory，目前我引入了isetrace，这个数据集不提供memory search的query和label，我做法是：

从巨大的trajectory中抽取具有直接召回、关联召回或多事实召回价值的局部信息，交给LLM生成自然语言query和exact gold（gold必须能唯一定位回Tool Call arguments或Tool Output中的原始字符区间）。目前authoring文件累计生成了6609条query；本轮seed-13实验使用的是6597条query的快照，并固定了包含2000条自然查询、1207条trajectory的测试集。query仍属于尚未完成人工审核的工程数据。

新数据集的格式和那些多跳问题检索不一样，后者会提供稳定的sentence大小和sentence_id，于是可以设置Full Support@k、Recall@k等指标，但是新数据集是真实Trajectory，没有天然的sentence_id，而我认为分割数据也是baseline的职责一部分，也可以成为ours method的优势。Dense等flat method直接按照固定chunk切片并计算向量相似度，ours method则基于trajectory构建图结构并按照图中的argument/output content unit划分数据。为公平比较不同粒度的候选，我引入Coverage@Tokens和Full Support@Tokens，即在固定token预算下统计召回内容覆盖了多少exact gold，以及是否覆盖全部gold。目前非训练ours method（Provenance Path）和训练ours method（Provenance R-GCN）均已实现，并已跑出seed-13初步结果；前者通过冻结E5取Top-5锚点后做最多6跳的双向溯源补全，后者使用查询无关的execution-provenance graph进行关系感知消息传播。

尽管如此，现在的这个实验距离最开始论文所设想的还有一定距离，一是没有multiagent，二是图的edge和node没有那么理想，举个例子：当初论文中有depends_on这个边，设想大概是表明两个tool use之间是有依赖或因果的。但是在trajectory中，程序只能看到两个tool use类型，程序无法知道这两个tool use是并列的read file又或是下一个read file的参数依赖上一个read fild的output。总结以下就是depends_on也好、decision也好这些edge和node都是需要llm参与的，如果不像之前RQ3那样做侵入式的skill，这些node也无法直接构建。现在我做的边是这样的：Earlier ToolOutput ──data.feeds──> Later ToolCall，从每个Earlier output文本中抽取URL、绝对路径、文件名、UUID、hash等特殊字符串，如果Later ToolCall的arguments中出现唯一的精确匹配，则构建data.feeds边。除此之外还保留call-return、argument/content ownership、artifact reads/writes和相邻content chunk等可观测关系。R-GCN主模型已经完成并得到初步正向结果，但候选粒度与图传播的贡献仍然混在一起，因此还需要w/o graph、provenance-unit Dense和relation ablation才能确认提升具体来自哪里。

当前seed-13初步结果：

下表数值均为百分比；Coverage和Full Support采用exact source span，并在固定token预算下计算。测试集为2000条自然查询；当前query尚未完成人工审核，因此这些数据只作为阶段性结果。

| 方法 | 视图 / 监督 | Cov@512 | Cov@1024 | Cov@2048 | FS@2048 | R@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Flat / 无训练 | 16.03 | 28.94 | 45.62 | 37.95 | 51.95 | 38.36 |
| Dense | Flat / 无训练 | 14.60 | 26.01 | 41.91 | 34.20 | 48.09 | 35.86 |
| GraphRAG | Flat graph / 无训练 | 15.01 | 28.01 | 45.56 | 37.90 | 50.61 | 37.12 |
| **Provenance Path（非训练ours）** | Provenance / 无训练 | **25.65** | **40.08** | **54.29** | **48.95** | 44.61 | 37.44 |
| Dense-FT | Flat / natural | 26.76 | 42.17 | 61.05 | 52.95 | **66.77** | **50.64** |
| **Provenance R-GCN（训练ours）** | Provenance / natural | **34.87** | **51.00** | **68.62** | **63.35** | 58.62 | 43.68 |
| Dense-FT | Flat / natural+template | 31.54 | 46.63 | 62.45 | 54.40 | **67.74** | **54.71** |
| **Provenance R-GCN（训练ours）** | Provenance / natural+template | **33.63** | **49.39** | **66.87** | **60.70** | 61.75 | 45.45 |
| Provenance R-GCN | Provenance / template-only | 17.40 | 23.89 | 31.88 | 26.80 | 15.18 | 14.11 |

ours method相对匹配基线的百分点变化如下：

| 对比 | ΔCov@512 | ΔCov@1024 | ΔCov@2048 | ΔFS@2048 | ΔR@5 | ΔMRR |
|---|---:|---:|---:|---:|---:|---:|
| Provenance Path − Dense | **+11.05** | **+14.07** | **+12.38** | **+14.75** | -3.48 | +1.58 |
| R-GCN natural − Dense-FT natural | **+8.11** | **+8.83** | **+7.57** | **+10.40** | -8.15 | -6.96 |
| R-GCN mixed − Dense-FT mixed | **+2.09** | **+2.76** | **+4.42** | **+6.30** | -5.99 | -9.26 |

结果说明ours method当前的主要优势是在1024--2048 token预算内补全分散证据。按query类型看，mixed R-GCN相对mixed Dense-FT的优势主要集中在linked recall（Cov@1024 +5.06、FS@2048 +7.84）和multi-fact recall（Cov@1024 +5.46、FS@2048 +11.92）；direct recall上的Cov@1024为-2.40，FS@2048基本持平（+0.16）。Template-only迁移到自然查询的效果较差，加入template也没有稳定改善R-GCN，因此后续仍以自然查询监督和候选匹配消融为重点。