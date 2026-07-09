# 7-9 重构计划
下文中 [need feedback] 标记是指这部分需要你的反馈，有可能是我不确定要不要这样执行（我可能有没有考虑到的问题）的计划，也可能是我需要你给出更具体和合理的方案和细节的判断。
这里是main分支。

# 背景
一套数据集，baseline的训练、评估流程由许多顶层scripts配合完成。例如先prepare_dataset，然后build_graph, 然后train_method , 然后evaluate，最后aggrate table。
这涉及许多中间文件和大量可调节的参数。
scripts下experiment是为了方便我快速实验的组织编排脚本，他负责锁定config，锁定workspace和确定各个stage和中间文件的依赖链条。比如dense-ft-rgcn-graph-retriever是依赖dense-ft作为seed scorer的。如果没跑dense-ft，experiment.py要解析到这个依赖并先跑出dense-ft的best checkpoint。使用experiment.py plan也可以看到run时将会跑的所有scripts步骤。

# 问题
尽管experiment.py确实可以剩下我大量敲scripts cli的时间。但是随着dataset、baseline增加和一些需求增加，暴露了许多不足之处。
experiment这边代码职责庞大，再增加下去就要变成屎山代码了。
而且就连最简单的覆写一个seed都很困难，因为各个scripts的config中seed是独立的，我每个文件都要改seed。没法统一改。device字段也是同理。这也让我想到了优化config系统。

#  目标 & 解决方案
让experiment.py专注于不同scripts之间的依赖处理，进一步优化实验体验（cli覆写值方便程度、调参与测试方便程度、结果可视化效果）。**但是不得破坏原有逻辑**。
这次重构的范围和力度可以很大，我说的不得破坏原有逻辑不是说必须保留scripts/s.py --config 。。。那种旧的调用形式。换句话说，而是不破坏原有*职责*，形式无所谓，但是核心职责不得变动。

引入Hydra，Pydantic和MLflow。（注意require python>=3.10，服务器上只有3.10的python） [need feedback]
**重构后我所要求的最终效果是**：
- experiment默认行为一致：python experiment/plan.py name=<name\> ; python experiment/run.py name=<name\> 会产出和现在python scripts/experiment.py run <name\> 结果一致的baseline和效果（但不要求output目录结构一样），举个例子，这样跑话会产出hotpotqa dataset+所有baseline在quick profile下的结果（不包括memory stream）-- 也就是默认config的效果。但是结果形式我都不要求一致（可能是放在MLflow中的）。[need feedback]
- experiment cli 覆写值行为：python experiment/run.py name=<name\> dataset=2wiki profile=cloud-full methods='[bm25,dense,...]' 等价于现在的 python scripts/experiment.py --config configs/experiment/2wiki_evidence_retrieval.json --profile cloud-full --methods bm25,dense,...。
- 相比于原有代码，这次重构将会**自然**新增的能力： python experiment/run.py <name\> device=cuda seed=14 (device和seed都要可以放在顶层config统一所有stage的seed和device)
- 新的output结构：将Hydra的outputdir格式定为：runs/2026-7-9/<name\>中 (2026-7-9替换成真实日期)。我有时会用Hydra的multirun，不要忘了配置这个。
- 独立的scripts的cli变化你来考虑 [need feedback]

## config 重构
- 所有config向yaml迁移。使用Hydra+Pydantic。
- **视具体情况**使用Hydra的instantiate()。避免大量 if something == cfg.something: construct something的代码。也不要滥用。 [need feedback]
- **正确组织config group结构**，不仅避免过深嵌套，也避免无比巨大的上帝配置文件。[need feedback]
- 去除代码中的默认值，那会带来歧义，默认值都放在config中。
- 去除原本那一大堆validate函数，改用Pydantic的model_validate(OmegaConf.to_container(cfg, resolve=True)) 对输入结构进行校验 [need feedback]
- 由于整个config结构都大改了，所以几乎所有代码都受到影响，可能会有所更改，不要偷懒，不许做个config的映射就不管domain层内部的细节了，我希望这次重构能彻底的优化config结构，而不是只优化了个experiment.py 这些上层代码。当然，如果有些domain层的代码设计的非常好，不需要改就别改。 [need feedback]
- 充分而正确利用Hydra的yaml插值，这可以避免大量的样板代码。比如让scripts层级的config的seed继承root config的seed。[need feedback]

## experiment 重构
- 内部不再负责config的validate或者锁定，这部分职责交给Hydra和Pydantic
- 只负责处理不同baseline的scripts的一些依赖，顺序和cache处理。[need feedback]
- 位置改动：将原本在scripts下的experiment挪到experiment目录中，原本的sub command 命令行解析不要了，改成(uv run) python experiment/<command\>.py key=value ...。但是这个experiment目录下脚本的定义还是那种scripts层的职责，很多主逻辑还是在graph_memory的experiment中的。但是现在scripts/的职责就真正的变成唯一的负责每个stage的命令了。

## 新增能力 MLflow
- 引入MLflow管理中间文件、中间指标和一些参数信息，考虑其是否可以替代那些run summary样板代码。[need feedback]
- 规范MLflow的数据目录，我打算放放在runs/<date\>/<name\>/mlflow中。或者就在runs/<date\>/<name\>? [need feedback]


## 代码规范与要求
**这是科研代码，不是生产系统！严禁做一堆无用兼容性代码，不需要兼容性，不需要兼容性！别给我做乱七八糟的fallback。最大化删除无用代码！越干净越好！可读性越高越好！**
必须写好类型注释，不允许随意出现Any类型，最好能让代码不再关注各种dataclass类型转换和验证问题，这些问题都交给Pydantic负责。
最好不要出现一堆可选值类型，就是type A | None这种东西，除非是为了简化实现的必要设计。我不想看到一大堆if判空。

思考问题：引入了MLflow，那么什么可以删除：
- TODO 你来写，必须列全了，这是我对你的测试。
- 
引入了Hydra+Pydantic，什么可以彻底删除，什么可以彻底简化：
- TODO 你来写
- 

## 细节冲突
我在编写此计划时以上层的、抽象的思维思考并计划，注定不会关注许多baseline的细节。你还需要为我指出那些**细节冲突**问题，也就是重构计划很理想，但是落实到实现中发现有细节做不到（如缺少信息，库能力不足等等问题）以便于我调整设计。[need feedback]

# experiment.py 已提供能力
- TODO 你来列举experiment已经提供的功能, 比如experiment可以做到恢复中断的实验（cache能力）。这个列表决定了最终重构后需要验收的能力。可能还会有所更改和删减，你先写。
-
-

# 最后
我还有什么没有考虑到的吗？我还有什么思考不充分的地方吗？你还有什么疏忽的判断吗？