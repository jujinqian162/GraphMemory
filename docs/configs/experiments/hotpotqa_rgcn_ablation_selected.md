# hotpotqa_rgcn_ablation_selected.json

对应配置文件：`configs/experiments/hotpotqa_rgcn_ablation_selected.json`

该配置用于服务器上的 R-GCN 消融运行。消融阶段只作用于 `dense_rgcn_graph_retriever`；配置仍保留主实验 method 列表，开启 `enable_ablation`，并列出首张消融表需要的非基线 variants。

`full_rgcn` 不需要手工列出。runner 会自动把它作为 alias row 指向主实验 artifact，不会重复训练。

运行 `--ablations-only` 前，需要先在同一个 named run 中完成普通 `dense_rgcn_graph_retriever` 主实验。消融表必须读取 `full_rgcn` 主指标作为 baseline；如果该文件不存在，runner 会在执行前报错，并给出缺失路径。

初始化并检查计划：

```powershell
python scripts/experiment.py init rgcn_ablation_cloud `
  --config configs/experiments/hotpotqa_rgcn_ablation_selected.json `
  --profile cloud-full

python scripts/experiment.py plan rgcn_ablation_cloud --ablations-only
```

只跑两个 variants：

```powershell
python scripts/experiment.py run rgcn_ablation_cloud `
  --ablations-only `
  --variant wo_bridge `
  --variant wo_hard_negatives
```
