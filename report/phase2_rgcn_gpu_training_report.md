# Phase 2 R-GCN GPU Training Report

## Run

- Run directory: `runs/quick_valid_100/learned/rgcn_gpu_100x100_e5_base_5ep`
- Train split: 100 tasks, 915 train pairs
- Dev split: 100 tasks
- Encoder: `intfloat/e5-base-v2`
- Model: full R-GCN, hidden dim 128, 2 layers, dropout 0.1
- Optimizer: AdamW, learning rate 1e-4, batch size 8, pos_weight enabled
- Device: CUDA, `NVIDIA GeForce RTX 5070 Laptop GPU`
- Runtime: 229.76 seconds

## Curves

![Phase 2 R-GCN GPU training curves](E:/College/AdviserProject/EPGM/graph_memory/report/phase2_rgcn_gpu_training_curves.png)

## Epoch Metrics

| epoch | train_loss | dev_loss | Recall@5 | Full Support@5 | MRR | best_dev_metric |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0048 | 0.4694 | 0.6450 | 0.3400 | 0.6730 | 0.4981 |
| 2 | 0.9279 | 0.3816 | 0.6867 | 0.3800 | 0.6968 | 0.5354 |
| 3 | 0.8995 | 0.3362 | 0.7017 | 0.4200 | 0.7393 | 0.5684 |
| 4 | 0.8825 | 0.3150 | 0.6792 | 0.4000 | 0.7570 | 0.5684 |
| 5 | 0.8649 | 0.3144 | 0.6633 | 0.3600 | 0.7639 | 0.5684 |

## Notes

The best checkpoint was selected at epoch 3. Loss continued decreasing through epoch 5, while the checkpoint selection metric peaked at epoch 3 and then did not improve. This suggests the run is already beginning to trade off ranking support metrics against loss on this small dev split.
