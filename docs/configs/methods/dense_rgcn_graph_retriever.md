# Dense R-GCN

`configs/method_configs/dense_rgcn_graph_retriever.yaml` owns the dense encoder, pair sampling, typed R-GCN model, optimizer, and model-selection settings. Planning schedules train/dev/test EvidenceGraph artifacts only where the R-GCN workflow needs them.

Training applies BCE to independent evidence-node logits. Inference scores all candidate nodes once and returns the complete ranking. The checkpoint schema is current-only; incompatible older checkpoints must be retrained.
