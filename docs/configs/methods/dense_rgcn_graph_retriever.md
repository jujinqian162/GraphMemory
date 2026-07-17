# Dense R-GCN

`configs/method/dense_rgcn_graph_retriever.yaml` owns the dense encoder, pair sampling, typed R-GCN model, optimizer, model selection, and one singular variant. The explicit Prefect Flow builds train/dev/test EvidenceGraph artifacts only for this evidence R-GCN branch.

Training applies BCE to independent evidence-node logits. Inference scores all candidate nodes once and returns the complete ranking. The checkpoint schema is current-only; incompatible older checkpoints must be retrained.
