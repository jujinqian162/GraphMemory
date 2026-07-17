# Dense-FT-seeded R-GCN

`configs/method/dense_ft_rgcn_graph_retriever.yaml` contains an explicit canonical Dense-FT seed-training section plus the node-wise R-GCN section. The Prefect Flow calls the Dense-FT Task directly and binds that processed model artifact to R-GCN training; Dense-FT is not a second final result or MLflow child run.

Training uses BCE over independent evidence-node logits; inference returns a complete ranking in one forward pass. The public method ID is `dense_ft_rgcn_graph_retriever`.
