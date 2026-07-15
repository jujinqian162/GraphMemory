# Dense-FT-seeded R-GCN

`configs/method_configs/dense_ft_rgcn_graph_retriever.yaml` uses the same node-wise R-GCN contract as the dense-seeded method. Registry training dependencies make Dense-FT explicit, and planning binds the resulting model directory as the seed encoder for R-GCN training.

Training uses BCE over independent evidence-node logits; inference returns a complete ranking in one forward pass. The public method ID is `dense_ft_rgcn_graph_retriever`.
