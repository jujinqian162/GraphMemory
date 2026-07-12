# Dense-FT-seeded R-GCN

`configs/method_configs/dense_ft_rgcn_graph_retriever.yaml` owns the R-GCN settings. The runtime method registry declares Dense-FT as its hidden training dependency. Planning trains Dense-FT first and binds its model directory as the required seed checkpoint for R-GCN training. The public selected method remains `dense_ft_rgcn_graph_retriever`; hidden dependency artifacts are explicit in typed state.

The R-GCN stage uses the same selection-conditioned beam decoder as `dense_rgcn_graph_retriever`; only the frozen embedding provider differs. Defaults are beam `2`, maximum five unique selections, set-level hypothesis deduplication, and learned `STOP`. The checkpoint records decoder, beam, loss, and optimizer-phase settings. Pre-beam R-GCN checkpoints must be retrained.
