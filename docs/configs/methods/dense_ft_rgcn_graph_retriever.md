# Dense-FT-seeded R-GCN

`configs/experiment/method_configs/dense_ft_rgcn_graph_retriever.yaml` owns the R-GCN settings and declares Dense-FT as a hidden training dependency. Planning trains Dense-FT first and binds its model directory as the seed checkpoint for R-GCN training. The public selected method remains `dense_ft_rgcn_graph_retriever`; hidden dependency artifacts are explicit in typed state.
