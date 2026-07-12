# Dense-seeded R-GCN

`configs/method_configs/dense_rgcn_graph_retriever.yaml` owns the dense encoder, pair sampling, R-GCN model, trainer, reporting, and selection settings. Profile interpolation supplies run scale; root seed/device overrides flow into pairs and training. Planning adds graph, pair, train, checkpoint-backed retrieval, and evaluation dependencies.
