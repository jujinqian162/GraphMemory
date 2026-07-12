# Dense-seeded R-GCN

`configs/method_configs/dense_rgcn_graph_retriever.yaml` owns the dense encoder, pair sampling, R-GCN model, beam decoder, beam-aware loss, optimizer phases, reporting, and selection settings. Profile interpolation supplies run scale; root seed/device overrides flow into pairs and training. Planning adds graph, pair, train, checkpoint-backed retrieval, and evaluation dependencies.

The graph is encoded once per task. A first-hop head starts the evidence sequence; subsequent-hop scoring conditions on the selected set, last node, and model-visible relation frontier. The default beam width is `2`, decoding selects at most five unique evidence nodes, and `STOP` leaves the base R-GCN score to fill remaining ranks. Five selections allow a four-evidence task to recover after one distractor.

This decoder is part of the R-GCN checkpoint contract. Checkpoints created before beam decoding are intentionally incompatible and must be retrained.
