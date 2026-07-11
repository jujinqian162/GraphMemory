# Graph-rerank search space

`configs/experiment/search_spaces/graph_rerank.yaml` owns the complete BM25- and dense-seeded graph-rerank candidate grid. Fixed choices remain one-element YAML lists so tuning has one uniform Cartesian-product contract. The generated tune-stage YAML embeds the resolved search space; the stage writes a selected JSON config, candidate JSON table, and adjacent YAML summary.
