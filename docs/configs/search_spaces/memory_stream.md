# Memory Stream search space

`configs/experiment/search_spaces/memory_stream.yaml` owns relevance, recency, importance, and decay candidates. Memory Stream is restricted to HotpotQA and requires the configured external importance artifact with an exactly matching capped split. The generated tune-stage YAML is the complete direct-script contract and writes selected/candidate JSON plus a YAML stage summary.
