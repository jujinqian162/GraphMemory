# Default HotpotQA experiment

`configs/experiment/config.yaml` composes the default HotpotQA workflow. It selects the approved seven methods (Memory Stream is opt-in because it requires an external artifact), the quick profile, graph defaults, search spaces, and MLflow tracking defaults.

```powershell
uv run python -m graph_memory.experiment.plan name=hotpotqa_quick
uv run python -m graph_memory.experiment.run name=hotpotqa_quick
```

Use Hydra groups and overrides for dataset, profile, methods, seed, device, top-k, stage bounds, cache, ablations, and method-specific values. The fully resolved values are persisted to the named run before execution.
