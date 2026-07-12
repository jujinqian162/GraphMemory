# Default experiment and HotpotQA override

`configs/config.yaml` composes the default MuSiQue workflow. It selects the approved seven methods (Memory Stream is opt-in because it requires an external artifact), the full profile, graph defaults, search spaces, and fixed MLflow tracking paths.

```powershell
uv run python experiment/plan.py name=hotpotqa_quick dataset=hotpotqa profile=quick
uv run python experiment/run.py name=hotpotqa_quick dataset=hotpotqa profile=quick
```

Use Hydra groups and overrides for dataset, profile, methods, seed, device, top-k, stage bounds, cache, ablations, and method-specific values. The fully resolved values are persisted to the named run before execution.
