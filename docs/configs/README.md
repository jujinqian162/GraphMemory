# Experiment configuration

Root: `configs/config.yaml`.

| Path | Ownership |
|---|---|
| `config.yaml` | composition, cache refresh, tracking, Hydra output layout |
| `dataset/*.yaml` | sources, capacities, validation |
| `profile/*.yaml` | split counts and trainable scale |
| `method/*.yaml` | one complete final-method contract |
| `stage/*.yaml` | shared stage fragments (e.g. Dense-FT training) |

Each job composes exactly one `method`. R-GCN configs expose one `method.variant` (default `full_rgcn`). List-valued variants are rejected.

```powershell
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=variants
```
