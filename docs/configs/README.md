# Experiment configuration

The only active experiment configuration root is `configs/config.yaml`.

| Path | Ownership |
| --- | --- |
| `config.yaml` | root composition, cache refresh, benchmark, tracking, and deterministic Hydra output layout |
| `dataset/*.yaml` | raw source files, capacities, and dataset validation policy |
| `profile/*.yaml` | split count policies and trainable scale |
| `method/*.yaml` | one complete final-method contract |
| `stage/*.yaml` | genuinely shared scientific stage fragments, currently canonical Dense-FT training |

Each job composes exactly one `method`. R-GCN method configs expose one `method.variant`, default `full_rgcn`; list-valued variants and variants on stateless methods are rejected. The Dense-FT-seeded R-GCN config contains an explicit `method.seed` Dense-FT section and one R-GCN section.

```powershell
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=datasets
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=variants
```
