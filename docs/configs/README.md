# Experiment configuration

The only active experiment configuration root is `configs/config.yaml`.

| Path | Ownership |
| --- | --- |
| `config.yaml` | root composition, graph/search-space values, tracking, and Hydra runtime layout |
| `dataset/*.yaml` | HotpotQA, 2WikiMultiHopQA, and MuSiQue sources/capacities |
| `profile/*.yaml` | split count policies and trainable scale |
| `method_configs/*.yaml` | all eight method-specific scientific contracts |
| `dataset/hotpotqa-memory-stream.yaml` | complete importance-backed HotpotQA input contract |

Hydra composes plan/run values and closed Pydantic V2 models validate the resolved container. The root `name: ???` is OmegaConf's mandatory-value marker and is legal only there: callers must supply `name=<name>`. Status, inspect, and reset use closed `key=value` command models, so they need neither Hydra nor command YAML files.

Inspect current entries with:

```powershell
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=datasets
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=methods
```
