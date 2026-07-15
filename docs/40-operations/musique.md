# MuSiQue operations

MuSiQue-Ans uses paragraph-level candidates and decomposition-derived evidence dependency labels. It remains an evidence-retrieval dataset.

```powershell
uv run python experiment/plan.py `
  name=musique_smoke dataset=musique profile=smoke device=cpu `
  'methods=[bm25,dense,graphrag]'

uv run python experiment/run.py `
  name=musique_smoke dataset=musique profile=smoke device=cpu `
  'methods=[bm25,dense,graphrag]'
```

Preparation keeps answer/support/decomposition labels out of ranking inputs. GraphRAG builds its entity graph internally. Selecting either R-GCN method schedules the required EvidenceGraph, pair, training, and checkpoint-backed retrieval stages.

R-GCN is a node-wise scorer: training uses node logits with BCE and inference produces one complete node ranking. Current checkpoints use the node-wise schema and incompatible older checkpoints must be retrained.
