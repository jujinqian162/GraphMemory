# MuSiQue operations

MuSiQue-Ans uses paragraph-level candidates and decomposition-derived evidence dependency labels. It remains an evidence-retrieval dataset.

```powershell
uv run python experiment/run.py -m `
  name=musique_smoke dataset=musique profile=smoke device=cpu `
  method=bm25,dense,graphrag
```

Preparation keeps answer/support/decomposition labels out of ranking inputs. GraphRAG builds its entity graph internally. An R-GCN job follows the explicit EvidenceGraph, pair, training, checkpoint-backed ranking, and evaluation branch; equal upstream Tasks are reusable across peer jobs.

R-GCN is a node-wise scorer: training uses node logits with BCE and inference produces one complete node ranking. Current checkpoints use the node-wise schema and incompatible older checkpoints must be retrained.
