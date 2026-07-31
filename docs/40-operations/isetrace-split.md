# ISETrace provisioning and fixed split

The RQ2 benchmark uses the complete revision-pinned ISETrace release, not the 32-trajectory sample. The sample under `data/isetrace/sample/` remains only for fast smoke tests and query-authoring inspection.

## Download

The registered source revision is:

```text
e40e04d41c04e4eb4bae181ebdd41b61c688081b
```

Download all eight trajectory shards and the intent corpus directly through the configured Hugging Face mirror:

```bash
uv run python scripts/prepare_dataset.py \
  --dataset isetrace \
  --name isetrace \
  --mirror
```

The complete local corpus contains 23,132 trajectories in eight shards plus 43,955 intent-corpus records. `prepare_dataset.py` verifies every registered file size. `--mirror` selects `hf-mirror.com` before the first request instead of waiting for an official-endpoint failure.

## Split policy

Build the fixed split with:

```bash
uv run python scripts/build_isetrace_split.py
```

Outputs are written under `data/isetrace/splits/v1/`:

- `manifest.json`: source revision, per-shard SHA-256 digests, policy, counts, and assignment-file digests;
- `train.jsonl`, `dev.jsonl`, and `test.jsonl`: stable trajectory IDs, source locations, intent IDs, and leakage-group IDs.

The policy is `isetrace-intent-components-v1`, seed 13, with an 80/10/10 ratio. The unit of assignment is not an individual trajectory or source shard. It is a connected component formed by:

1. trajectories sharing any `source_intent_id`; and
2. trajectories containing exactly equal intent text after case-folding and whitespace normalization.

Whole components are assigned together. Components are processed by descending size with seeded deterministic deficit balancing, yielding exact trajectory counts while preventing the same source task from crossing splits. This matters because 2,002 source intents occur in two trajectories and transitive components contain as many as 109 trajectories.

| Split | Raw trajectories | Intent components | Canonically valid | Dependency-motif eligible |
|---|---:|---:|---:|---:|
| Train | 18,506 | 16,936 | 18,376 | 17,272 |
| Dev | 2,313 | 2,115 | 2,296 | 2,177 |
| Test | 2,313 | 2,115 | 2,296 | 2,153 |
| **Total** | **23,132** | **21,166** | **22,968** | **21,602** |

The current canonical adapter rejects 164 records with incomplete tool-call/output pairing. A further 1,366 valid trajectories contain no dependency motif beyond `call_result`; they remain assigned for auditability but are excluded from the primary dependency-retrieval query pool. `call_result` is diagnostic and must be reported separately.

The full-data M2 audit extracts 295,744 dependency motifs:

| Motif | Count |
|---|---:|
| `value_flow` | 143,061 |
| `multi_hop_flow` | 76,086 |
| `artifact_lifecycle` | 55,382 |
| `multi_source_join` | 21,215 |

M3 query selection must consume these fixed assignments. Training may use only train trajectories, model/checkpoint selection only dev trajectories, and all template-natural generalization reporting only test trajectories. LLM-authored test queries must be generated from frozen test `QuerySpec` records rather than resampling raw trajectories.
