# Verification Record

## Logical Encoder Calls

The before counts use the pre-change call structure on the same deterministic fixtures used by the new tests. The after counts are asserted by the current tests.

| Path | Fixture | Before | After |
|---|---|---:|---:|
| Flat dense ranking | 2 tasks | 4 calls: query and passages per task | 1 bulk call |
| Default graph features | 2 tasks in one graph batch | 6 calls: one embedding plus query/passages seed calls per task | 1 joint bulk call |
| Three-epoch training | 1 train batch and unchanged dev set | 4 calls: train once and dev once per epoch | 2 calls: train once and dev once |
| Initial-score precompute | 33 tasks | 66 calls through task-local dense query/passage encoding | 2 bounded bulk calls with group size 32 |
| Hard-dense preparation | 2 tasks | 4 calls through task-local dense query/passage encoding | 1 bulk signal call |

These are logical `SentenceEncoder.encode()` call counts, not physical GPU kernel counts. Sentence-Transformers may still split one logical call into multiple internal text mini-batches according to `encoder.batch_size`.

## Timing Policy

Wall-clock timing is informational only. No timing threshold is a correctness gate because hardware, text lengths, transfers, and GPU saturation change the result. Normal public retrieval remains task-oriented so existing per-task `latency_ms` attribution is not replaced with amortized batch latency.

## Workspace Comparison

The completed `lock-5-opti` quick workspace was compared with `rgcn_quick_train`. Both use the same 100-task train/dev/test splits, CUDA training config, five epochs, task-graph batch size 8, and local `intfloat-e5-base-v2` model. The new workspace explicitly preserves encoder text mini-batch size 64.

| Stage | `rgcn_quick_train` | `lock-5-opti` | Speed-up |
|---|---:|---:|---:|
| Dense graph-rerank tuning | 76.83 s | 72.60 s | 1.06x |
| R-GCN training | 218.88 s | 28.29 s | 7.74x |
| Dense retrieval | 24.48 s | 21.21 s | 1.15x |
| Dense graph-rerank retrieval | 24.18 s | 22.19 s | 1.09x |
| R-GCN retrieval | 42.72 s | 23.87 s | 1.79x |
| End-to-end workspace wall clock | 466 s | 257 s | 1.81x |

The five training epochs preserve all reported dev ranking metrics exactly. Train and dev loss differences are only about `1e-8` to `1e-9`, consistent with changed GPU mini-batch shapes. Final dense graph-rerank `Recall@5` changed from `0.79317` to `0.79117` on the real GPU run while dense and R-GCN metrics remained unchanged; this is recorded as the expected batch-dependent floating-point risk rather than a correctness threshold.

Sentence-Transformers progress bars report physical text mini-batches inside each logical `encode()` call, not training epochs:

- `22/22` or `24/24` corresponds to a bounded tuning group containing up to 32 tasks.
- `5/5` or `6/6` corresponds to one task-graph batch containing up to 8 tasks.
- Repeated `1/1` after training corresponds to task-oriented retrieval/inference. The bars reset once per task because public retrieval still owns per-task latency measurement.

Training constructs both train and dev frozen graph batches before the epoch loop, so later epochs do not re-run SentenceTransformer. For checkpoint graph retrieval, the joint provider reduces each task from separate embedding and seed forwards to one logical `1/1` call, which explains the R-GCN retrieval improvement even though the displayed bar still reads `1/1`.

## Real Encoder Smoke

Executed outside the Windows sandbox on 2026-06-11 with local `models/intfloat-e5-base-v2` on `cuda:0`.

- Embedding dimension: 768.
- Result shapes: `(3, 768)` and `(4, 768)`.
- All embedding values and ranking scores were finite.
- Row norms ranged from `0.9999999765` to `1.0000000717`.
- Ranked results had complete memory-node coverage and satisfied descending score plus node-ID tie-break ordering.
- Smoke elapsed time including model load: 3.786 seconds.
- The current `get_embedding_dimension()` API was used without the previous deprecation warning.

## Final Verification Status

- Strict OpenSpec validation: passed.
- Focused pytest: 105 passed in 6.07 seconds.
- Full pytest: 302 passed in 48.92 seconds.
- Ruff: passed.
- Basedpyright: 0 errors, 0 warnings, 0 notes.
- Real-encoder CUDA smoke: passed.
- `git diff --check`: passed, with only LF-to-CRLF warnings.
