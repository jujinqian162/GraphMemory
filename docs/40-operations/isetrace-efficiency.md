# ISETrace E11 controlled efficiency benchmark

E11 measures online retrieval efficiency without retraining. It reuses the frozen
seed-13 checkpoints and the exact 2,000-query formal test artifact for five methods:

1. Flat Dense-FT;
2. Provenance-Unit Dense-FT;
3. Provenance-Unit Cross-Encoder-FT;
4. exact seed passthrough (`wo_graph`);
5. residual R-GCN (`full_rgcn`).

The benchmark loads each model once, runs 20 warm-up queries, then performs three
complete sequential passes over the test set. The harness explicitly enables the
same high/TF32 CUDA matmul mode used by the formal retrieval runs, and every
measured query is bounded by CUDA synchronization. The first measured pass must
reproduce the formal top-10 ranking exactly or the run fails. Results include
mean/P50/P95 latency, repeated
full-test throughput, peak allocated VRAM, setup time, model size, hardware, code
commit, and input/model/prediction artifact digests.

## Preconditions

Run on one fixed, otherwise idle GPU. Use one process at a time; concurrent E11
processes or unrelated GPU workloads invalidate the comparison. The repository
must be at the committed E11 implementation with no tracked modifications. The
five formal run directories and every processed artifact referenced by their
`assets/manifest.yaml` files must still exist.

On SenseCore:

```bash
cd /mnt/afs/zhengmingkai/jjq/GraphMemory
set +u
source cloud_setup/setup.bash
mkdir -p results/isetrace/e11/measurements
```

The commands below assume the formal outputs remain under `runs/`. If a delivered
copy is used instead, replace only the `--run` path; its manifest must still resolve
the original processed artifacts.

## Run the five measurements

```bash
common=(
  --device cuda:0
  --warmup-queries 20
  --repeats 3
  --expected-task-count 2000
)

python scripts/benchmark_isetrace_efficiency.py \
  --run runs/isetrace_v7_dense_ft_flat_s13_evalv8 \
  --label flat_dense_ft \
  --output results/isetrace/e11/measurements/flat_dense_ft.json \
  "${common[@]}"

python scripts/benchmark_isetrace_efficiency.py \
  --run runs/isetrace_v7_dense_ft_pu_s13_evalv8 \
  --label pu_dense_ft \
  --output results/isetrace/e11/measurements/pu_dense_ft.json \
  "${common[@]}"

python scripts/benchmark_isetrace_efficiency.py \
  --run runs/isetrace_v7_cross_encoder_provenance_unit_s13_evalv8 \
  --label pu_cross_encoder_ft \
  --output results/isetrace/e11/measurements/pu_cross_encoder_ft.json \
  "${common[@]}"

python scripts/benchmark_isetrace_efficiency.py \
  --run runs/isetrace_v7_pu_dense_ft_rgcn_wo_graph_s13_evalv8 \
  --label exact_seed \
  --output results/isetrace/e11/measurements/exact_seed.json \
  "${common[@]}"

python scripts/benchmark_isetrace_efficiency.py \
  --run runs/isetrace_v7_pu_dense_ft_rgcn_full_s13_evalv8 \
  --label residual_rgcn \
  --output results/isetrace/e11/measurements/residual_rgcn.json \
  "${common[@]}"
```

Do not parallelize these commands. A failed exact-ranking check means the current
checkout cannot be used to benchmark that formal checkpoint; do not weaken or
remove the check.

## Aggregate and render the paper-ready table

```bash
inputs=(
  --input flat_dense_ft=results/isetrace/e11/measurements/flat_dense_ft.json
  --input pu_dense_ft=results/isetrace/e11/measurements/pu_dense_ft.json
  --input pu_cross_encoder_ft=results/isetrace/e11/measurements/pu_cross_encoder_ft.json
  --input exact_seed=results/isetrace/e11/measurements/exact_seed.json
  --input residual_rgcn=results/isetrace/e11/measurements/residual_rgcn.json
)

python scripts/report_isetrace_efficiency.py \
  "${inputs[@]}" \
  --output-json results/isetrace/e11/efficiency.json \
  --output-csv results/isetrace/e11/efficiency.csv \
  --output-tex results/isetrace/e11/efficiency-table.tex
```

The aggregator rejects mixed test artifacts, devices, hardware, commits, warm-up
counts, repeat counts, dirty checkouts, non-seed-13 checkpoints, or incomplete
ranking validation. Preserve the five measurement JSON files together with the
aggregate outputs and a SHA-256 manifest:

```bash
find results/isetrace/e11 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > results/isetrace/e11/SHA256SUMS
```
