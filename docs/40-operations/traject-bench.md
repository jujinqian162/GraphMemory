# TRAJECT-Bench data and baseline runbook

The repository adapts TRAJECT-Bench as an offline tool-retrieval benchmark. It does not execute tools or evaluate the upstream end-to-end agent pipeline. Candidate text comes only from each domain's public tool catalog; the gold call arguments, outputs, final answer, and trajectory order remain evaluation labels.

## Download a pinned dataset revision

Run from the repository root on the server. This path needs only Bash, `curl`,
and standard `mkdir`/`dirname`; every file is fetched through the verified
`alpha.hf-mirror.com` endpoint.

```bash
REV=fbd4151a4897c4115679e184bf6c427d9d90e955
BASE="https://alpha.hf-mirror.com/datasets/bigboss24/TRAJECT-Bench/resolve/$REV"
OUT=data/traject_bench/raw

download() {
  path="$1"
  mkdir -p "$OUT/$(dirname "$path")"
  curl --fail --location --retry 5 --retry-all-errors \
    "$BASE/$path?download=true" --output "$OUT/$path"
}

domains=(eCommerce Education Email Finance Gaming Mapping Music News_Media Travel Weather)
for domain in "${domains[@]}"; do
  download "parallel/$domain/simple_ver.json"
  download "parallel/$domain/hard_ver.json"
  download "tools/${domain}_tool.json"
done

sequential_domains=(eCommerce Education Finance Gaming Mapping Music News_Media Travel Weather)
for domain in "${sequential_domains[@]}"; do
  download "sequential/$domain/traj_query.json"
done
download "sequential/Travel/simple_ver.json"
```

The expected directory contains `parallel/`, `sequential/`, and `tools/` directly under `data/traject_bench/raw`.

The official GitHub repository is a supported fallback source. Its data lives one level deeper under `public_data/`; the adapter recognizes both layouts and both published schema variants.

```bash
git clone https://github.com/PengfeiHePower/TRAJECT-Bench.git data/traject_bench/raw
git -C data/traject_bench/raw checkout 2723fd890778dbfb6af9e3aa8ee1c22272979468
```

## Operational partitions and filtering

The benchmark does not publish a training split. Repository workflow names are mapped deterministically as follows:

| Workflow split | Official partition | Raw queries | Valid at pinned revision |
| --- | --- | ---: | ---: |
| `train` | parallel/simple | 2,000 | 1,993 |
| `dev` | parallel/hard | 2,000 | 1,993 |
| `test` | sequential | 1,870 | 1,820 |

The pinned Hugging Face revision drops 64 upstream queries that name at least one gold tool absent from that domain's public catalog. Filtering happens before seeded profile sampling. The adapter reports the rejected count and reason; it never reconstructs a missing candidate from gold call fields.

The pinned GitHub revision contains an older public-data variant with smaller valid pools (1,987 train, 1,975 dev, and 1,335 test). Its smoke and quick profiles run unchanged. The configured full capacities target the primary Hugging Face revision; override the requested full counts when intentionally running the GitHub snapshot.

## Fast BM25 baseline

Use `smoke` first to verify the full prepare, retrieve, evaluate, and aggregate path with one task per operational split:

```bash
uv run python experiment/run.py \
  name=traject_bench_bm25_smoke \
  dataset=traject_bench profile=smoke device=cpu \
  'methods=[bm25]'
```

Then run the 100-task-per-split quick profile:

```bash
uv run python experiment/run.py \
  name=traject_bench_bm25_quick \
  dataset=traject_bench profile=quick device=cpu \
  'methods=[bm25]'
```

## Fast Dense baseline

The default Dense method configuration expects `models/intfloat-e5-base-v2`.
Download only the required files from the pinned model revision:

```bash
MODEL_REV=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
MODEL_BASE="https://alpha.hf-mirror.com/intfloat/e5-base-v2/resolve/$MODEL_REV"
MODEL_OUT=models/intfloat-e5-base-v2

model_files=(
  config.json model.safetensors modules.json sentence_bert_config.json
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt
  1_Pooling/config.json
)
for path in "${model_files[@]}"; do
  mkdir -p "$MODEL_OUT/$(dirname "$path")"
  curl --fail --location --retry 5 --retry-all-errors \
    "$MODEL_BASE/$path?download=true" --output "$MODEL_OUT/$path"
done
```

The mirror may redirect the large `model.safetensors` object to Hugging Face's
Xet storage domain. If that storage domain is also blocked, BM25 remains fully
runnable, but Dense requires copying this model directory from another machine
or using another model mirror.

Run a GPU smoke and then the quick baseline:

```bash
uv run python experiment/run.py \
  name=traject_bench_dense_smoke \
  dataset=traject_bench profile=smoke device=cuda \
  'methods=[dense]'

uv run python experiment/run.py \
  name=traject_bench_dense_quick \
  dataset=traject_bench profile=quick device=cuda \
  'methods=[dense]'
```

Use `device=cpu` when CUDA is unavailable. Inspect run state and resolved artifacts with:

```bash
uv run python experiment/status.py name=traject_bench_dense_quick
uv run python experiment/inspect.py kind=datasets
```

The result table's Recall, Evidence F1, Full Support, and MRR names retain the shared repository schema but mean tool-retrieval quality for this adapter. Do not report them as TRAJECT-Bench Exact Match, Usage, Trajectory Satisfaction, or Solution Accuracy.
