# MuSiQue-Ans Data And Run Guide

Status: current dataset-specific runbook.

## Supported Boundary

This repository supports the answerable MuSiQue-Ans train and dev JSONL files as `dataset: "musique"`. Each official paragraph is one retrieval item. Supporting-paragraph labels stay in the label artifact, while decomposition references provide dependency supervision for graph-aware metrics.

MuSiQue-Full answerability and sufficiency evaluation is not supported. Do not point the current experiment config at MuSiQue-Full files.

## Download Official Data

Use the official [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique) repository and its download script:

```powershell
$ProjectRoot = (Resolve-Path .).Path
$MuSiQueRepo = Join-Path (Split-Path $ProjectRoot -Parent) "musique-upstream"
git clone https://github.com/StonyBrookNLP/musique.git $MuSiQueRepo
Push-Location $MuSiQueRepo
bash download_data.sh
Pop-Location
```

Run this block from the graph-memory repository root. The official script downloads the released archive into the upstream repository's `data/` directory. It requires `bash`, `gdown`, and `unzip`; use the manual download link in the official repository when those tools are unavailable.

The official repository warns that MuSiQue dev/test single-hop source questions can overlap with training data from its seed datasets. If those seed datasets are used for pretraining or retrieval, consult the downloaded `data/dev_test_singlehop_questions_v1.0.json` exclusion list before reporting results.

## Place The Raw Files

Copy the official MuSiQue-Ans files into the paths owned by the named experiment config:

```powershell
$RawDir = Join-Path $ProjectRoot "data/musique/raw"
New-Item -ItemType Directory -Force $RawDir
Copy-Item (Join-Path $MuSiQueRepo "data/musique_ans_v1.0_train.jsonl") (Join-Path $RawDir "musique_ans_v1.0_train.jsonl")
Copy-Item (Join-Path $MuSiQueRepo "data/musique_ans_v1.0_dev.jsonl") (Join-Path $RawDir "musique_ans_v1.0_dev.jsonl")
```

Expected layout:

```text
data/musique/raw/
  musique_ans_v1.0_train.jsonl
  musique_ans_v1.0_dev.jsonl
```

The workflow derives both the dev and test experiment splits from the official dev file with non-overlapping configured offsets. Training uses the official train file.

## Run A Smoke Experiment

Initialize a small non-trainable workflow first:

```powershell
uv run python scripts/experiment.py init musique_smoke `
  --config configs/experiments/musique_evidence_retrieval.json `
  --profile smoke `
  --methods bm25,bm25_graph_rerank `
  --force
```

Inspect and run the generated stages:

```powershell
uv run python scripts/experiment.py plan musique_smoke
uv run python scripts/experiment.py run musique_smoke
uv run python scripts/experiment.py status musique_smoke
```

Omit `--methods` during initialization to select the config's complete baseline and trainable method set. Those trainable paths require the same model/device dependencies documented in the general [command runbook](commands.md).

## Prepare A File Directly

Use the direct preparation command only for artifact debugging or a manually managed split:

```powershell
uv run python scripts/prepare_musique.py `
  --input data/musique/raw/musique_ans_v1.0_dev.jsonl `
  --output_input data/musique/processed/dev_memory_tasks.input.json `
  --output_labels data/musique/processed/dev_memory_tasks.labels.json `
  --output_combined data/musique/processed/dev_memory_tasks.json `
  --max_examples 100 `
  --seed 13 `
  --offset 0
```

The ranking input must not contain `answer`, `answer_aliases`, `is_supporting`, decomposition answers, or dependency labels. The optional combined file is for inspection and must not be consumed by retrieval or graph construction.

## Interpret Metrics

MuSiQue evidence recall is paragraph-level. HotpotQA and 2WikiMultiHopQA use sentence-level evidence in this repository. Always state the evidence granularity when comparing datasets; equal Recall@k values do not represent equal retrieval difficulty across these schemas.
