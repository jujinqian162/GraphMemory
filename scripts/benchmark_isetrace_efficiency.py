from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.analysis.efficiency import benchmark_retrieval
from graph_memory.experiment.artifacts import (
    ArtifactRef,
    DatasetArtifactRef,
    ModelArtifactRef,
    artifact_payload_path,
)
from graph_memory.experiment.config import (
    CrossEncoderMethodConfig,
    DenseFinetuneMethodConfig,
    ProvenanceRgcnMethodConfig,
    ResolvedExperimentConfig,
)
from graph_memory.experiment.persistence import read_yaml, read_yaml_model
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.io import read_json, write_json
from graph_memory.stages.retrieve import build_retrieve_stage


ARTIFACTS_ADAPTER = TypeAdapter(list[ArtifactRef])
PROVENANCE_GRAPHS_ADAPTER = TypeAdapter(list[ProvenanceGraph])
SUPPORTED_METHODS = (
    DenseFinetuneMethodConfig,
    CrossEncoderMethodConfig,
    ProvenanceRgcnMethodConfig,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one frozen ISETrace formal run after model load, with CUDA "
            "synchronization, warm-up, repeated timing, and peak-memory measurement."
        )
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-task-count", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.label.strip():
        raise ValueError("label must be non-empty")
    if args.expected_task_count <= 0:
        raise ValueError("expected-task-count must be positive")
    _configure_numeric_precision(args.device)

    run_dir = args.run.resolve()
    config = read_yaml_model(
        run_dir / "config" / "resolved.yaml", ResolvedExperimentConfig
    )
    if config.dataset.name != "isetrace" or config.profile != "full":
        raise ValueError("E11 requires an ISETrace full-profile formal run")
    if not isinstance(config.method, SUPPORTED_METHODS):
        raise ValueError(
            "E11 core benchmark supports Dense-FT, Cross-Encoder-FT, and "
            f"Provenance R-GCN; got {config.method.method!r}"
        )

    assets = _load_assets(run_dir / "assets" / "manifest.yaml")
    test = _one_test_dataset(assets)
    model = _one_method_model(assets, config)
    dependency_models = _dependency_models(assets, model=model)
    deployment_models = _deployment_models(config, model, dependency_models)

    input_started = time.perf_counter()
    task_inputs = cast(list[object], read_json(artifact_payload_path(test, "tasks")))
    provenance_graphs = (
        PROVENANCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(test, "provenance_graphs"))
        )
        if isinstance(config.method, ProvenanceRgcnMethodConfig)
        else []
    )
    input_load_seconds = time.perf_counter() - input_started
    if len(task_inputs) != args.expected_task_count:
        raise ValueError(
            f"formal test task count changed: expected={args.expected_task_count} "
            f"observed={len(task_inputs)}"
        )

    method_started = time.perf_counter()
    retrieval_method, retrieval_provenance, requests = build_retrieve_stage(
        config.method,
        dataset=config.dataset.name,
        task_inputs=task_inputs,
        evidence_graphs=None,
        provenance_graphs=provenance_graphs,
        model=model,
        encoder_source=None,
        device=args.device,
    )
    _synchronize_device(args.device)
    method_setup_seconds = time.perf_counter() - method_started
    if len(requests) != len(task_inputs):
        raise ValueError("retrieval request projection changed the formal task count")

    benchmark = benchmark_retrieval(
        retrieval_method=retrieval_method,
        requests=requests,
        top_k=config.top_k,
        warmup_queries=args.warmup_queries,
        repeats=args.repeats,
        device=args.device,
    )
    output = {
        "schema_version": 1,
        "label": args.label,
        "run": str(run_dir),
        "method": config.method.method,
        "variant": config.variant,
        "seed": config.seed,
        "test_artifact": _artifact_identity(test),
        "model_artifact": _artifact_identity(model),
        "dependency_model_artifacts": [
            _artifact_identity(asset) for asset in dependency_models
        ],
        "deployment_models": deployment_models,
        "deployment_model_bytes": sum(
            cast(int, component["size_bytes"]) for component in deployment_models
        ),
        "input_load_seconds": input_load_seconds,
        "method_setup_seconds": method_setup_seconds,
        "retrieval_provenance": retrieval_provenance,
        "hardware": _hardware(args.device),
        "code": _code_identity(),
        "benchmark": benchmark.model_dump(mode="json"),
    }
    write_json(args.output, output)
    return 0


def _load_assets(path: Path) -> list[ArtifactRef]:
    raw = read_yaml(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"asset manifest must be an object: {path}")
    return ARTIFACTS_ADAPTER.validate_python(raw.get("assets"))


def _one_test_dataset(assets: Sequence[ArtifactRef]) -> DatasetArtifactRef:
    matches = [
        asset
        for asset in assets
        if isinstance(asset, DatasetArtifactRef) and asset.origin.get("split") == "test"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"formal run must reference one test dataset, got {len(matches)}"
        )
    return matches[0]


def _one_method_model(
    assets: Sequence[ArtifactRef], config: ResolvedExperimentConfig
) -> ModelArtifactRef:
    matches = [
        asset
        for asset in assets
        if isinstance(asset, ModelArtifactRef)
        and asset.origin.get("method") == config.method.method
        and asset.origin.get("variant") == config.variant
    ]
    if len(matches) != 1:
        raise ValueError(
            f"formal run must reference one final method model, got {len(matches)}"
        )
    return matches[0]


def _dependency_models(
    assets: Sequence[ArtifactRef], *, model: ModelArtifactRef
) -> list[ModelArtifactRef]:
    seed_digest = model.origin.get("seed_model_digest")
    if seed_digest is None:
        return []
    if not isinstance(seed_digest, str) or not seed_digest:
        raise ValueError("final model seed_model_digest must be a non-empty string")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, ModelArtifactRef) and asset.digest == seed_digest
    ]
    if len(matches) != 1:
        raise ValueError(
            f"formal run must reference one seed model digest={seed_digest}, "
            f"got {len(matches)}"
        )
    return matches


def _deployment_models(
    config: ResolvedExperimentConfig,
    model: ModelArtifactRef,
    dependencies: Sequence[ModelArtifactRef],
) -> list[dict[str, object]]:
    final_role = (
        "checkpoint"
        if isinstance(config.method, ProvenanceRgcnMethodConfig)
        else "model"
    )
    components = [
        ("final", model, final_role),
        *[("seed", item, "model") for item in dependencies],
    ]
    result: list[dict[str, object]] = []
    for component, asset, role in components:
        payloads = [payload for payload in asset.payloads if payload.role == role]
        if len(payloads) != 1:
            raise ValueError(
                f"model digest={asset.digest} must contain one payload role={role!r}"
            )
        payload = payloads[0]
        result.append(
            {
                "component": component,
                "asset_digest": asset.digest,
                "payload_role": role,
                "payload_digest": payload.digest,
                "size_bytes": payload.size_bytes,
            }
        )
    return result


def _artifact_identity(asset: ArtifactRef) -> dict[str, object]:
    return {
        "kind": asset.kind.value,
        "digest": asset.digest,
        "size_bytes": asset.size_bytes,
        "payload_digests": {payload.role: payload.digest for payload in asset.payloads},
    }


def _configure_numeric_precision(device: str) -> None:
    run_device = torch.device(device)
    if run_device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _synchronize_device(device: str) -> None:
    run_device = torch.device(device)
    if run_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.synchronize(run_device)


def _hardware(device: str) -> dict[str, object]:
    run_device = torch.device(device)
    result: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "device": str(run_device),
    }
    if run_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        index = (
            run_device.index
            if run_device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device_index": index,
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return result


def _code_identity() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "tracked_worktree_dirty": dirty}


if __name__ == "__main__":
    raise SystemExit(main())
