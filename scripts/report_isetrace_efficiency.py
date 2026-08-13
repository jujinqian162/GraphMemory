from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


METHODS = {
    "flat_dense_ft": ("Flat Dense-FT", "dense_ft", "flat"),
    "pu_dense_ft": ("Prov.-Unit Dense-FT", "dense_ft", "provenance_unit"),
    "pu_cross_encoder_ft": (
        "Prov.-Unit Cross-Encoder-FT",
        "cross_encoder",
        "provenance_unit",
    ),
    "exact_seed": ("Exact seed passthrough", "provenance_rgcn", "wo_graph"),
    "residual_rgcn": ("Residual R-GCN", "provenance_rgcn", "full_rgcn"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the five controlled ISETrace E11 benchmark runs."
    )
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = _labeled_paths(args.input)
    if set(paths) != set(METHODS):
        raise ValueError(
            f"E11 requires labels={sorted(METHODS)}, observed={sorted(paths)}"
        )
    results = {label: _read_json(path) for label, path in paths.items()}
    _validate_results(results)
    rows = [_summary_row(label, results[label]) for label in METHODS]
    common = _common_identity(results)

    _write_json(
        args.output_json,
        {
            "schema_version": 1,
            "benchmark": "isetrace_e11_core5",
            "common": common,
            "rows": rows,
            "sources": {label: str(paths[label].resolve()) for label in METHODS},
        },
    )
    _write_csv(args.output_csv, rows)
    _write_tex(args.output_tex, rows, common)
    return 0


def _validate_results(results: Mapping[str, Mapping[str, object]]) -> None:
    first: Mapping[str, object] | None = None
    for label, result in results.items():
        display_name, expected_method, expected_variant = METHODS[label]
        del display_name
        if result.get("schema_version") != 1:
            raise ValueError(f"label={label!r} has unsupported schema version")
        if result.get("label") != label:
            raise ValueError(f"label mismatch inside result={label!r}")
        if (
            result.get("method") != expected_method
            or result.get("variant") != expected_variant
        ):
            raise ValueError(
                f"label={label!r} method identity changed: "
                f"{result.get('method')}:{result.get('variant')}"
            )
        if result.get("seed") != 13:
            raise ValueError(f"label={label!r} must use fixed checkpoint seed 13")
        code = _mapping(result.get("code"), f"{label}.code")
        if code.get("tracked_worktree_dirty") is not False:
            raise ValueError(f"label={label!r} was benchmarked from a dirty checkout")
        benchmark = _mapping(result.get("benchmark"), f"{label}.benchmark")
        if benchmark.get("ranking_validation") != "exact_top_k":
            raise ValueError(f"label={label!r} did not reproduce formal rankings")
        if first is None:
            first = result
            continue
        for field in ("test_artifact", "hardware", "code"):
            if result.get(field) != first.get(field):
                raise ValueError(f"E11 results disagree on common field={field!r}")
        first_benchmark = _mapping(first.get("benchmark"), "first.benchmark")
        for field in ("device", "task_count", "top_k", "warmup_queries", "repeats"):
            if benchmark.get(field) != first_benchmark.get(field):
                raise ValueError(f"E11 results disagree on benchmark field={field!r}")


def _summary_row(label: str, result: Mapping[str, object]) -> dict[str, object]:
    benchmark = _mapping(result["benchmark"], f"{label}.benchmark")
    latency = _mapping(benchmark["latency"], f"{label}.latency")
    throughput = _mapping(benchmark["throughput"], f"{label}.throughput")
    memory = _mapping(benchmark["device_memory"], f"{label}.device_memory")
    deployment_model_bytes = _integer(
        result.get("deployment_model_bytes"), f"{label}.deployment_model_bytes"
    )
    input_seconds = _number(
        result.get("input_load_seconds"), f"{label}.input_load_seconds"
    )
    setup_seconds = _number(
        result.get("method_setup_seconds"), f"{label}.method_setup_seconds"
    )
    return {
        "label": label,
        "method": METHODS[label][0],
        "initialization_seconds": input_seconds + setup_seconds,
        "mean_latency_ms": _number(latency.get("mean_ms"), f"{label}.mean_ms"),
        "p50_latency_ms": _number(latency.get("p50_ms"), f"{label}.p50_ms"),
        "p95_latency_ms": _number(latency.get("p95_ms"), f"{label}.p95_ms"),
        "mean_queries_per_second": _number(
            throughput.get("mean_queries_per_second"), f"{label}.mean_qps"
        ),
        "sample_std_queries_per_second": _number(
            throughput.get("sample_std_queries_per_second"), f"{label}.std_qps"
        ),
        "peak_vram_gib": _integer(
            memory.get("peak_allocated_bytes"), f"{label}.peak_allocated_bytes"
        )
        / (1024**3),
        "incremental_peak_vram_gib": _integer(
            memory.get("incremental_peak_bytes"), f"{label}.incremental_peak_bytes"
        )
        / (1024**3),
        "model_size_mib": deployment_model_bytes / (1024**2),
    }


def _common_identity(results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    first = results[next(iter(METHODS))]
    benchmark = _mapping(first["benchmark"], "benchmark")
    hardware = _mapping(first["hardware"], "hardware")
    code = _mapping(first["code"], "code")
    test = _mapping(first["test_artifact"], "test_artifact")
    return {
        "test_artifact_digest": test["digest"],
        "task_count": benchmark["task_count"],
        "top_k": benchmark["top_k"],
        "warmup_queries": benchmark["warmup_queries"],
        "repeats": benchmark["repeats"],
        "device": benchmark["device"],
        "device_name": hardware.get("device_name"),
        "device_total_memory_bytes": hardware.get("device_total_memory_bytes"),
        "torch": hardware.get("torch"),
        "cuda_runtime": hardware.get("cuda_runtime"),
        "cudnn": hardware.get("cudnn"),
        "float32_matmul_precision": hardware.get("float32_matmul_precision"),
        "cuda_matmul_allow_tf32": hardware.get("cuda_matmul_allow_tf32"),
        "cudnn_allow_tf32": hardware.get("cudnn_allow_tf32"),
        "code_commit": code["commit"],
        "ranking_validation": "exact_top_k",
    }


def _labeled_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"input must be LABEL=JSON, got {value!r}")
        label, raw_path = value.split("=", maxsplit=1)
        if not label or not raw_path:
            raise ValueError(f"input must be LABEL=JSON, got {value!r}")
        if label in result:
            raise ValueError(f"duplicate input label={label!r}")
        result[label] = Path(raw_path)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    common: Mapping[str, object],
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Method & Init. (s) & Mean / P95 (ms) & Queries/s & Peak VRAM (GiB) & Model (MiB) \\",
        r"\hline",
    ]
    for row in rows:
        values = {
            field: _number(row[field], f"tex.{field}")
            for field in (
                "initialization_seconds",
                "mean_latency_ms",
                "p95_latency_ms",
                "mean_queries_per_second",
                "sample_std_queries_per_second",
                "peak_vram_gib",
                "model_size_mib",
            )
        }
        lines.append(
            f"{row['method']} & {values['initialization_seconds']:.2f} & "
            f"{values['mean_latency_ms']:.2f} / {values['p95_latency_ms']:.2f} & "
            f"{values['mean_queries_per_second']:.2f}$\\pm$"
            f"{values['sample_std_queries_per_second']:.2f} & "
            f"{values['peak_vram_gib']:.2f} & "
            f"{values['model_size_mib']:.1f} \\\\"
        )
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Controlled sequential retrieval efficiency on the frozen "
                f"{_integer(common['task_count'], 'common.task_count'):,}-query ISETrace test set. "
                r"Each fixed checkpoint is loaded once, warmed up, and evaluated in "
                f"{_integer(common['repeats'], 'common.repeats')} complete repetitions. Queries/s "
                r"reports mean$\pm$sample standard deviation across repetitions; "
                r"latency is pooled across queries and repetitions.}"
            ),
            r"\label{tab:efficiency}",
            r"\end{table*}",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return _mapping(value, str(path))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _number(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{path} must be finite and non-negative")
    return number


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
