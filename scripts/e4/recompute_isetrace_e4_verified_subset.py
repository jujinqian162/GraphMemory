from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_main_results import main as aggregate_main_results  # noqa: E402

AUDIT_DIR = ROOT / "results/isetrace/e4-audit"
AUDITED_TEST_ARTIFACT_DIGEST = (
    "f079b3219d273ea302fafe2427660ca4d0de17842cae1f145117ef4a5709e396"
)
EVALUATED_TEST_ARTIFACT_DIGEST = (
    "ff1760c654192e27aad91bd2732b63c3556ae4036364938f26f818818b4ff72a"
)
CORE_PAYLOAD_DIGESTS = {
    "labels.json": "b50467b6503a7a02659990bb1fb2a312985599baab6f1bf5bf6919fe6fc74285",
    "provenance_graphs.json": "ecbd10f4dcdaed77b31b400a11a331984630217fca0174b0320ca44e62433f2f",
    "tasks.json": "f5a6603c5d7ad794061504bcb6687ccefecd17c7a866c94dda9b62f0bd2f443a",
}
QUERY_METADATA = (
    ROOT / "data/isetrace/query-authoring/isetrace-v7-raw.jsonl.metadata.jsonl"
)


def _run(path: str) -> Path:
    return ROOT / "results/isetrace/evalv8/runs" / path


MODEL_SEEDS = (13, 17, 29, 37, 41)


def _flat_dense_ft_run(seed: int) -> Path:
    if seed == 37:
        return _run("dense-ft/flat/isetrace_v7_main_dft_flat_s37_recovery")
    return _run(f"dense-ft/flat/isetrace_v7_dense_ft_flat_s{seed}_evalv8")


def _rgcn_run(kind: str, seed: int) -> Path:
    if seed in (13, 17, 29):
        suffix = "full" if kind == "residual" else "wo_graph"
        return _run(
            f"rgcn/{kind}/isetrace_v7_pu_dense_ft_rgcn_{suffix}_s{seed}_evalv8"
        )
    directory = "full_rgcn" if kind == "residual" else "wo_graph"
    suffix = "full" if kind == "residual" else "nograph"
    return _run(f"rgcn/{directory}/isetrace_v7_e2rel_rgcn_{suffix}_s{seed}")


MAIN_RUNS: dict[str, tuple[Path, ...]] = {
    "bm25": (_run("deterministic/isetrace_v7_bm25_evalv8"),),
    "dense_flat": (_run("deterministic/isetrace_v7_dense_flat_evalv8"),),
    "provenance_unit_dense": (_run("deterministic/isetrace_v7_dense_pu_evalv8"),),
    "graphrag": (_run("deterministic/isetrace_v7_graphrag_evalv8"),),
    "provenance_path": (_run("deterministic/isetrace_v7_provenance_path_evalv8"),),
    "dense_ft_flat": tuple(_flat_dense_ft_run(seed) for seed in MODEL_SEEDS),
    "cross_encoder_flat": tuple(
        _run(f"cross-encoder-ft/flat/isetrace_v7_cross_encoder_flat_s{seed}_evalv8")
        for seed in MODEL_SEEDS
    ),
    "dense_ft_provenance_unit": tuple(
        _run(f"dense-ft/provenance-unit/isetrace_v7_dense_ft_pu_s{seed}_evalv8")
        for seed in MODEL_SEEDS
    ),
    "cross_encoder_provenance_unit": tuple(
        _run(
            f"cross-encoder-ft/provenance-unit/"
            f"isetrace_v7_cross_encoder_provenance_unit_s{seed}_evalv8"
        )
        for seed in MODEL_SEEDS
    ),
    "residual_rgcn": tuple(
        _rgcn_run("residual", seed) for seed in MODEL_SEEDS
    ),
}

RGCN_PASSTHROUGH_RUNS: dict[str, tuple[Path, ...]] = {
    "pu_seed": tuple(
        _rgcn_run("seed-passthrough", seed) for seed in MODEL_SEEDS
    ),
    "residual_rgcn": MAIN_RUNS["residual_rgcn"],
}

TRAINABLE_MAIN = {
    "dense_ft_flat",
    "cross_encoder_flat",
    "dense_ft_provenance_unit",
    "cross_encoder_provenance_unit",
    "residual_rgcn",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-aggregate the frozen ISETrace main comparison on an adjudicated "
            "E4 verified subset using existing formal per-task metrics."
        )
    )
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_dir = args.audit_dir.resolve()
    summary = _read_mapping(audit_dir / "adjudication_summary.json")
    if (
        summary.get("reviewer_kind")
        != "independent model-agent review; not human annotation"
    ):
        raise ValueError("unexpected reviewer provenance in adjudication summary")
    if summary.get("meets_minimum_verified_target") is not True:
        raise ValueError("adjudicated E4 subset does not meet the configured minimum")
    task_ids_path = audit_dir / "verified_task_ids.json"
    task_ids = _read_task_ids(task_ids_path)
    if len(task_ids) != summary.get("verified_task_count"):
        raise ValueError("verified task-ID count disagrees with adjudication summary")

    _aggregate(
        runs=MAIN_RUNS,
        trainable=TRAINABLE_MAIN,
        baseline="provenance_unit_dense",
        task_ids_path=task_ids_path,
        task_count=len(task_ids),
        output=audit_dir / "verified_subset_main_table.json",
        output_csv=audit_dir / "verified_subset_main_table.csv",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _aggregate(
        runs=RGCN_PASSTHROUGH_RUNS,
        trainable={"pu_seed", "residual_rgcn"},
        baseline="pu_seed",
        task_ids_path=task_ids_path,
        task_count=len(task_ids),
        output=audit_dir / "verified_subset_rgcn_vs_passthrough.json",
        output_csv=audit_dir / "verified_subset_rgcn_vs_passthrough.csv",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_json(
        audit_dir / "verified_subset_rerun_manifest.json",
        {
            "schema_version": 1,
            "purpose": "offline re-aggregation from existing formal ISETrace per-task metrics",
            "audited_test_artifact_digest": AUDITED_TEST_ARTIFACT_DIGEST,
            "evaluated_test_artifact_digest": EVALUATED_TEST_ARTIFACT_DIGEST,
            "core_payload_digests_equal": CORE_PAYLOAD_DIGESTS,
            "verified_task_count": len(task_ids),
            "verified_task_ids_sha256": _sha256(task_ids_path),
            "adjudication_summary_sha256": _sha256(
                audit_dir / "adjudication_summary.json"
            ),
            "query_metadata_sha256": _sha256(QUERY_METADATA),
            "bootstrap": {
                "samples": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
                "cluster_unit": "trajectory",
            },
            "outputs": [
                "verified_subset_main_table.json",
                "verified_subset_main_table.csv",
                "verified_subset_rgcn_vs_passthrough.json",
                "verified_subset_rgcn_vs_passthrough.csv",
            ],
            "runs": {
                label: [str(path.relative_to(ROOT)) for path in paths]
                for label, paths in {
                    **MAIN_RUNS,
                    "pu_seed": RGCN_PASSTHROUGH_RUNS["pu_seed"],
                }.items()
            },
        },
    )
    return 0


def _aggregate(
    *,
    runs: dict[str, tuple[Path, ...]],
    trainable: set[str],
    baseline: str,
    task_ids_path: Path,
    task_count: int,
    output: Path,
    output_csv: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    argv: list[str] = [
        "--baseline",
        baseline,
        "--task-ids",
        str(task_ids_path),
        "--expected-task-count",
        str(task_count),
        "--query-metadata",
        str(QUERY_METADATA),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--bootstrap-seed",
        str(bootstrap_seed),
        "--output",
        str(output),
        "--output-csv",
        str(output_csv),
    ]
    for label, paths in runs.items():
        for path in paths:
            argv.extend(("--run", f"{label}={path}"))
    for label in sorted(trainable):
        argv.extend(("--trainable", label))
    if aggregate_main_results(argv) != 0:
        raise RuntimeError("main-result aggregation returned a nonzero status")
    result = _read_mapping(output)
    if result.get("test_artifact_digest") != EVALUATED_TEST_ARTIFACT_DIGEST:
        raise ValueError(f"re-aggregation returned wrong test artifact: {output}")
    subset = result.get("task_subset")
    if not isinstance(subset, dict) or subset.get("task_count") != task_count:
        raise ValueError(f"re-aggregation did not preserve selected subset: {output}")


def _read_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], value)


def _read_task_ids(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected nonempty task-ID JSON array: {path}")
    task_ids = {item for item in value if isinstance(item, str) and item}
    if len(task_ids) != len(value):
        raise ValueError(f"task-ID JSON has invalid or duplicate entries: {path}")
    return task_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
