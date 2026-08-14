from __future__ import annotations

import json
from pathlib import Path

import torch

from graph_memory.analysis.efficiency import benchmark_retrieval
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.requests import (
    RankingMethodRequest,
    TextCandidate,
    TextRankingRequest,
)
from scripts.benchmark_isetrace_efficiency import _configure_numeric_precision
from scripts.report_isetrace_efficiency import METHODS, main as report_main


class _FixedRetriever:
    name = "fixed"

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        del top_k
        assert isinstance(request, TextRankingRequest)
        return _ranking(request)


class _KernelStateRetriever:
    name = "kernel-state"

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        del top_k
        assert isinstance(request, TextRankingRequest)
        calls = self.calls.get(request.task_id, 0)
        self.calls[request.task_id] = calls + 1
        ranking = _ranking(request)
        if calls == 0:
            return ranking
        return RetrievalMethodResult(ranked_nodes=tuple(reversed(ranking.ranked_nodes)))


def _ranking(request: TextRankingRequest) -> RetrievalMethodResult:
    return RetrievalMethodResult(
        ranked_nodes=tuple(
            RankedNode(node_id=candidate.item_id, score=float(3 - index))
            for index, candidate in enumerate(request.candidates)
        )
    )


def _requests() -> list[TextRankingRequest]:
    return [
        TextRankingRequest(
            task_id=f"q{task_index}",
            query_text="query",
            candidates=tuple(
                TextCandidate(
                    item_id=f"q{task_index}-c{index}", text="text", metadata={}
                )
                for index in range(3)
            ),
        )
        for task_index in range(2)
    ]


def test_configure_numeric_precision_uses_formal_cuda_mode() -> None:
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        _configure_numeric_precision("cuda:0")

        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_benchmark_retrieval_reports_latency_and_throughput() -> None:
    result = benchmark_retrieval(
        retrieval_method=_FixedRetriever(),
        requests=_requests(),
        top_k=2,
        warmup_queries=1,
        repeats=2,
        device="cpu",
    )

    assert result.task_count == 2
    assert result.warmup_queries == 1
    assert result.repeats == 2
    assert result.latency.p95_ms >= 0.0
    assert len(result.throughput.repeat_seconds) == 2
    assert result.throughput.mean_queries_per_second > 0.0
    assert result.device_memory.peak_allocated_bytes == 0


def test_benchmark_does_not_compare_rankings_between_passes() -> None:
    retriever = _KernelStateRetriever()

    benchmark_retrieval(
        retrieval_method=retriever,
        requests=_requests(),
        top_k=2,
        warmup_queries=1,
        repeats=1,
        device="cpu",
    )

    assert retriever.calls == {"q0": 3, "q1": 2}


def _benchmark_result(label: str) -> dict[str, object]:
    _, method, variant = METHODS[label]
    return {
        "schema_version": 1,
        "label": label,
        "method": method,
        "variant": variant,
        "seed": 13,
        "test_artifact": {"kind": "dataset", "digest": "test", "size_bytes": 10},
        "model_artifact": {
            "kind": "model",
            "digest": label,
            "size_bytes": 2 * 1024**2,
        },
        "dependency_model_artifacts": [],
        "deployment_models": [],
        "deployment_model_bytes": 1024**2,
        "formal_predictions": {
            "kind": "predictions",
            "digest": f"pred-{label}",
            "size_bytes": 10,
        },
        "input_load_seconds": 1.0,
        "method_setup_seconds": 2.0,
        "hardware": {
            "device": "cuda:0",
            "device_name": "GPU",
            "device_total_memory_bytes": 16 * 1024**3,
            "torch": "2.0",
            "cuda_runtime": "12.8",
            "cudnn": 9000,
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        },
        "code": {"commit": "abc", "tracked_worktree_dirty": False},
        "benchmark": {
            "schema_version": 1,
            "device": "cuda:0",
            "task_count": 2000,
            "top_k": 10,
            "warmup_queries": 20,
            "repeats": 3,
            "latency": {"mean_ms": 2.0, "p50_ms": 1.5, "p95_ms": 4.0},
            "throughput": {
                "mean_queries_per_second": 500.0,
                "sample_std_queries_per_second": 5.0,
                "repeat_queries_per_second": [495.0, 500.0, 505.0],
                "repeat_seconds": [4.04, 4.0, 3.96],
            },
            "device_memory": {
                "baseline_allocated_bytes": 1024**3,
                "peak_allocated_bytes": 2 * 1024**3,
                "incremental_peak_bytes": 1024**3,
            },
            "ranking_validation": "exact_top_k",
        },
    }


def test_report_aggregates_core_five_across_harness_commits(tmp_path: Path) -> None:
    argv: list[str] = []
    for label in METHODS:
        result = _benchmark_result(label)
        if label == "residual_rgcn":
            result["code"] = {"commit": "def", "tracked_worktree_dirty": False}
            del result["formal_predictions"]
            benchmark = result["benchmark"]
            assert isinstance(benchmark, dict)
            del benchmark["ranking_validation"]
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        argv.extend(("--input", f"{label}={path}"))
    output_json = tmp_path / "efficiency.json"
    output_csv = tmp_path / "efficiency.csv"
    output_tex = tmp_path / "efficiency.tex"
    argv.extend(
        (
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-tex",
            str(output_tex),
        )
    )

    assert report_main(argv) == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["common"]["task_count"] == 2000
    assert len(report["rows"]) == 5
    assert report["rows"][0]["initialization_seconds"] == 3.0
    assert report["rows"][0]["peak_vram_gib"] == 2.0
    assert "Residual R-GCN" in output_tex.read_text(encoding="utf-8")
