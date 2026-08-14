from __future__ import annotations

import math
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from graph_memory.contracts.model import DomainModel, NonEmptyStr, PositiveInt
from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.requests import RankingMethodRequest


class LatencySummary(DomainModel):
    mean_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)


class ThroughputSummary(DomainModel):
    mean_queries_per_second: float = Field(gt=0.0)
    sample_std_queries_per_second: float = Field(ge=0.0)
    repeat_queries_per_second: tuple[float, ...] = Field(min_length=1)
    repeat_seconds: tuple[float, ...] = Field(min_length=1)


class DeviceMemorySummary(DomainModel):
    baseline_allocated_bytes: int = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    incremental_peak_bytes: int = Field(ge=0)


class RetrievalEfficiencyResult(DomainModel):
    schema_version: Literal[1] = 1
    device: NonEmptyStr
    task_count: PositiveInt
    top_k: PositiveInt
    warmup_queries: PositiveInt
    repeats: PositiveInt
    latency: LatencySummary
    throughput: ThroughputSummary
    device_memory: DeviceMemorySummary
    ranking_validation: Literal["exact_top_k"]


def benchmark_retrieval(
    *,
    retrieval_method: RetrievalMethod,
    requests: Sequence[RankingMethodRequest],
    expected_ranked_node_ids: Mapping[str, Sequence[str]],
    top_k: int,
    warmup_queries: int,
    repeats: int,
    device: str,
) -> RetrievalEfficiencyResult:
    """Benchmark an already-built retriever under sequential single-query service.

    Model construction and request projection happen before this function. A separate
    unmeasured pass reproduces the frozen formal top-k rankings before warm-up can
    change accelerator kernel selection. CUDA is synchronized around every measured
    query, so latency includes CPU preparation, accelerator execution, and result
    materialization without measuring queued work.
    """
    if not requests:
        raise ValueError("benchmark requires at least one request")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if warmup_queries <= 0:
        raise ValueError("warmup_queries must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    task_ids = [request.task_id for request in requests]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark request task IDs must be unique")
    if set(task_ids) != set(expected_ranked_node_ids):
        raise ValueError(
            "formal rankings and benchmark requests have different task IDs"
        )
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA benchmark requested but CUDA is unavailable: {device}"
        )

    for request in requests:
        result = retrieval_method.rank_task(request, top_k=top_k)
        observed = tuple(node.node_id for node in result.ranked_nodes[:top_k])
        expected = tuple(expected_ranked_node_ids[request.task_id][:top_k])
        if observed != expected:
            raise ValueError(
                f"benchmark changed formal ranking for task_id={request.task_id!r}: "
                f"expected={expected} observed={observed}"
            )
    _synchronize(run_device)

    for request in requests[: min(warmup_queries, len(requests))]:
        retrieval_method.rank_task(request, top_k=top_k)
    _synchronize(run_device)
    baseline_memory = _allocated_memory(run_device)
    _reset_peak_memory(run_device)

    all_latency_ms: list[float] = []
    repeat_seconds: list[float] = []
    for _ in range(repeats):
        _synchronize(run_device)
        repeat_started = time.perf_counter()
        for request in requests:
            _synchronize(run_device)
            started = time.perf_counter()
            retrieval_method.rank_task(request, top_k=top_k)
            _synchronize(run_device)
            all_latency_ms.append((time.perf_counter() - started) * 1000.0)
        _synchronize(run_device)
        repeat_seconds.append(time.perf_counter() - repeat_started)

    peak_memory = _peak_memory(run_device)
    qps = [len(requests) / seconds for seconds in repeat_seconds]
    return RetrievalEfficiencyResult(
        device=str(run_device),
        task_count=len(requests),
        top_k=top_k,
        warmup_queries=min(warmup_queries, len(requests)),
        repeats=repeats,
        latency=LatencySummary(
            mean_ms=statistics.fmean(all_latency_ms),
            p50_ms=float(np.percentile(all_latency_ms, 50)),
            p95_ms=float(np.percentile(all_latency_ms, 95)),
        ),
        throughput=ThroughputSummary(
            mean_queries_per_second=statistics.fmean(qps),
            sample_std_queries_per_second=(
                statistics.stdev(qps) if len(qps) > 1 else 0.0
            ),
            repeat_queries_per_second=tuple(qps),
            repeat_seconds=tuple(repeat_seconds),
        ),
        device_memory=DeviceMemorySummary(
            baseline_allocated_bytes=baseline_memory,
            peak_allocated_bytes=peak_memory,
            incremental_peak_bytes=max(0, peak_memory - baseline_memory),
        ),
        ranking_validation="exact_top_k",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _allocated_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.memory_allocated(device))


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    value = int(torch.cuda.max_memory_allocated(device))
    if not math.isfinite(float(value)):
        raise ValueError("peak device memory must be finite")
    return value


__all__ = [
    "DeviceMemorySummary",
    "LatencySummary",
    "RetrievalEfficiencyResult",
    "ThroughputSummary",
    "benchmark_retrieval",
]
