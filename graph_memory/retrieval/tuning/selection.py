from __future__ import annotations

from graph_memory.contracts.metrics import MetricRow


def retrieval_tuning_objective(row: MetricRow) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )


def retrieval_candidate_key(
    row: MetricRow,
) -> tuple[float, float, float, float]:
    return (
        retrieval_tuning_objective(row),
        float(row.get("Full Support@10", 0.0)),
        -float(row.get("Retrieval Latency / Query", 0.0)),
        -float(row.get("Avg Retrieved Edges", 0.0)),
    )
