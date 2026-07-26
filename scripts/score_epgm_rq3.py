"""Score the RQ3 real-multi-agent-trace run produced by run_epgm_provenance.py.

Reads one JSONL per method from a run directory and reports the metrics used in
the paper's real-trace table: Recall@5/@10, complete-support counts (FS@5/@10),
MRR, emitted typed-edge counts, and a per-level Recall@10 breakdown.

Usage
-----
    uv run python -m scripts.score_epgm_rq3 --run-dir runs/epgm_rq3/task2
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEVELS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class QueryScore:
    qid: str
    level: str
    recall_at_5: float
    recall_at_10: float
    complete_at_5: bool
    complete_at_10: bool
    reciprocal_rank: float
    emitted_typed_edges: int


def _ranked_ids(row: dict[str, Any]) -> list[str]:
    return [item["node_id"] for item in row["ranked_evidence"]]


def score_row(row: dict[str, Any]) -> QueryScore:
    gold = list(dict.fromkeys(row["reference_gold_evidence_node_ids"]))
    ranked = _ranked_ids(row)
    gold_set = set(gold)

    def recall(k: int) -> float:
        if not gold_set:
            return 0.0
        return len(gold_set & set(ranked[:k])) / len(gold_set)

    reciprocal_rank = 0.0
    for index, node_id in enumerate(ranked, start=1):
        if node_id in gold_set:
            reciprocal_rank = 1.0 / index
            break

    return QueryScore(
        qid=row["qid"],
        level=row.get("level") or "unknown",
        recall_at_5=recall(5),
        recall_at_10=recall(10),
        complete_at_5=bool(gold_set) and gold_set.issubset(set(ranked[:5])),
        complete_at_10=bool(gold_set) and gold_set.issubset(set(ranked[:10])),
        reciprocal_rank=reciprocal_rank,
        emitted_typed_edges=len(row.get("retrieved_edges") or []),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_method(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    scores = [score_row(row) for row in rows]
    by_level = {
        level: _mean([s.recall_at_10 for s in scores if s.level == level])
        for level in LEVELS
    }
    return {
        "method": path.stem,
        "queries": len(scores),
        "R@5": 100.0 * _mean([s.recall_at_5 for s in scores]),
        "R@10": 100.0 * _mean([s.recall_at_10 for s in scores]),
        "FS@5": sum(s.complete_at_5 for s in scores),
        "FS@10": sum(s.complete_at_10 for s in scores),
        "MRR": 100.0 * _mean([s.reciprocal_rank for s in scores]),
        "typed_edges": sum(1 for s in scores if s.emitted_typed_edges > 0),
        "by_level": {level: 100.0 * value for level, value in by_level.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = sorted(args.run_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no .jsonl files under {args.run_dir}")
    results = [score_method(path) for path in paths]
    total = results[0]["queries"]

    header = (
        f"{'method':<34}{'R@5':>8}{'R@10':>8}{'FS@5':>8}{'FS@10':>8}"
        f"{'MRR':>8}{'edges':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['method']:<34}{row['R@5']:>8.2f}{row['R@10']:>8.2f}"
            f"{str(row['FS@5']) + '/' + str(row['queries']):>8}"
            f"{str(row['FS@10']) + '/' + str(row['queries']):>8}"
            f"{row['MRR']:>8.2f}"
            f"{str(row['typed_edges']) + '/' + str(row['queries']):>8}"
        )
    print()
    level_header = f"{'method':<34}" + "".join(f"{level:>10}" for level in LEVELS)
    print("Recall@10 by level")
    print(level_header)
    print("-" * len(level_header))
    for row in results:
        print(
            f"{row['method']:<34}"
            + "".join(f"{row['by_level'][level]:>10.2f}" for level in LEVELS)
        )
    print(f"\nqueries={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
