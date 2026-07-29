from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.analysis import analyze_provenance_ablation_rows
from graph_memory.io import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute seed summaries and query-paired provenance ablation intervals."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="full_rgcn")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args()
    rows = read_json(args.input)
    if not isinstance(rows, list):
        raise ValueError("Ablation analysis input must be a JSON list.")
    result = analyze_provenance_ablation_rows(
        rows,
        baseline_variant=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
