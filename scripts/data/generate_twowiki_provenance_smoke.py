from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_memory.datasets.twowiki_provenance import convert_twowiki_source_records
from graph_memory.io import read_json, write_json


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    source_path = root / "tests/fixtures/twowiki_provenance_smoke_source.json"
    output_path = root / "tests/fixtures/twowiki_provenance_smoke.json"
    source = read_json(source_path)
    if not isinstance(source, list):
        raise ValueError(f"Smoke source must be a JSON list: {source_path}")
    converted = convert_twowiki_source_records(
        source,
        candidate_cap=6,
        seed=13,
        strict=True,
    )
    write_json(output_path, converted.records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
