from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_memory.datasets.twowiki_provenance import (
    ProvenanceGraphConstructionConfig,
    TWOWIKI_PROVENANCE_SCHEMA_VERSION,
    audit_twowiki_source_records,
    convert_twowiki_source_records,
    deterministic_dev_test_partition,
)
from graph_memory.datasets.twowiki_provenance.records import (
    ProvenanceEdgeRecord,
    TwoWikiProvenanceRawRecord,
)
from graph_memory.io import read_json, write_json
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    train_source = Path(args.train_source)
    dev_source = Path(args.dev_source)
    output_dir = Path(args.output_dir)
    graph_config = ProvenanceGraphConstructionConfig(
        strategy=args.edge_scorer,
        successors_per_output=args.successors_per_output,
        hybrid_dense_weight=args.hybrid_dense_weight,
        scorer_identity=args.scorer_identity,
        query_template_version=args.query_template_version,
        semantic_temperature=args.semantic_temperature,
        weight_floor=args.weight_floor,
        branch_policy_version=args.branch_policy_version,
        near_rank_bucket=args.near_rank_bucket,
        mid_rank_bucket=args.mid_rank_bucket,
        tail_rank_bucket=args.tail_rank_bucket,
    )
    dense_ranker = _dense_ranker(args) if args.edge_scorer in {"dense", "hybrid"} else None
    if args.audit_only:
        train_raw = _record_list(train_source)
        train_accepted, train_rejected = audit_twowiki_source_records(
            train_raw,
            candidate_cap=args.candidate_cap,
            seed=args.seed,
            graph_config=graph_config,
            dense_ranker=dense_ranker,
        )
        train_source_count = len(train_raw)
        del train_raw
        dev_raw = _record_list(dev_source)
        dev_accepted, dev_rejected = audit_twowiki_source_records(
            dev_raw,
            candidate_cap=args.candidate_cap,
            seed=args.seed,
            graph_config=graph_config,
            dense_ranker=dense_ranker,
        )
        print(
            json.dumps(
                {
                    "train": {
                        "source": train_source_count,
                        "accepted": train_accepted,
                        "rejected": train_rejected,
                    },
                    "dev_source": {
                        "source": len(dev_raw),
                        "accepted": dev_accepted,
                        "rejected": dev_rejected,
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    train_raw = _record_list(train_source)
    train_conversion = convert_twowiki_source_records(
        train_raw,
        candidate_cap=args.candidate_cap,
        seed=args.seed,
        strict=args.strict,
        graph_config=graph_config,
        dense_ranker=dense_ranker,
    )
    train_source_count = len(train_raw)
    train_count = len(train_conversion.records)
    train_rejected = train_conversion.rejected_reason_counts
    write_json(output_dir / "train.json", train_conversion.records)
    train_statistics = _split_statistics(train_conversion.records)
    del train_raw, train_conversion

    dev_raw = _record_list(dev_source)
    dev_conversion = convert_twowiki_source_records(
        dev_raw,
        candidate_cap=args.candidate_cap,
        seed=args.seed,
        strict=args.strict,
        graph_config=graph_config,
        dense_ranker=dense_ranker,
    )
    dev_records, test_records = deterministic_dev_test_partition(
        dev_conversion.records,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
    )
    dev_source_count = len(dev_raw)
    write_json(output_dir / "dev.json", dev_records)
    write_json(output_dir / "test.json", test_records)

    manifest = {
        "schema_version": TWOWIKI_PROVENANCE_SCHEMA_VERSION,
        "dataset": "twowiki_provenance",
        "source_dataset": "2wiki",
        "synthetic_execution_graph": True,
        "source": {
            "train": {
                "path": train_source.as_posix(),
                "sha256": _sha256(train_source),
                "records": train_source_count,
            },
            "dev": {
                "path": dev_source.as_posix(),
                "sha256": _sha256(dev_source),
                "records": dev_source_count,
            },
        },
        "parameters": {
            "seed": args.seed,
            "candidate_cap": args.candidate_cap,
            "dev_fraction": args.dev_fraction,
            "strict": args.strict,
            "graph_construction": {
                **graph_config.identity(),
                "dense_model": args.dense_model if dense_ranker is not None else None,
                "dense_query_prefix": (
                    args.dense_query_prefix if dense_ranker is not None else None
                ),
                "dense_passage_prefix": (
                    args.dense_passage_prefix if dense_ranker is not None else None
                ),
            },
            "split_policy": "source_train_to_train_source_dev_seeded_dev_test",
        },
        "counts": {
            "train": train_count,
            "dev": len(dev_records),
            "test": len(test_records),
            "train_rejected": train_rejected,
            "dev_source_rejected": dev_conversion.rejected_reason_counts,
        },
    }
    statistics = {
        "schema_version": TWOWIKI_PROVENANCE_SCHEMA_VERSION,
        "dataset": "twowiki_provenance",
        "splits": {
            "train": train_statistics,
            "dev": _split_statistics(dev_records),
            "test": _split_statistics(test_records),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "statistics.json", statistics)
    return 0


def _split_statistics(
    records: Sequence[TwoWikiProvenanceRawRecord],
) -> dict[str, object]:
    candidate_counts = [len(record["ranking"]["candidates"]) for record in records]
    node_counts = [len(record["ranking"]["graph"]["nodes"]) for record in records]
    edge_counts = [len(record["ranking"]["graph"]["edges"]) for record in records]
    feed_out_degrees = [
        sum(
            1
            for edge in record["ranking"]["graph"]["edges"]
            if edge["edge_type"] == "feeds" and edge["source"] == candidate["output_id"]
        )
        for record in records
        for candidate in record["ranking"]["candidates"]
    ]
    gold_ranks: list[int] = []
    gold_weights: list[float] = []
    non_gold_ranks: list[int] = []
    non_gold_weights: list[float] = []
    role_counts: Counter[str] = Counter()
    gold_bucket_counts: Counter[str] = Counter()
    non_gold_bucket_counts: Counter[str] = Counter()
    source_masses: list[float] = []
    source_mass_violations = 0
    gold_head_count = 0
    for record in records:
        label = record["label"]
        gold_source, gold_target = label["gold_dependency_edges"][0]
        call_to_output = {
            candidate["call_id"]: candidate["output_id"]
            for candidate in record["ranking"]["candidates"]
        }
        feeds_by_source: defaultdict[str, list[ProvenanceEdgeRecord]] = defaultdict(
            list
        )
        for edge in record["ranking"]["graph"]["edges"]:
            if edge["edge_type"] != "feeds":
                continue
            feeds_by_source[edge["source"]].append(edge)
            metadata = edge["metadata"]
            role = cast(str, metadata["branch_role"])
            bucket = cast(str, metadata["rank_bucket"])
            role_counts[role] += 1
            is_gold = (
                edge["source"] == gold_source
                and call_to_output[edge["target"]] == gold_target
            )
            if is_gold:
                gold_ranks.append(cast(int, metadata["semantic_rank"]))
                gold_weights.append(float(edge["weight"]))
                gold_bucket_counts[bucket] += 1
                gold_head_count += int(role == "semantic_head")
            else:
                non_gold_ranks.append(cast(int, metadata["semantic_rank"]))
                non_gold_weights.append(float(edge["weight"]))
                non_gold_bucket_counts[bucket] += 1
        construction = record["ranking"]["metadata"]["graph_construction"]
        if not isinstance(construction, dict):
            raise TypeError("graph_construction metadata must be an object.")
        floor_value = construction["weight_floor"]
        if not isinstance(floor_value, int | float):
            raise TypeError("graph_construction.weight_floor must be numeric.")
        floor = float(floor_value)
        expected_mass = 2.0 * floor + (1.0 - floor)
        for edges in feeds_by_source.values():
            mass = sum(float(edge["weight"]) for edge in edges)
            source_masses.append(mass)
            source_mass_violations += int(abs(mass - expected_mass) > 1e-9)
    return {
        "records": len(records),
        "candidate_outputs": _summary(candidate_counts),
        "nodes": _summary(node_counts),
        "edges": _summary(edge_counts),
        "gold_path_length": 1,
        "feeds_out_degree": _summary(feed_out_degrees),
        "gold_edge_semantic_rank": _summary(gold_ranks),
        "non_gold_edge_semantic_rank": _summary(non_gold_ranks),
        "gold_edge_weight": _summary(gold_weights),
        "non_gold_edge_weight": _summary(non_gold_weights),
        "branch_role_counts": dict(sorted(role_counts.items())),
        "gold_rank_bucket_counts": dict(sorted(gold_bucket_counts.items())),
        "non_gold_rank_bucket_counts": dict(
            sorted(non_gold_bucket_counts.items())
        ),
        "gold_head_count": gold_head_count,
        "gold_head_rate": gold_head_count / len(records) if records else 0.0,
        "source_feed_mass": _summary(source_masses),
        "source_feed_mass_violation_count": source_mass_violations,
    }


def _summary(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _record_list(path: Path) -> list[object]:
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"2Wiki source must be a JSON list: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dense_ranker(args: argparse.Namespace) -> DenseTaskRetriever:
    return DenseTaskRetriever(
        config=DenseConfig(
            model_name=args.dense_model,
            query_prefix=args.dense_query_prefix,
            passage_prefix=args.dense_passage_prefix,
            batch_size=args.dense_batch_size,
            device=args.device,
        ),
        device=args.device,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert labeled 2Wiki files into synthetic execution-provenance raw data."
    )
    parser.add_argument("--train-source", default="data/2wiki/raw/train.json")
    parser.add_argument("--dev-source", default="data/2wiki/raw/dev.json")
    parser.add_argument("--output-dir", default="data/twowiki_provenance/raw")
    parser.add_argument("--candidate-cap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    parser.add_argument(
        "--edge-scorer", choices=("bm25", "dense", "hybrid"), default="bm25"
    )
    parser.add_argument("--successors-per-output", type=int, default=2)
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.5)
    parser.add_argument("--scorer-identity", default="provenance_semantic_v3")
    parser.add_argument("--query-template-version", default="question_source_v1")
    parser.add_argument("--semantic-temperature", type=float, default=0.1)
    parser.add_argument("--weight-floor", type=float, default=0.5)
    parser.add_argument("--branch-policy-version", default="rank_banded_v1")
    parser.add_argument("--near-rank-bucket", type=_rank_bucket, default=(2, 4))
    parser.add_argument("--mid-rank-bucket", type=_rank_bucket, default=(5, 8))
    parser.add_argument("--tail-rank-bucket", type=_rank_bucket, default=(9, None))
    parser.add_argument("--dense-model", default="models/intfloat-e5-base-v2")
    parser.add_argument("--dense-query-prefix", default="query: ")
    parser.add_argument("--dense-passage-prefix", default="passage: ")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser


def _rank_bucket(value: str) -> tuple[int, int | None]:
    try:
        lower_text, upper_text = value.split(":", maxsplit=1)
        lower = int(lower_text)
        upper = None if upper_text in {"*", "none", "None"} else int(upper_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "rank bucket must use LOWER:UPPER or LOWER:*"
        ) from error
    return lower, upper


if __name__ == "__main__":
    raise SystemExit(main())
