from __future__ import annotations

from dataclasses import is_dataclass
from typing import get_type_hints

import scripts.aggregate_tables as aggregate_tables
import scripts.build_graphs as build_graphs
import scripts.evaluate_retrieval as evaluate_retrieval
import scripts.prepare_hotpotqa as prepare_hotpotqa
import scripts.run_retrieval as run_retrieval
import scripts.tune_graph_rerank as tune_graph_rerank


def test_prepare_hotpotqa_uses_typed_args_dataclass() -> None:
    args = prepare_hotpotqa.parse_args(
        [
            "--input",
            "raw.json",
            "--output_input",
            "tasks.input.json",
            "--output_labels",
            "tasks.labels.json",
            "--output_combined",
            "tasks.json",
            "--max_examples",
            "100",
            "--seed",
            "13",
            "--offset",
            "10",
        ]
    )

    assert is_dataclass(args)
    assert type(args).__name__ == "PrepareHotpotQAArgs"
    assert args.output_input == "tasks.input.json"
    assert args.max_examples == 100
    assert get_type_hints(type(args))["output_input"] is str


def test_build_graphs_uses_typed_args_dataclass() -> None:
    args = build_graphs.parse_args(
        [
            "--input",
            "tasks.input.json",
            "--output",
            "graphs.json",
            "--max_query_overlap",
            "11",
            "--use_spacy",
        ]
    )

    assert is_dataclass(args)
    assert type(args).__name__ == "BuildGraphsArgs"
    assert args.output == "graphs.json"
    assert args.max_query_overlap == 11
    assert args.use_spacy is True


def test_run_retrieval_uses_typed_args_dataclass() -> None:
    args = run_retrieval.parse_args(
        [
            "--method",
            "bm25_graph_rerank",
            "--tasks",
            "tasks.input.json",
            "--graphs",
            "graphs.json",
            "--output",
            "ranked.json",
            "--graph_config",
            "config.json",
        ]
    )

    assert is_dataclass(args)
    assert type(args).__name__ == "RunRetrievalArgs"
    assert args.method == "bm25_graph_rerank"
    assert args.graph_config == "config.json"


def test_tune_graph_rerank_uses_typed_args_dataclass() -> None:
    args = tune_graph_rerank.parse_args(
        [
            "--method",
            "bm25_graph_rerank",
            "--tasks",
            "tasks.input.json",
            "--labels",
            "tasks.labels.json",
            "--graphs",
            "graphs.json",
            "--output_config",
            "selected.json",
        ]
    )

    assert is_dataclass(args)
    assert type(args).__name__ == "TuneGraphRerankArgs"
    assert args.labels == "tasks.labels.json"
    assert args.top_k == 10


def test_evaluate_retrieval_uses_typed_args_dataclass() -> None:
    args = evaluate_retrieval.parse_args(
        [
            "--pred",
            "ranked.json",
            "--labels",
            "tasks.labels.json",
            "--graphs",
            "graphs.json",
            "--output",
            "metrics.csv",
            "--failure_case_limit",
            "5",
        ]
    )

    assert is_dataclass(args)
    assert type(args).__name__ == "EvaluateRetrievalArgs"
    assert args.labels == "tasks.labels.json"
    assert args.failure_case_limit == 5


def test_aggregate_tables_uses_typed_args_dataclass() -> None:
    args = aggregate_tables.parse_args(
        [
            "--input_dir",
            "results",
            "--output_main",
            "main.csv",
            "--output_path",
            "path.csv",
            "--output_efficiency",
            "efficiency.csv",
        ]
    )

    assert is_dataclass(args)
    assert type(args).__name__ == "AggregateTablesArgs"
    assert args.input_dir == "results"
    assert args.output_efficiency == "efficiency.csv"
