from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from graph_memory.evaluation.service import evaluate_results
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.builder import build_graph
from graph_memory.datasets.hotpotqa import convert_hotpotqa_example, parse_hotpotqa_example
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.registry.retrieval_builders import RETRIEVAL_REGISTRY
from graph_memory.registry.stage_configs import RetrieveIO, RetrieveStageConfig
from graph_memory.stages.retrieve import run_retrieve_stage
from scripts.workflow.manifest import initialize_experiment
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.types import StageId
from tests.test_experiment_runner import (
    TRAINABLE_METHOD,
    _write_rgcn_training_config,
    _write_trainable_experiment_config,
)


def run_retrieval(
    *,
    method,
    task_inputs,
    graphs,
    top_k,
    encoder_model="intfloat/e5-base-v2",
    query_prefix="query: ",
    passage_prefix="passage: ",
    dense_encoder=None,
    graph_config=None,
):
    job = RETRIEVAL_REGISTRY.settings_from_runtime(
        method=method,
        top_k=top_k,
        dense_config=DenseConfig(
            model_name=encoder_model,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        ),
        graph_config=graph_config,
    )
    result = run_retrieve_stage(
        RetrieveStageConfig(
            io=RetrieveIO(
                tasks=Path("memory_tasks.input.json"),
                graphs=Path("graphs.json") if graphs is not None else None,
                output=Path("ranked.json"),
                summary=Path("ranked.run_summary.json"),
            ),
            job=job,
        ),
        task_inputs=task_inputs,
        graphs=graphs,
        graph_config=graph_config,
        dense_encoder=dense_encoder,
    )
    return result.predictions


def test_public_script_parser_contracts_are_frozen() -> None:
    import scripts.aggregate_tables as aggregate_tables
    import scripts.build_graphs as build_graphs
    import scripts.build_train_pairs as build_train_pairs
    import scripts.evaluate_retrieval as evaluate_retrieval
    import scripts.prepare_hotpotqa as prepare_hotpotqa
    import scripts.run_retrieval as run_retrieval_script
    import scripts.train_graph_retriever as train_graph_retriever
    import scripts.tune_graph_rerank as tune_graph_rerank

    expected: dict[str, dict[str, dict[str, Any]]] = {
        "scripts.prepare_hotpotqa": {
            "input": _store("--input", required=True),
            "output_input": _store("--output_input", required=True),
            "output_labels": _store("--output_labels", required=True),
            "output_combined": _store("--output_combined"),
            "max_examples": _store("--max_examples", value_type="int"),
            "seed": _store("--seed", default=13, value_type="int"),
            "offset": _store("--offset", default=0, value_type="int"),
            "strict_invalid_examples": _flag("--strict_invalid_examples"),
        },
        "scripts.build_graphs": {
            "input": _store("--input", required=True),
            "output": _store("--output", required=True),
            "max_query_overlap": _store("--max_query_overlap", default=20, value_type="int"),
            "max_entity_neighbors": _store("--max_entity_neighbors", default=10, value_type="int"),
            "max_bridge_edges": _store("--max_bridge_edges", default=50, value_type="int"),
            "use_spacy": _flag("--use_spacy"),
        },
        "scripts.run_retrieval": {
            "method": _store(
                "--method",
                required=True,
                choices=(
                    "bm25",
                    "dense",
                    "bm25_graph_rerank",
                    "dense_graph_rerank",
                    "dense_rgcn_graph_retriever",
                ),
            ),
            "tasks": _store("--tasks", required=True),
            "graphs": _store("--graphs"),
            "output": _store("--output", required=True),
            "top_k": _store("--top_k", default=10, value_type="int"),
            "encoder_model": _store("--encoder_model", default="intfloat/e5-base-v2"),
            "query_prefix": _store("--query_prefix", default="query: "),
            "passage_prefix": _store("--passage_prefix", default="passage: "),
            "graph_config": _store("--graph_config"),
            "checkpoint": _store("--checkpoint"),
            "device": _store("--device", default="cpu"),
        },
        "scripts.tune_graph_rerank": {
            "method": _store("--method", required=True, choices=("bm25_graph_rerank", "dense_graph_rerank")),
            "tasks": _store("--tasks", required=True),
            "labels": _store("--labels", required=True),
            "graphs": _store("--graphs", required=True),
            "output_config": _store("--output_config", required=True),
            "encoder_model": _store("--encoder_model", default="intfloat/e5-base-v2"),
            "query_prefix": _store("--query_prefix", default="query: "),
            "passage_prefix": _store("--passage_prefix", default="passage: "),
            "top_k": _store("--top_k", default=10, value_type="int"),
            "grid_config": _store("--grid_config"),
        },
        "scripts.build_train_pairs": {
            "tasks": _store("--tasks", required=True),
            "labels": _store("--labels", required=True),
            "graphs": _store("--graphs", required=True),
            "output": _store("--output", required=True),
            "random_seed": _store("--random_seed", default=13, value_type="int"),
            "easy_random_per_positive": _store("--easy_random_per_positive", default=2, value_type="int"),
            "hard_bm25_per_positive": _store("--hard_bm25_per_positive", default=2, value_type="int"),
            "hard_dense_per_positive": _store("--hard_dense_per_positive", default=2, value_type="int"),
            "hard_graph_neighbor_per_positive": _store("--hard_graph_neighbor_per_positive", default=1, value_type="int"),
            "hard_pool_size": _store("--hard_pool_size", default=30, value_type="int"),
            "config": _store("--config"),
        },
        "scripts.train_graph_retriever": {
            "train_tasks": _store("--train_tasks", required=True),
            "train_labels": _store("--train_labels", required=True),
            "train_graphs": _store("--train_graphs", required=True),
            "train_pairs": _store("--train_pairs", required=True),
            "dev_tasks": _store("--dev_tasks", required=True),
            "dev_labels": _store("--dev_labels", required=True),
            "dev_graphs": _store("--dev_graphs", required=True),
            "output_dir": _store("--output_dir", required=True),
            "encoder_model": _store("--encoder_model", default="intfloat/e5-base-v2"),
            "query_prefix": _store("--query_prefix", default="query: "),
            "passage_prefix": _store("--passage_prefix", default="passage: "),
            "hidden_dim": _store("--hidden_dim", default=256, value_type="int"),
            "num_layers": _store("--num_layers", default=2, value_type="int"),
            "dropout": _store("--dropout", default=0.1, value_type="float"),
            "ablation": _store(
                "--ablation",
                default="full_rgcn",
                choices=("full_rgcn", "wo_graph", "wo_edge_type", "wo_bridge", "wo_edge_weight", "wo_seed_score"),
            ),
            "epochs": _store("--epochs", default=1, value_type="int"),
            "batch_size": _store("--batch_size", default=1, value_type="int"),
            "learning_rate": _store("--learning_rate", default=0.0001, value_type="float"),
            "max_grad_norm": _store("--max_grad_norm", default=1.0, value_type="float"),
            "random_seed": _store("--random_seed", default=13, value_type="int"),
            "pos_weight": _flag("--pos_weight"),
            "device": _store("--device", default="cpu"),
            "config": _store("--config"),
        },
        "scripts.evaluate_retrieval": {
            "pred": _store("--pred", required=True),
            "labels": _store("--labels"),
            "gold": _store("--gold"),
            "graphs": _store("--graphs", required=True),
            "output": _store("--output", required=True),
            "failure_cases_output": _store("--failure_cases_output"),
            "failure_case_limit": _store("--failure_case_limit", default=0, value_type="int"),
        },
        "scripts.aggregate_tables": {
            "input_dir": _store("--input_dir", required=True),
            "output_main": _store("--output_main", required=True),
            "output_path": _store("--output_path", required=True),
            "output_efficiency": _store("--output_efficiency", required=True),
            "ablation_index": _store("--ablation_index"),
            "output_ablation": _store("--output_ablation"),
            "ablation_selection": _append("--ablation_selection", default=[]),
        },
    }
    actual = {
        "scripts.prepare_hotpotqa": _parser_contract(prepare_hotpotqa.build_parser()),
        "scripts.build_graphs": _parser_contract(build_graphs.build_parser()),
        "scripts.run_retrieval": _parser_contract(run_retrieval_script.build_parser()),
        "scripts.tune_graph_rerank": _parser_contract(tune_graph_rerank.build_parser()),
        "scripts.build_train_pairs": _parser_contract(build_train_pairs.build_parser()),
        "scripts.train_graph_retriever": _parser_contract(train_graph_retriever.build_parser()),
        "scripts.evaluate_retrieval": _parser_contract(evaluate_retrieval.build_parser()),
        "scripts.aggregate_tables": _parser_contract(aggregate_tables.build_parser()),
    }

    assert actual == expected


def test_experiment_parser_contract_is_frozen() -> None:
    import scripts.experiment as experiment

    parser = experiment.build_parser()
    root_subparsers = _subparsers(parser)

    assert tuple(root_subparsers.choices) == (
        "init",
        "plan",
        "run",
        "status",
        "stages",
        "methods",
        "configs",
        "profile",
        "profiles",
        "recipes",
        "ablations",
    )
    assert _parser_contract(root_subparsers.choices["init"]) == {
        "name": _positional(),
        "run_root": _store("--run-root", default="runs"),
        "profile": _store("--profile"),
        "method": _append("--method"),
        "methods": _store("--methods"),
        "top_k": _store("--top-k", value_type="int"),
        "config": _store("--config"),
        "force": _flag("--force"),
    }
    assert _parser_contract(root_subparsers.choices["plan"]) == {
        "name": _positional(),
        "run_root": _store("--run-root", default="runs"),
        "method": _append("--method"),
        "methods": _store("--methods"),
        "stages": _store("--stages"),
        "from_stage": _store("--from"),
        "to_stage": _store("--to"),
        "color": _store("--color", default="auto", choices=("auto", "always", "never")),
        "variant": _append("--variant"),
        "ablations_only": _flag("--ablations-only"),
    }
    assert _parser_contract(root_subparsers.choices["run"]) == {
        "name": _positional(),
        "run_root": _store("--run-root", default="runs"),
        "profile": _store("--profile"),
        "method": _append("--method"),
        "methods": _store("--methods"),
        "top_k": _store("--top-k", value_type="int"),
        "config": _store("--config"),
        "stages": _store("--stages"),
        "from_stage": _store("--from"),
        "to_stage": _store("--to"),
        "force": _flag("--force"),
        "variant": _append("--variant"),
        "ablations_only": _flag("--ablations-only"),
    }
    assert _parser_contract(root_subparsers.choices["status"]) == {
        "name": _positional(),
        "run_root": _store("--run-root", default="runs"),
        "method": _append("--method"),
        "methods": _store("--methods"),
    }

    nested_expectations = {
        "stages": {"list": {"method": _append("--method"), "methods": _store("--methods")}},
        "methods": {"list": {}},
        "configs": {"list": {"kind": _store("--kind", default="all", choices=("all", "experiments", "search-spaces", "training"))}},
        "profile": {"list": {"config": _store("--config")}},
        "profiles": {"list": {"config": _store("--config")}},
        "recipes": {"list": {}},
        "ablations": {"list": {"method": _store("--method")}},
    }
    for command, expected_subcommands in nested_expectations.items():
        subparsers = _subparsers(root_subparsers.choices[command])
        assert tuple(subparsers.choices) == tuple(expected_subcommands)
        for subcommand, expected_contract in expected_subcommands.items():
            assert _parser_contract(subparsers.choices[subcommand]) == expected_contract


def test_workflow_plan_contract_freezes_manifest_commands_and_ablation_fail_fast() -> None:
    run_root = _fresh_ignored_run_root("workflow-plan")
    config_path = _write_ablation_config(run_root, variants=["wo_graph", "wo_hard_negatives"])

    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=run_root / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    assert manifest["schema_version"] == 2
    assert manifest["profile"] == "quick"
    assert manifest["selected_methods"] == [TRAINABLE_METHOD]
    assert manifest["selected_stages"] == ["prepare", "graphs", "pairs", "train", "retrieve", "evaluate", "aggregate"]
    assert manifest["effective_config"]["training"][TRAINABLE_METHOD]["profile"] == "quick"
    assert list(manifest["artifacts"]["ablations"][TRAINABLE_METHOD]) == [
        "full_rgcn",
        "wo_graph",
        "wo_hard_negatives",
    ]

    with pytest.raises(ValueError, match="Ablation-only execution requires ordinary baseline metrics"):
        build_stage_plan(
            manifest,
            methods=[TRAINABLE_METHOD],
            variants=["wo_graph"],
            ablations_only=True,
        )

    _write_main_rgcn_metrics_placeholder(manifest)
    commands = build_stage_plan(
        manifest,
        methods=[TRAINABLE_METHOD],
        variants=["wo_graph"],
        ablations_only=True,
    )

    assert [command.stage for command in commands] == [
        StageId.PREPARE,
        StageId.PREPARE,
        StageId.PREPARE,
        StageId.GRAPHS,
        StageId.GRAPHS,
        StageId.GRAPHS,
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
        StageId.AGGREGATE,
    ]
    assert [command.variant for command in commands if command.stage is StageId.TRAIN] == ["wo_graph"]
    assert [command.method for command in commands if command.stage is StageId.PAIRS] == [TRAINABLE_METHOD]
    pair_command = next(command for command in commands if command.stage is StageId.PAIRS)
    train_command = next(command for command in commands if command.stage is StageId.TRAIN)
    aggregate_command = next(command for command in commands if command.stage is StageId.AGGREGATE)

    assert pair_command.argv[1].endswith("scripts/build_train_pairs.py")
    assert "--config" in pair_command.argv
    assert pair_command.argv[pair_command.argv.index("--output") + 1].endswith(
        "learned/dense_rgcn_graph_retriever/train.pairs.json"
    )
    assert train_command.argv[1].endswith("scripts/train_graph_retriever.py")
    assert _posix_arg(train_command.argv[train_command.argv.index("--output_dir") + 1]).endswith(
        "ablations/dense_rgcn_graph_retriever/wo_graph"
    )
    assert _repeated_arg_values(aggregate_command.argv, "--ablation_selection") == [
        f"{TRAINABLE_METHOD}=full_rgcn",
        f"{TRAINABLE_METHOD}=wo_graph",
    ]


def test_foundation_domain_golden_fixture_is_frozen() -> None:
    converted = convert_hotpotqa_example(parse_hotpotqa_example(_raw_hotpotqa_example()))
    graph = build_graph(converted.task_input, GraphBuildConfig())

    assert converted.task_input == {
        "task_id": "hotpot_ex1",
        "query": "Which river runs through the city with the Eiffel Tower?",
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "The Eiffel Tower is in Paris.",
                "source": "Eiffel Tower",
                "sentence_id": 0,
                "position": 0,
            },
            {
                "id": "m1",
                "node_type": "document_sentence",
                "text": "It opened in 1889.",
                "source": "Eiffel Tower",
                "sentence_id": 1,
                "position": 1,
            },
            {
                "id": "m2",
                "node_type": "document_sentence",
                "text": "Paris is a city in France.",
                "source": "Paris",
                "sentence_id": 0,
                "position": 2,
            },
            {
                "id": "m3",
                "node_type": "document_sentence",
                "text": "The Seine runs through Paris.",
                "source": "Paris",
                "sentence_id": 1,
                "position": 3,
            },
        ],
    }
    assert converted.task_labels == {
        "task_id": "hotpot_ex1",
        "gold_answer": "Seine",
        "gold_evidence_nodes": ["m0", "m3"],
        "gold_dependency_edges": [],
    }
    assert graph["edges"] == [
        {"source": "m0", "target": "m1", "edge_type": "sequential", "weight": 1.0, "directed": False},
        {"source": "m2", "target": "m3", "edge_type": "sequential", "weight": 1.0, "directed": False},
        {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": pytest.approx(9.31093021621633), "directed": True},
        {"source": "q", "target": "m1", "edge_type": "query_overlap", "weight": pytest.approx(9.31093021621633), "directed": True},
        {"source": "q", "target": "m2", "edge_type": "query_overlap", "weight": pytest.approx(1.6931471805599454), "directed": True},
        {"source": "q", "target": "m3", "edge_type": "query_overlap", "weight": pytest.approx(1.6931471805599454), "directed": True},
        {"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 3.0, "directed": False},
        {"source": "m0", "target": "m2", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        {"source": "m0", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        {"source": "m2", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 2.0, "directed": False},
        {"source": "m0", "target": "m3", "edge_type": "bridge", "weight": 2.0, "directed": False},
    ]

    bm25 = run_retrieval(method="bm25", task_inputs=[converted.task_input], graphs=None, top_k=2)[0]
    dense = run_retrieval(
        method="dense",
        task_inputs=[converted.task_input],
        graphs=None,
        top_k=2,
        encoder_model="fake-model",
        dense_encoder=_FakeEncoder(),
    )[0]
    graph_rerank = run_retrieval(
        method="bm25_graph_rerank",
        task_inputs=[converted.task_input],
        graphs=[graph],
        top_k=2,
        graph_config=GraphRerankConfig(
            lambda_init=0.0,
            lambda_query=0.1,
            lambda_neighbor=0.0,
            lambda_bridge=1.0,
            seed_top_s=1,
            max_hops=1,
        ),
    )[0]

    assert [(node["node_id"], node["score"]) for node in bm25["ranked_nodes"]] == [
        ("m2", pytest.approx(0.8703361707904812)),
        ("m3", pytest.approx(0.8703361707904812)),
        ("m0", pytest.approx(0.0)),
        ("m1", pytest.approx(0.0)),
    ]
    assert dense["input_tokens"] == 22
    assert [(node["node_id"], node["score"]) for node in dense["ranked_nodes"]] == [
        ("m1", pytest.approx(1.0)),
        ("m0", pytest.approx(0.7071067811865475)),
        ("m2", pytest.approx(0.0)),
        ("m3", pytest.approx(0.0)),
    ]
    assert [(node["node_id"], node["score"]) for node in graph_rerank["ranked_nodes"]] == [
        ("m0", pytest.approx(1.1)),
        ("m2", pytest.approx(0.01818451154978141)),
        ("m3", pytest.approx(0.01818451154978141)),
        ("m1", pytest.approx(0.0)),
    ]
    assert graph_rerank["retrieved_subgraph"] == {
        "nodes": ["m0", "m2"],
        "edges": [
            {"source": "m0", "target": "m2", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 2.0, "directed": False},
        ],
    }

    metrics = evaluate_results([graph_rerank], [converted.task_labels], [graph])[0]
    metrics_without_latency = {key: value for key, value in metrics.items() if key != "Retrieval Latency / Query"}
    assert metrics_without_latency == {
        "Method": "bm25_graph_rerank",
        "Recall@2": 0.5,
        "Recall@5": 1.0,
        "Recall@10": 1.0,
        "Evidence F1@5": 0.5714285714285715,
        "Evidence F1@10": 0.33333333333333337,
        "Full Support@5": 1.0,
        "Full Support@10": 1.0,
        "MRR": 1.0,
        "Connected Evidence Recall@5": 1.0,
        "Connected Evidence Recall@10": 1.0,
        "Query-Evidence Connectivity@10": 1.0,
        "Path Recall@10": "N/A",
        "Edge Recall@10": "N/A",
        "Index Build Time": 0.0,
        "Graph Construction Time": 0.0,
        "Memory Size": 4.0,
        "Avg Retrieved Nodes": 2.0,
        "Avg Retrieved Edges": 2.0,
    }


def _parser_contract(parser: argparse.ArgumentParser) -> dict[str, dict[str, Any]]:
    return {
        action.dest: _action_contract(action)
        for action in parser._actions
        if action.dest != "help" and not isinstance(action, argparse._SubParsersAction)
    }


def _action_contract(action: argparse.Action) -> dict[str, Any]:
    return {
        "options": tuple(action.option_strings),
        "required": bool(getattr(action, "required", False)),
        "default": action.default,
        "type": getattr(action.type, "__name__", None),
        "choices": tuple(action.choices) if action.choices is not None else None,
        "nargs": action.nargs,
        "action": type(action).__name__,
    }


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    return next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))


def _store(
    option: str,
    *,
    required: bool = False,
    default: object | None = None,
    value_type: str | None = None,
    choices: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "options": (option,),
        "required": required,
        "default": default,
        "type": value_type,
        "choices": choices,
        "nargs": None,
        "action": "_StoreAction",
    }


def _append(option: str, *, default: object | None = None) -> dict[str, Any]:
    return {
        "options": (option,),
        "required": False,
        "default": default,
        "type": None,
        "choices": None,
        "nargs": None,
        "action": "_AppendAction",
    }


def _flag(option: str) -> dict[str, Any]:
    return {
        "options": (option,),
        "required": False,
        "default": False,
        "type": None,
        "choices": None,
        "nargs": 0,
        "action": "_StoreTrueAction",
    }


def _positional() -> dict[str, Any]:
    return {
        "options": (),
        "required": True,
        "default": None,
        "type": None,
        "choices": None,
        "nargs": None,
        "action": "_StoreAction",
    }


def _fresh_ignored_run_root(name: str) -> Path:
    root = Path("report/tmp/core-refactor-baseline-tests") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _write_ablation_config(run_root: Path, *, variants: list[str]) -> Path:
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = run_root / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = run_root / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["enable_ablation"] = True
    payload["ablation_variants"] = {TRAINABLE_METHOD: variants}
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _write_main_rgcn_metrics_placeholder(manifest: dict[str, object]) -> None:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    metrics = artifacts["metrics"]
    assert isinstance(metrics, dict)
    path = Path(metrics[TRAINABLE_METHOD])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("core refactor baseline metrics placeholder", encoding="utf-8")


def _repeated_arg_values(argv: list[str], name: str) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv) if value == name]


def _posix_arg(value: str) -> str:
    return value.replace("\\", "/")


def _raw_hotpotqa_example() -> dict[str, object]:
    return {
        "_id": "ex1",
        "question": "Which river runs through the city with the Eiffel Tower?",
        "answer": "Seine",
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is in Paris.", "It opened in 1889."]],
            ["Paris", ["Paris is a city in France.", "The Seine runs through Paris."]],
        ],
        "supporting_facts": [["Eiffel Tower", 0], ["Paris", 1]],
    }


class _FakeEncoder:
    def encode(self, texts: Sequence[str], batch_size: int = 64, normalize_embeddings: bool = True) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "eiffel" in lowered else 0.0,
                    1.0 if "paris" in lowered else 0.0,
                    1.0 if "seine" in lowered else 0.0,
                ]
            )
        array = np.array(vectors, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            array = array / norms
        return array
