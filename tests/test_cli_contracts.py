from __future__ import annotations

import argparse
from typing import Any


def test_core_script_parser_contracts_are_frozen() -> None:
    import scripts.build_train_pairs as build_train_pairs
    import scripts.evaluate_retrieval as evaluate_retrieval
    import scripts.run_retrieval as run_retrieval
    import scripts.train_graph_retriever as train_graph_retriever

    assert _parser_contract(run_retrieval.build_parser()) == {
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
    }
    assert _parser_contract(build_train_pairs.build_parser()) == {
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
    }
    assert _parser_contract(train_graph_retriever.build_parser()) == {
        "train_tasks": _store("--train_tasks", required=True),
        "train_labels": _store("--train_labels"),
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
    }
    assert _parser_contract(evaluate_retrieval.build_parser()) == {
        "pred": _store("--pred", required=True),
        "labels": _store("--labels"),
        "gold": _store("--gold"),
        "graphs": _store("--graphs", required=True),
        "output": _store("--output", required=True),
        "failure_cases_output": _store("--failure_cases_output"),
        "failure_case_limit": _store("--failure_case_limit", default=0, value_type="int"),
    }


def test_experiment_parser_contract_is_frozen() -> None:
    import scripts.experiment as experiment

    root_subparsers = _subparsers(experiment.build_parser())

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
        "no_cache": _flag("--no-cache"),
        "variant": _append("--variant"),
        "ablations_only": _flag("--ablations-only"),
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
