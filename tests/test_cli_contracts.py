from __future__ import annotations

import argparse
from typing import Any

from graph_memory.registry import Registry


def test_core_script_parser_contracts_are_frozen() -> None:
    assert _parser_contract(Registry.configs.RETRIEVE.parser_factory()) == {
        "config": _store("--config", required=True),
    }
    assert _parser_contract(Registry.configs.PAIRS.parser_factory()) == {
        "config": _store("--config", required=True),
    }
    assert _parser_contract(Registry.configs.TRAIN.parser_factory()) == {
        "config": _store("--config", required=True),
    }
    assert _parser_contract(Registry.configs.EVALUATE.parser_factory()) == {
        "config": _store("--config", required=True),
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
