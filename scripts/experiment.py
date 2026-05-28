from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.experiment import (
    build_stage_plan,
    format_commands,
    format_status,
    initialize_experiment,
    inspect_experiment_status,
    load_experiment_config,
    load_manifest,
    run_stage_plan,
    update_manifest_status,
)

LOGGER = logging.getLogger("experiment")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    if args.command == "init":
        config = load_experiment_config(args.config)
        cli_overrides = _cli_overrides(args)
        manifest = initialize_experiment(
            args.name,
            config=config,
            run_root=args.run_root,
            profile=args.profile,
            methods=_parse_csv(args.methods),
            cli_overrides=cli_overrides,
            force=args.force,
        )
        print(manifest["paths"]["manifest"])
        return 0

    if args.command == "plan":
        manifest = load_manifest(args.name, run_root=args.run_root)
        commands = build_stage_plan(
            manifest,
            stages=_parse_csv(args.stages),
            methods=_parse_csv(args.methods),
            from_stage=args.from_stage,
        )
        print(format_commands(commands))
        return 0

    if args.command == "run":
        manifest = _load_or_initialize_for_run(args)
        commands = build_stage_plan(
            manifest,
            stages=_parse_csv(args.stages),
            methods=_parse_csv(args.methods),
            from_stage=args.from_stage,
        )
        run_stage_plan(commands)
        update_manifest_status(manifest)
        return 0

    if args.command == "status":
        manifest = load_manifest(args.name, run_root=args.run_root)
        print(format_status(inspect_experiment_status(manifest)))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run named graph-memory experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create or load a named experiment manifest.")
    _add_common_run_args(init_parser, include_config=True)
    init_parser.add_argument("--force", action="store_true", help="Reinitialize an existing manifest.")

    plan_parser = subparsers.add_parser("plan", help="Print low-level commands without executing them.")
    _add_existing_manifest_args(plan_parser)
    plan_parser.add_argument("--stages", default=None, help="Comma-separated stages to plan.")
    plan_parser.add_argument("--from", dest="from_stage", default=None, help="Plan from this stage onward.")

    run_parser = subparsers.add_parser("run", help="Execute selected experiment stages.")
    _add_common_run_args(run_parser, include_config=True)
    run_parser.add_argument("--stages", default=None, help="Comma-separated stages to run.")
    run_parser.add_argument("--from", dest="from_stage", default=None, help="Run from this stage onward.")
    run_parser.add_argument("--force", action="store_true", help="Reinitialize an existing manifest before running.")

    status_parser = subparsers.add_parser("status", help="Show experiment artifact status.")
    _add_existing_manifest_args(status_parser)
    return parser


def _add_common_run_args(parser: argparse.ArgumentParser, *, include_config: bool) -> None:
    parser.add_argument("name")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated retrieval methods.")
    parser.add_argument("--top-k", type=int, default=None)
    if include_config:
        parser.add_argument("--config", default=None)


def _add_existing_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--methods", default=None, help="Comma-separated retrieval methods.")


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if getattr(args, "top_k", None) is not None:
        overrides["top_k"] = args.top_k
    return overrides or None


def _load_or_initialize_for_run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.run_root) / args.name / "manifest.json"
    if manifest_path.exists() and not args.force:
        return load_manifest(args.name, run_root=args.run_root)
    config = load_experiment_config(args.config)
    return initialize_experiment(
        args.name,
        config=config,
        run_root=args.run_root,
        profile=args.profile,
        methods=_parse_csv(args.methods),
        cli_overrides=_cli_overrides(args),
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
