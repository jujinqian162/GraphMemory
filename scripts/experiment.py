from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workflow import (
    build_stage_plan,
    discover_ablation_variants,
    format_commands,
    format_status,
    initialize_experiment,
    inspect_experiment_status,
    list_config_entries,
    list_method_specs,
    list_profile_specs,
    list_recipe_specs,
    list_stage_specs,
    load_experiment_config,
    load_manifest,
    prune_manifest_completed_prefix,
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
        methods = _parse_methods(args)
        manifest = initialize_experiment(
            args.name,
            config=config,
            run_root=args.run_root,
            profile=args.profile,
            methods=methods,
            cli_overrides=cli_overrides,
            force=args.force,
        )
        print(manifest["paths"]["manifest"])
        return 0

    if args.command == "plan":
        manifest = load_manifest(args.name, run_root=args.run_root)
        methods = _parse_methods(args)
        commands = build_stage_plan(
            manifest,
            stages=_parse_csv(args.stages),
            methods=methods,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            variants=args.variant,
            ablations_only=args.ablations_only,
        )
        if not args.no_cache:
            commands = list(prune_manifest_completed_prefix(manifest, commands).commands)
        print(format_commands(commands, color=_color_enabled(args.color)))
        return 0

    if args.command == "run":
        manifest = _load_or_initialize_for_run(args)
        methods = _parse_methods(args)
        commands = build_stage_plan(
            manifest,
            stages=_parse_csv(args.stages),
            methods=methods,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            variants=args.variant,
            ablations_only=args.ablations_only,
        )
        if not args.no_cache:
            commands = list(prune_manifest_completed_prefix(manifest, commands).commands)
        run_stage_plan(commands)
        update_manifest_status(manifest)
        return 0

    if args.command == "status":
        manifest = load_manifest(args.name, run_root=args.run_root)
        print(format_status(inspect_experiment_status(manifest)))
        return 0

    if args.command == "stages":
        print(_format_stage_specs(list_stage_specs(_parse_methods(args))))
        return 0

    if args.command == "methods":
        print(_format_method_specs(list_method_specs()))
        return 0

    if args.command == "configs":
        print(_format_config_entries(list_config_entries(args.kind)))
        return 0

    if args.command in {"profile", "profiles"}:
        config = load_experiment_config(args.config)
        print(_format_profile_specs(list_profile_specs(config)))
        return 0

    if args.command == "recipes":
        print(_format_recipe_specs(list_recipe_specs()))
        return 0

    if args.command == "ablations":
        print(_format_ablation_specs(discover_ablation_variants(args.method)))
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
    plan_parser.add_argument("--to", dest="to_stage", default=None, help="Plan through this stage.")
    plan_parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    plan_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Show all selected commands instead of pruning the completed prefix from live artifact status.",
    )
    _add_ablation_selection_args(plan_parser)

    run_parser = subparsers.add_parser("run", help="Execute selected experiment stages.")
    _add_common_run_args(run_parser, include_config=True)
    run_parser.add_argument("--stages", default=None, help="Comma-separated stages to run.")
    run_parser.add_argument("--from", dest="from_stage", default=None, help="Run from this stage onward.")
    run_parser.add_argument("--to", dest="to_stage", default=None, help="Run through this stage.")
    run_parser.add_argument("--force", action="store_true", help="Reinitialize an existing manifest before running.")
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Execute all selected commands instead of pruning the completed prefix from live artifact status.",
    )
    _add_ablation_selection_args(run_parser)

    status_parser = subparsers.add_parser("status", help="Show experiment artifact status.")
    _add_existing_manifest_args(status_parser)

    stages_parser = subparsers.add_parser("stages", help="Inspect public experiment stages.")
    stages_subparsers = stages_parser.add_subparsers(dest="subcommand", required=True)
    stages_list_parser = stages_subparsers.add_parser("list", help="List stages in workflow order.")
    _add_method_selection_args(stages_list_parser)

    methods_parser = subparsers.add_parser("methods", help="Inspect retrieval methods.")
    methods_subparsers = methods_parser.add_subparsers(dest="subcommand", required=True)
    methods_subparsers.add_parser("list", help="List supported retrieval methods.")

    configs_parser = subparsers.add_parser("configs", help="Inspect experiment config contracts.")
    configs_subparsers = configs_parser.add_subparsers(dest="subcommand", required=True)
    configs_list_parser = configs_subparsers.add_parser("list", help="List known config files.")
    configs_list_parser.add_argument(
        "--kind",
        choices=("all", "experiments", "search-spaces", "training"),
        default="all",
    )

    _add_profile_parser(subparsers, "profile")
    _add_profile_parser(subparsers, "profiles")

    recipes_parser = subparsers.add_parser("recipes", help="Inspect experiment recipes.")
    recipes_subparsers = recipes_parser.add_subparsers(dest="subcommand", required=True)
    recipes_subparsers.add_parser("list", help="List experiment recipe summaries.")

    ablations_parser = subparsers.add_parser("ablations", help="Inspect registered ablation suites.")
    ablations_subparsers = ablations_parser.add_subparsers(dest="subcommand", required=True)
    ablations_list_parser = ablations_subparsers.add_parser("list", help="List registered ablation variants.")
    ablations_list_parser.add_argument("--method", default=None)
    return parser


def _add_common_run_args(parser: argparse.ArgumentParser, *, include_config: bool) -> None:
    parser.add_argument("name")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--profile", default=None)
    _add_method_selection_args(parser)
    parser.add_argument("--top-k", type=int, default=None)
    if include_config:
        parser.add_argument("--config", default=None)


def _add_existing_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name")
    parser.add_argument("--run-root", default="runs")
    _add_method_selection_args(parser)


def _add_method_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", action="append", default=None, help="Retrieval method; repeat for multiple methods.")
    parser.add_argument("--methods", default=None, help="Comma-separated retrieval methods.")


def _add_ablation_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", action="append", default=None, help="Ablation variant; repeat to narrow work.")
    parser.add_argument("--ablations-only", action="store_true", help="Plan only ablation work and shared prerequisites.")


def _add_profile_parser(subparsers: Any, name: str) -> None:
    profiles_parser = subparsers.add_parser(name, help="Inspect profiles from an experiment config.")
    profiles_subparsers = profiles_parser.add_subparsers(dest="subcommand", required=True)
    profiles_list_parser = profiles_subparsers.add_parser("list", help="List profiles in an experiment config.")
    profiles_list_parser.add_argument("--config", default=None)


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_methods(args: argparse.Namespace) -> list[str] | None:
    repeated = getattr(args, "method", None) or []
    csv_methods = _parse_csv(getattr(args, "methods", None))
    if repeated and csv_methods:
        raise ValueError("Use either --method or --methods, not both.")
    if repeated:
        methods: list[str] = []
        for value in repeated:
            methods.extend(_parse_csv(value) or [])
        return methods
    return csv_methods


def _color_enabled(value: str) -> bool:
    if value == "always":
        return True
    if value == "never":
        return False
    return sys.stdout.isatty()


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
        methods=_parse_methods(args),
        cli_overrides=_cli_overrides(args),
        force=args.force,
    )


def _format_stage_specs(rows: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{row['name']}\tdefault={row['default']}\t{row['description']}" for row in rows)


def _format_method_specs(rows: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(
            f"{row['name']}\tworkflow={row['workflow']}\t"
            f"graphs={row['requires_graphs']}\t"
            f"tune_config={row['requires_graph_config']}\t"
            f"checkpoint={row['requires_checkpoint']}\t"
            f"dense_encoder={row['requires_dense_encoder']}"
        )
    return "\n".join(lines)


def _format_config_entries(entries: Sequence[Any]) -> str:
    return "\n".join(f"{entry.kind}\t{entry.name}\t{entry.path}" for entry in entries)


def _format_profile_specs(rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        splits = row["splits"]
        split_parts = [
            (
                f"{split}[source={splits[split]['source']} "
                f"max_examples={splits[split]['max_examples']} "
                f"seed={splits[split]['seed']} "
                f"offset={splits[split]['offset']}]"
            )
            for split in ("train", "dev", "test")
        ]
        lines.append(
            f"{row['name']}\ttrain={row['train']}\tdev={row['dev']}\ttest={row['test']}\t"
            f"splits={'; '.join(split_parts)}"
        )
    return "\n".join(lines)


def _format_recipe_specs(rows: Sequence[dict[str, str]]) -> str:
    return "\n".join(
        f"{row['name']}\trecipe={row['recipe']}\tdataset={row['dataset']}\ttask={row['task']}\tpath={row['path']}"
        for row in rows
    )


def _format_ablation_specs(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        f"{row['method']}\tvariant={row['variant']}\t"
        f"dimensions={','.join(row['changed_dimensions']) or '-'}\t"
        f"baseline_alias={str(row['baseline_alias']).lower()}"
        for row in rows
    )


if __name__ == "__main__":
    raise SystemExit(main())
