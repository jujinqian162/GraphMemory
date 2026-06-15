from __future__ import annotations

import ast
from pathlib import Path


PYTHON310_MISSING_TYPING_NAMES = {
    "Never",
    "NotRequired",
    "Required",
    "Self",
    "TypeAliasType",
    "TypeVarTuple",
    "Unpack",
}
PYTHON310_NEW_MODULES = (
    "graph_memory/tuning/grid_search.py",
    "graph_memory/retrieval/tuning/selection.py",
    "graph_memory/retrieval/tuning/seed_scores.py",
    "graph_memory/retrieval/tuning/graph_rerank_grid.py",
    "graph_memory/retrieval/tuning/graph_rerank.py",
    "graph_memory/retrieval/tuning/memory_stream_grid.py",
    "graph_memory/retrieval/tuning/memory_stream.py",
    "graph_memory/retrieval/methods/memory_stream/config.py",
)


def test_project_declares_python310_minimum() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"

    assert 'requires-python = ">=3.10"' in pyproject.read_text(encoding="utf-8")


def test_python310_only_types_come_from_compatibility_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_roots = (repo_root / "graph_memory", repo_root / "scripts")

    violations: list[str] = []
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module != "typing":
                    continue
                imported_names = {alias.name for alias in node.names}
                missing_names = imported_names & PYTHON310_MISSING_TYPING_NAMES
                if missing_names:
                    relative_path = path.relative_to(repo_root).as_posix()
                    names = ", ".join(sorted(missing_names))
                    violations.append(f"{relative_path}:{node.lineno} imports {names} from typing")

    assert violations == []


def test_new_tuning_modules_parse_as_python310_syntax() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for relative in PYTHON310_NEW_MODULES:
        path = repo_root / relative
        try:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 10),
            )
        except SyntaxError as error:
            violations.append(f"{relative}:{error.lineno}:{error.msg}")

    assert violations == []
