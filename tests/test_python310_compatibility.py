from __future__ import annotations

import ast
from pathlib import Path


PYTHON310_MISSING_TYPING_NAMES = {"NotRequired"}


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
