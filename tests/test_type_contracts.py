from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import get_origin

from graph_memory import retrieval
from graph_memory.types import JsonObject


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT / "graph_memory",
    ROOT / "scripts",
    ROOT / "tests",
]
ALLOWED_BOUNDARY_FILES = {
    ROOT / "graph_memory" / "io.py",
    ROOT / "graph_memory" / "validation.py",
}
BARE_DICT_PATTERN = re.compile(r"(^|[\[|, ])dict($|[\]|, ])")


def test_non_boundary_annotations_use_named_record_types() -> None:
    violations = []
    for scan_root in SCAN_ROOTS:
        for path in sorted(scan_root.rglob("*.py")):
            if path in ALLOWED_BOUNDARY_FILES or "__pycache__" in path.parts:
                continue
            syntax_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for annotation, line_number in _iter_annotations(syntax_tree):
                annotation_text = ast.unparse(annotation)
                if "list[dict" in annotation_text or BARE_DICT_PATTERN.search(annotation_text):
                    relative_path = path.relative_to(ROOT).as_posix()
                    violations.append(f"{relative_path}:{line_number}: {annotation_text}")

    assert violations == []


def test_json_object_uses_covariant_mapping() -> None:
    assert get_origin(JsonObject) is Mapping


def test_graph_rerank_config_is_narrowed_before_graph_call() -> None:
    source = inspect.getsource(retrieval.run_retrieval)

    assert "assert rerank_config is not None" in source


def _iter_annotations(syntax_tree: ast.AST) -> Iterator[tuple[ast.expr, int]]:
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation, node.lineno
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation, node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns, node.lineno
