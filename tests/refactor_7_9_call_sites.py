from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CallSite:
    path: Path
    line: int
    keywords: frozenset[str]

    @property
    def is_test(self) -> bool:
        return "tests" in self.path.parts


def find_calls(repository_root: Path, function_name: str) -> tuple[CallSite, ...]:
    """Return concrete calls without treating tests as production callers."""

    sites: list[CallSite] = []
    for relative_root in (Path("graph_memory"), Path("scripts"), Path("tests")):
        for path in (repository_root / relative_root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                name = (
                    called.id
                    if isinstance(called, ast.Name)
                    else called.attr
                    if isinstance(called, ast.Attribute)
                    else None
                )
                if name != function_name:
                    continue
                sites.append(
                    CallSite(
                        path=path.relative_to(repository_root),
                        line=node.lineno,
                        keywords=frozenset(
                            keyword.arg
                            for keyword in node.keywords
                            if keyword.arg is not None
                        ),
                    )
                )
    return tuple(sorted(sites, key=lambda site: (str(site.path), site.line)))

