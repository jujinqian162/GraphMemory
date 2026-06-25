from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "graph_memory"
DOMAIN_PACKAGE_ROOTS = (
    PACKAGE_ROOT / "contracts",
    PACKAGE_ROOT / "datasets",
    PACKAGE_ROOT / "evaluation",
    PACKAGE_ROOT / "graphs",
    PACKAGE_ROOT / "infrastructure",
    PACKAGE_ROOT / "models",
    PACKAGE_ROOT / "retrieval",
    PACKAGE_ROOT / "text",
    PACKAGE_ROOT / "training_pairs",
    PACKAGE_ROOT / "validation",
)
ROOT_WORKFLOW_PORT_MODULES = {
    "graph_memory.io",
    "graph_memory.observability",
}
# Inner packages must not depend on the application/orchestration layers above
# them. This enforces the dependency direction of the architecture so a future
# edit cannot quietly introduce an upward import.
FORBIDDEN_PACKAGE_IMPORTS = {
    PACKAGE_ROOT / "contracts": (
        "graph_memory.datasets",
        "graph_memory.graphs",
        "graph_memory.retrieval",
        "graph_memory.training_pairs",
        "graph_memory.models",
        "graph_memory.evaluation",
        "graph_memory.application",
        "scripts",
    ),
    PACKAGE_ROOT / "graphs": (
        "graph_memory.retrieval",
        "graph_memory.training_pairs",
        "graph_memory.models",
        "graph_memory.evaluation",
        "graph_memory.application",
        "scripts",
    ),
    PACKAGE_ROOT / "retrieval": ("graph_memory.application", "scripts"),
    PACKAGE_ROOT / "models" / "graph_retriever": ("graph_memory.application", "scripts"),
    PACKAGE_ROOT / "infrastructure": (
        "graph_memory.datasets",
        "graph_memory.graphs",
        "graph_memory.retrieval",
        "graph_memory.training_pairs",
        "graph_memory.models",
        "graph_memory.evaluation",
        "graph_memory.application",
        "scripts",
    ),
}


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.lineno, node.module))
    return imports


def _package_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def test_core_package_dependency_direction_is_enforced() -> None:
    violations: list[str] = []
    for root, forbidden_prefixes in FORBIDDEN_PACKAGE_IMPORTS.items():
        for path in _package_files(root):
            for lineno, imported in _imported_modules(path):
                for forbidden in forbidden_prefixes:
                    if imported == forbidden or imported.startswith(f"{forbidden}."):
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}")

    assert violations == []


def test_domain_packages_do_not_import_root_workflow_integration_ports() -> None:
    violations: list[str] = []
    for root in DOMAIN_PACKAGE_ROOTS:
        for path in _package_files(root):
            for lineno, imported in _imported_modules(path):
                if imported in ROOT_WORKFLOW_PORT_MODULES:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}")

    assert violations == []
