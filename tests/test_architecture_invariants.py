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
    PACKAGE_ROOT / "models" / "graph_retriever": (
        "graph_memory.application",
        "scripts",
    ),
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
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}"
                        )

    assert violations == []


def test_domain_packages_do_not_import_root_workflow_integration_ports() -> None:
    violations: list[str] = []
    for root in DOMAIN_PACKAGE_ROOTS:
        for path in _package_files(root):
            for lineno, imported in _imported_modules(path):
                if imported in ROOT_WORKFLOW_PORT_MODULES:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}"
                    )

    assert violations == []


def test_legacy_validation_surfaces_cannot_return() -> None:
    retired_paths = (
        PACKAGE_ROOT / "validation",
        PACKAGE_ROOT / "contracts" / "ranking.py",
        PACKAGE_ROOT / "contracts" / "graphs.py",
        PACKAGE_ROOT / "contracts" / "training_pairs.py",
        PACKAGE_ROOT / "contracts" / "metrics.py",
        PACKAGE_ROOT / "contracts" / "errors.py",
    )
    assert [path.relative_to(REPO_ROOT) for path in retired_paths if path.exists()] == []

    retired_imports = (
        "graph_memory.validation",
        "graph_memory.contracts.ranking",
        "graph_memory.contracts.graphs",
        "graph_memory.contracts.training_pairs",
        "graph_memory.contracts.metrics",
        "graph_memory.contracts.errors",
    )
    violations: list[str] = []
    for path in _package_files(PACKAGE_ROOT):
        for lineno, imported in _imported_modules(path):
            if any(
                imported == retired or imported.startswith(f"{retired}.")
                for retired in retired_imports
            ):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}"
                )
    assert violations == []


def test_central_validate_apis_cannot_return() -> None:
    allowed_tensor_assertions = {
        "graph_memory/models/graph_batching.py:validate_graph_batch",
    }
    violations: list[str] = []
    for path in _package_files(PACKAGE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "validate_"
            ):
                symbol = f"{path.relative_to(REPO_ROOT).as_posix()}:{node.name}"
                if symbol not in allowed_tensor_assertions:
                    violations.append(symbol)
    assert violations == []


def test_prefect_cutover_removes_runner_owned_orchestration() -> None:
    forbidden_files = {
        "experiment/plan.py",
        "graph_memory/experiment/execution.py",
        "graph_memory/experiment/invocation.py",
        "graph_memory/experiment/planning.py",
        "graph_memory/experiment/resume.py",
        "graph_memory/experiment/stage_cli.py",
        "graph_memory/experiment/stage_models.py",
        "graph_memory/experiment/stage_status.py",
        "graph_memory/experiment/state.py",
    }
    assert [
        path for path in sorted(forbidden_files) if (REPO_ROOT / path).exists()
    ] == []

    production_roots = (REPO_ROOT / "experiment", PACKAGE_ROOT / "experiment")
    forbidden_tokens = (
        "WorkflowPlanner",
        "StageInvocation",
        "MlflowClient",
        "subprocess.run(",
        "nested=True",
        ".submit(",
        "ablation.variants",
        "run_state.yaml",
    )
    violations: list[str] = []
    for root in production_roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                if token in source:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert violations == []
