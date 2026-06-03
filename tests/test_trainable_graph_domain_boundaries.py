from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "graph_memory"
SCAN_ROOTS = (PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests")
THIS_FILE = Path(__file__).resolve()
LEGACY_LEARNED_MODEL_MODULES = {
    "graph_memory.learned.batching",
    "graph_memory.learned.checkpoint",
    "graph_memory.learned.features",
    "graph_memory.learned.inference",
    "graph_memory.learned.model",
    "graph_memory.learned.tensorize",
    "graph_memory.learned.training",
}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            if path == THIS_FILE:
                continue
            files.append(path)
    return files


def _imports_module(path: Path, module_name: str) -> list[str]:
    matches: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name or alias.name.startswith(f"{module_name}."):
                    matches.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if imported == module_name or imported.startswith(f"{module_name}."):
                matches.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{imported}")
    return matches


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_training_pairs_domain_is_owned_entry() -> None:
    assert _has_module("graph_memory.training_pairs")


def test_learned_data_import_path_is_removed_from_runtime_callers() -> None:
    old_imports: list[str] = []
    for path in _python_files():
        old_imports.extend(_imports_module(path, "graph_memory.learned.data"))

    assert old_imports == []


def test_graph_retriever_model_domain_is_owned_entry() -> None:
    assert _has_module("graph_memory.models.graph_retriever")


def test_trainable_graph_retrieval_adapter_is_retrieval_owned() -> None:
    assert _has_module("graph_memory.retrieval.methods.trainable_graph")


def test_learned_model_import_paths_are_removed_from_runtime_callers() -> None:
    old_imports: list[str] = []
    for path in _python_files():
        if PACKAGE_ROOT / "learned" in path.parents:
            continue
        for module_name in LEGACY_LEARNED_MODEL_MODULES:
            old_imports.extend(_imports_module(path, module_name))

    assert old_imports == []


def test_model_inference_does_not_import_training_lifecycle() -> None:
    inference_path = PACKAGE_ROOT / "models" / "graph_retriever" / "inference.py"
    assert inference_path.exists()
    assert _imports_module(inference_path, "graph_memory.models.graph_retriever.training") == []
