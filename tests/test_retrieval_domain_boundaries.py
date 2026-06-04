from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import fields
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "graph_memory"
SCAN_ROOTS = (PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests")
LEGACY_ROOT_MODULES = ("retrieval.py", "rerank.py", "rerank_config.py", "tuning.py")
LEGACY_RERANK_IMPORTS = {
    "graph_memory.rerank",
    "graph_memory.rerank_config",
    "graph_memory.tuning",
}
NO_LOOSE_DENSE_PREFIX_MODULES = (
    PACKAGE_ROOT / "retrieval" / "execution" / "service.py",
    PACKAGE_ROOT / "retrieval" / "factory.py",
    PACKAGE_ROOT / "retrieval" / "resolver.py",
    PACKAGE_ROOT / "retrieval" / "tuning" / "service.py",
)
NO_LOOSE_EXECUTION_PARAMETERS = {
    "encoder_model",
    "query_prefix",
    "passage_prefix",
    "graph_config",
    "checkpoint_path",
    "text_embedding_provider",
    "seed_signal_provider",
    "device",
}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            files.append(path)
    return files


def test_retrieval_build_context_is_removed() -> None:
    matches: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "RetrievalBuildContext" in text:
            matches.append(str(path.relative_to(REPO_ROOT)))

    assert matches == []


def test_legacy_root_retrieval_modules_are_removed() -> None:
    existing = [
        str((PACKAGE_ROOT / module_name).relative_to(REPO_ROOT))
        for module_name in LEGACY_ROOT_MODULES
        if (PACKAGE_ROOT / module_name).exists()
    ]

    assert existing == []


def test_old_graph_rerank_import_paths_are_absent() -> None:
    old_imports: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in LEGACY_RERANK_IMPORTS:
                        old_imports.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module in LEGACY_RERANK_IMPORTS:
                old_imports.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.module}")

    assert old_imports == []


def test_dense_prefixes_do_not_cross_resolver_and_factory_as_loose_fields() -> None:
    matches: list[str] = []
    for path in NO_LOOSE_DENSE_PREFIX_MODULES:
        text = path.read_text(encoding="utf-8")
        for token in ("query_prefix", "passage_prefix"):
            if token in text:
                matches.append(f"{path.relative_to(REPO_ROOT)}:{token}")

    assert matches == []


def test_application_run_retrieval_is_the_use_case_boundary() -> None:
    module_spec = importlib.util.find_spec("graph_memory.application.run_retrieval")
    assert module_spec is not None

    module = importlib.import_module("graph_memory.application.run_retrieval")
    run_retrieval = module.run_retrieval
    request_type = module.RunRetrievalRequest

    signature = inspect.signature(run_retrieval)
    assert list(signature.parameters) == ["request"]

    request_fields = {field.name for field in fields(request_type)}
    assert {"method", "task_inputs", "graphs", "top_k", "dense_runtime", "trainable_runtime"}.issubset(
        request_fields
    )
    assert request_fields.isdisjoint(
        {
            "encoder_model",
            "query_prefix",
            "passage_prefix",
            "checkpoint_path",
            "text_embedding_provider",
            "seed_signal_provider",
            "device",
        }
    )


def test_resolver_input_groups_trainable_runtime_instead_of_loose_checkpoint_fields() -> None:
    from graph_memory.retrieval.requests import RetrievalMethodResolveRequest

    request_fields = {field.name for field in fields(RetrievalMethodResolveRequest)}

    assert "trainable_runtime" in request_fields
    assert request_fields.isdisjoint(
        {"checkpoint_path", "text_embedding_provider", "seed_signal_provider", "device"}
    )


def test_retrieval_execution_runs_built_method_without_resolving_runtime_parameters() -> None:
    from graph_memory.retrieval.execution.service import run_retrieval

    parameter_names = set(inspect.signature(run_retrieval).parameters)

    assert "retrieval_method" in parameter_names
    assert parameter_names.isdisjoint(NO_LOOSE_EXECUTION_PARAMETERS)
