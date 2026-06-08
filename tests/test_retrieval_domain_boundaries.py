from __future__ import annotations

import ast
import importlib.util
import inspect
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
LEGACY_GRAPH_RERANK_EXPORTS = {
    "build_score_debug_record",
    "config_digest",
    "graph_rerank",
    "graph_rerank_with_breakdown",
}
NO_LOOSE_DENSE_PREFIX_MODULES = (
    PACKAGE_ROOT / "retrieval" / "execution" / "service.py",
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


def test_old_graph_rerank_debug_and_convenience_exports_are_absent() -> None:
    from graph_memory.retrieval.methods import graph_rerank

    source = (PACKAGE_ROOT / "retrieval" / "methods" / "graph_rerank" / "engine.py").read_text(encoding="utf-8")
    init_source = (PACKAGE_ROOT / "retrieval" / "methods" / "graph_rerank" / "__init__.py").read_text(encoding="utf-8")

    assert not (PACKAGE_ROOT / "retrieval" / "methods" / "graph_rerank" / "debug.py").exists()
    for name in LEGACY_GRAPH_RERANK_EXPORTS:
        assert not hasattr(graph_rerank, name)
        assert f"def {name}(" not in source
        assert f'"{name}"' not in init_source
        assert f"import {name}" not in init_source
    assert "include_score_breakdown" not in source


def test_dense_prefixes_do_not_cross_resolver_and_factory_as_loose_fields() -> None:
    matches: list[str] = []
    for path in NO_LOOSE_DENSE_PREFIX_MODULES:
        text = path.read_text(encoding="utf-8")
        for token in ("query_prefix", "passage_prefix"):
            if token in text:
                matches.append(f"{path.relative_to(REPO_ROOT)}:{token}")

    assert matches == []


def test_application_run_retrieval_layer_is_removed_after_stage_migration() -> None:
    assert not (PACKAGE_ROOT / "application").exists()
    assert not _has_module("graph_memory.application")
    assert not _has_module("graph_memory.application.run_retrieval")


def test_trainable_retrieval_uses_unified_retrieval_script_entry() -> None:
    assert not (REPO_ROOT / "scripts" / "run_trainable_retrieval.py").exists()
    assert not _has_module("scripts.run_trainable_retrieval")

    import scripts.run_retrieval as run_retrieval_script

    action = run_retrieval_script.build_parser()._option_string_actions["--method"]
    assert action.choices is not None
    assert "dense_rgcn_graph_retriever" in action.choices


def test_runtime_request_module_does_not_reintroduce_trainable_or_stage_request_objects() -> None:
    source = (PACKAGE_ROOT / "retrieval" / "requests.py").read_text(encoding="utf-8")

    assert "class DenseRuntime" in source
    assert "class TrainableGraphRuntime" not in source
    assert "RunRetrievalRequest" not in source
    assert "RetrievalMethodResolveRequest" not in source


def test_retrieval_execution_runs_built_method_without_resolving_runtime_parameters() -> None:
    from graph_memory.retrieval.execution.service import run_retrieval

    parameter_names = set(inspect.signature(run_retrieval).parameters)

    assert "retrieval_method" in parameter_names
    assert parameter_names.isdisjoint(NO_LOOSE_EXECUTION_PARAMETERS)


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False
