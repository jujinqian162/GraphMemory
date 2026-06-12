from __future__ import annotations

import importlib.util
from pathlib import Path


PRODUCTION_ROOTS = (Path("graph_memory"), Path("scripts"))
LEGACY_RETRIEVAL_BUILD_MODULES = {
    "graph_memory.retrieval.factory": Path("graph_memory/retrieval/factory.py"),
    "graph_memory.retrieval.resolver": Path("graph_memory/retrieval/resolver.py"),
}


def test_training_dict_slicing_helpers_are_removed() -> None:
    helper_names = (
        "negative_sampling_config_from_training_config",
        "trainable_training_config_from_training_config",
    )

    offenders = [
        f"{path}:{helper}"
        for path in _production_python_files()
        for helper in helper_names
        if helper in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_model_config_package_does_not_export_training_config_compat_helpers() -> None:
    import graph_memory.models.graph_retriever.config as model_config_api

    helper_names = (
        "load_trainable_training_config",
        "resolve_trainable_training_config",
        "encoder_config_from_training_config",
        "model_config_values_from_training_config",
        "negative_sampling_config_from_training_config",
        "trainable_training_config_from_training_config",
        "device_from_training_config",
    )

    assert [helper for helper in helper_names if hasattr(model_config_api, helper)] == []


def test_builder_id_is_removed() -> None:
    offenders = [
        str(path)
        for path in _production_python_files()
        if "builder_id" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_public_method_string_dispatch_is_bounded_to_current_registry() -> None:
    allowed_paths = {
        Path("graph_memory/registry/methods.py"),
        Path("graph_memory/registry/retrieval.py"),
        Path("graph_memory/registry/retrieval_builders.py"),
        Path("graph_memory/registry/stage_configs.py"),
    }
    patterns = ("method ==", "method in {", "method in (")

    offenders = [
        f"{path}:{pattern}"
        for path in _production_python_files()
        if path not in allowed_paths
        for pattern in patterns
        if pattern in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_depreciate_tags_are_removed_from_production_code() -> None:
    offenders = [
        str(path)
        for path in _production_python_files()
        if "#TAG depreciate" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_legacy_retrieval_build_request_modules_are_removed() -> None:
    existing_paths = [str(path) for path in LEGACY_RETRIEVAL_BUILD_MODULES.values() if path.exists()]
    importable_modules = [
        module_name
        for module_name in LEGACY_RETRIEVAL_BUILD_MODULES
        if importlib.util.find_spec(module_name) is not None
    ]

    assert existing_paths == []
    assert importable_modules == []


def test_runtime_request_module_keeps_only_shared_runtime_objects() -> None:
    source = Path("graph_memory/retrieval/requests.py").read_text(encoding="utf-8")

    assert "class DenseRuntime" in source
    assert "class TrainableGraphRuntime" not in source
    assert "SeedRetrieverBuildRequest" not in source
    assert "RetrievalMethodResolveRequest" not in source
    assert "FlatMethodBuildRequest" not in source
    assert "GraphRerankMethodBuildRequest" not in source
    assert "TrainableGraphMethodBuildRequest" not in source
    assert "MethodBuildRequest" not in source


def _production_python_files() -> list[Path]:
    return [
        path
        for root in PRODUCTION_ROOTS
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
