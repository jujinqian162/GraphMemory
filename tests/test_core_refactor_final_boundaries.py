from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "graph_memory"
SOURCE_ROOTS = (PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests")
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
REMOVED_MODULES = {
    "graph_memory.experiment",
    "graph_memory.types",
    "graph_memory.hotpotqa",
    "graph_memory.splits",
    "graph_memory.entities",
    "graph_memory.indexes",
    "graph_memory.learned",
}
REMOVED_PATHS = {
    PACKAGE_ROOT / "experiment.py",
    PACKAGE_ROOT / "types.py",
    PACKAGE_ROOT / "hotpotqa.py",
    PACKAGE_ROOT / "splits.py",
    PACKAGE_ROOT / "entities.py",
    PACKAGE_ROOT / "indexes",
    PACKAGE_ROOT / "learned",
}
ROOT_PORT_IMPORTS = {
    "io.py": {"graph_memory.infrastructure.io"},
    "observability.py": {
        "graph_memory.infrastructure.run_summary",
        "graph_memory.infrastructure.runtime_environment",
    },
    "retrieval_registry.py": {"graph_memory.registry.projections"},
    "training_config.py": {"graph_memory.config.training_compat"},
}
ROOT_WORKFLOW_PORT_MODULES = {
    "graph_memory.io",
    "graph_memory.observability",
    "graph_memory.retrieval_registry",
    "graph_memory.training_config",
}
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


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.lineno, node.module))
    return imports


def _imports_removed_module(imported: str) -> bool:
    return any(imported == module or imported.startswith(f"{module}.") for module in REMOVED_MODULES)


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_obsolete_compatibility_paths_are_removed() -> None:
    existing_paths = [str(path.relative_to(REPO_ROOT)) for path in sorted(REMOVED_PATHS) if path.exists()]
    importable_modules = sorted(module for module in REMOVED_MODULES if _has_module(module))

    assert existing_paths == []
    assert importable_modules == []


def test_unreachable_refactor_leftovers_are_removed() -> None:
    import graph_memory.config.training_compat as training_compat
    import graph_memory.infrastructure.io as infrastructure_io
    import graph_memory.io as root_io
    import graph_memory.registry.retrieval_builders as retrieval_builders
    import graph_memory.registry.training as training_registry
    import graph_memory.retrieval.contracts as retrieval_contracts
    import graph_memory.retrieval.methods.graph_rerank.components as rerank_components
    import graph_memory.training_config as root_training_config
    import scripts.build_train_pairs as build_train_pairs
    import scripts.evaluate_retrieval as evaluate_retrieval
    import scripts.run_retrieval as run_retrieval
    import scripts.train_method as train_method
    import scripts.workflow.manifest as workflow_manifest
    import scripts.workflow.types as workflow_types
    from graph_memory.config.codec import JsonConfigCodec
    from graph_memory.config.loader import ConfigLoader
    from scripts.workflow.resume import WorkflowStatusKey

    removed_training_helpers = {
        "EncoderConfig",
        "ModelConfigValues",
        "encoder_config_from_training_config",
        "model_config_values_from_training_config",
    }
    for module in (training_compat, root_training_config):
        assert removed_training_helpers.isdisjoint(vars(module))

    assert not hasattr(infrastructure_io, "load_config")
    assert not hasattr(root_io, "load_config")
    assert not hasattr(ConfigLoader, "write_resolved")
    assert not hasattr(JsonConfigCodec, "write")

    assert not hasattr(retrieval_builders, "RETRIEVAL_REGISTRY")
    assert not hasattr(retrieval_builders, "RuntimeRetrievalRegistry")
    assert not hasattr(training_registry, "TRAINING_REGISTRY")
    assert not hasattr(retrieval_contracts, "Retriever")

    assert not hasattr(rerank_components, "ComponentName")
    assert not hasattr(rerank_components.InitialScoreComponent(weight=1.0, normalization="minmax"), "component_name")

    assert not hasattr(workflow_types, "RunUnit")
    assert not hasattr(WorkflowStatusKey, "to_manifest_key")
    assert not hasattr(workflow_manifest, "STAGE_ORDER")

    removed_script_helpers = {
        build_train_pairs: {"BuildTrainPairsArgs", "build_parser", "parse_args"},
        evaluate_retrieval: {"build_parser"},
        run_retrieval: {"build_parser", "parse_args"},
        train_method: {"TrainMethodArgs", "build_parser", "parse_args"},
    }
    for module, names in removed_script_helpers.items():
        assert names.isdisjoint(vars(module))


def test_source_imports_do_not_use_removed_compatibility_paths() -> None:
    old_imports: list[str] = []
    for path in _python_files():
        if path == Path(__file__).resolve():
            continue
        for lineno, imported in _imported_modules(path):
            if _imports_removed_module(imported):
                old_imports.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}")

    assert old_imports == []


def test_remaining_trainable_and_sampling_records_live_in_owned_domains() -> None:
    from graph_memory.models.graph_retriever.config.records import (
        NodeFeatureConfig,
        TrainableModelConfig,
        TrainableTrainingConfig,
    )
    from graph_memory.models.graph_retriever.internals.contracts import GraphBatch, TrainingBatch
    from graph_memory.training_pairs.config import NegativeSamplingConfig

    assert NegativeSamplingConfig.__module__ == "graph_memory.training_pairs.config"
    assert NodeFeatureConfig.__module__ == "graph_memory.models.graph_retriever.config.records"
    assert TrainableModelConfig.__module__ == "graph_memory.models.graph_retriever.config.records"
    assert TrainableTrainingConfig.__module__ == "graph_memory.models.graph_retriever.config.records"
    assert GraphBatch.__module__ == "graph_memory.models.graph_retriever.internals.contracts"
    assert TrainingBatch.__module__ == "graph_memory.models.graph_retriever.internals.contracts"


def test_root_workflow_integration_ports_import_only_approved_targets() -> None:
    violations: list[str] = []
    for file_name, allowed in ROOT_PORT_IMPORTS.items():
        path = PACKAGE_ROOT / file_name
        assert path.exists()
        for lineno, imported in _imported_modules(path):
            if not (imported.startswith("graph_memory.") or imported.startswith("scripts.")):
                continue
            if imported not in allowed:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}")

    assert violations == []


def test_core_package_dependency_direction_is_enforced() -> None:
    violations: list[str] = []
    for root, forbidden_prefixes in FORBIDDEN_PACKAGE_IMPORTS.items():
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for lineno, imported in _imported_modules(path):
                for forbidden in forbidden_prefixes:
                    if imported == forbidden or imported.startswith(f"{forbidden}."):
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}")

    assert violations == []


def test_domain_packages_do_not_import_root_workflow_integration_ports() -> None:
    violations: list[str] = []
    for root in DOMAIN_PACKAGE_ROOTS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for lineno, imported in _imported_modules(path):
                if imported in ROOT_WORKFLOW_PORT_MODULES:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{imported}")

    assert violations == []
