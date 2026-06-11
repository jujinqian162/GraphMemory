from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import replace
from types import ModuleType

import pytest


def _projection_module() -> ModuleType:
    try:
        import graph_memory.registry.projections as projections
    except ModuleNotFoundError:
        pytest.fail("graph_memory.registry.projections must own compatibility projection helpers.")
    return projections


def test_catalog_and_root_registry_project_registry_metadata_source() -> None:
    import graph_memory.registry.retrieval as registry_retrieval
    import graph_memory.retrieval.catalog as catalog
    import graph_memory.retrieval_registry as root_registry
    from graph_memory.registry import Registry

    projections = _projection_module()
    metadata = getattr(registry_retrieval, "RETRIEVAL_METHOD_METADATA", None)

    assert metadata is not None
    assert getattr(Registry.retrieval, "metadata", None) == metadata
    assert set(catalog.METHOD_REGISTRY) == set(metadata)
    assert root_registry.METHOD_REGISTRY == catalog.METHOD_REGISTRY
    assert catalog.get_supported_methods() == tuple(metadata)
    assert root_registry.get_supported_methods() == tuple(metadata)

    for method, source in metadata.items():
        catalog_spec = catalog.get_method_spec(method)
        root_spec = root_registry.get_method_spec(method)
        expected_seed = source.seed_method.value if source.seed_method is not None else None

        assert root_spec == catalog_spec
        assert catalog_spec.name == method
        assert catalog_spec.requires_graphs == source.requires_graphs
        assert catalog_spec.requires_graph_config == source.requires_graph_config
        assert catalog_spec.requires_checkpoint == source.requires_checkpoint
        assert catalog_spec.requires_dense_encoder == source.requires_dense_encoder
        assert catalog_spec.seed_method == expected_seed
        assert catalog_spec.builder_id == projections._legacy_builder_id_for(source)


def test_catalog_and_root_registry_have_no_local_method_tables() -> None:
    import graph_memory.retrieval.catalog as catalog
    import graph_memory.retrieval_registry as root_registry

    catalog_source = inspect.getsource(catalog)
    root_registry_source = inspect.getsource(root_registry)

    assert "RetrievalMethodSpec(" not in catalog_source
    assert "METHOD_REGISTRY: dict" not in catalog_source
    assert "builder_id=\"" not in catalog_source
    assert "from graph_memory.retrieval.catalog import" not in root_registry_source
    assert "from graph_memory.registry.projections import" in root_registry_source


def test_projection_helper_projects_new_registry_metadata_without_catalog_edit() -> None:
    import graph_memory.registry.retrieval as registry_retrieval

    projections = _projection_module()
    metadata = getattr(registry_retrieval, "RETRIEVAL_METHOD_METADATA", None)

    assert metadata is not None
    extended = dict(metadata)
    extended["dense_clone_for_projection"] = replace(
        metadata["dense"],
        name="dense_clone_for_projection",
    )

    projected = projections._project_retrieval_method_registry(extended)

    assert projected["dense_clone_for_projection"].name == "dense_clone_for_projection"
    assert projected["dense_clone_for_projection"].requires_dense_encoder is True
    assert projected["dense_clone_for_projection"].builder_id == "dense"


def test_projection_imports_do_not_force_registry_app_cycle() -> None:
    command = (
        "import graph_memory.retrieval.methods.graph_rerank.engine; "
        "import graph_memory.retrieval.catalog; "
        "import graph_memory.retrieval_registry"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
