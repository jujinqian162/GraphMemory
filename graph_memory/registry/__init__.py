from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_memory.registry.app import Registry


def __getattr__(name: str) -> Any:
    if name == "Registry":
        from graph_memory.registry.app import Registry

        return Registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Registry"]
