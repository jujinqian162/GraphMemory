"""Repository-owned identity for the registered ISETrace source corpus."""

from __future__ import annotations

ISETRACE_REVISION = "e40e04d41c04e4eb4bae181ebdd41b61c688081b"
# Frozen natural-query ownership weights for the registered v7 corpus. They
# resolve to 2,692/393/981 for the 4,066-query server corpus while allowing a
# newly content-addressed natural source to receive its own deterministic plan.
ISETRACE_NATURAL_SPLIT_WEIGHTS = {
    "train": 2692,
    "dev": 393,
    "test": 981,
}

__all__ = ["ISETRACE_NATURAL_SPLIT_WEIGHTS", "ISETRACE_REVISION"]
