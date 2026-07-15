from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Python 3.10-compatible subset of :class:`enum.StrEnum`."""

    __str__ = str.__str__


__all__ = ["StrEnum"]
