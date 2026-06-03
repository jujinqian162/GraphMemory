from __future__ import annotations

import platform
import sys


def collect_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


__all__ = ["collect_environment"]
