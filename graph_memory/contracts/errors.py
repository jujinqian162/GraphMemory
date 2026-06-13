from __future__ import annotations


class ContractValidationError(ValueError):
    """Raised when an artifact violates a documented project contract."""


__all__ = ["ContractValidationError"]
