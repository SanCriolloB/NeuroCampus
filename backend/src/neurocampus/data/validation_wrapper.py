# Re-export del wrapper real para compat con tests/legacy imports.
from __future__ import annotations

from neurocampus.validation.validation_wrapper import run_validations

__all__ = ["run_validations"]
