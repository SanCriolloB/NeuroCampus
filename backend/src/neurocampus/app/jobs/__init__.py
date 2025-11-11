# backend/src/neurocampus/models/__init__.py
from .rbm_manual import RestrictedBoltzmannMachine
from .bm_manual import BoltzmannMachine

__all__ = ["RestrictedBoltzmannMachine", "BoltzmannMachine"]
