# backend/src/neurocampus/models/__init__.py
"""
Paquete de modelos de dominio (independientes de frameworks).
Exporta RBM/BM manuales para uso por estrategias y jobs.
"""

from .rbm_manual import RestrictedBoltzmannMachine
from .bm_manual import BoltzmannMachine

# utilidades (opcionales, pero útiles para tests/depuración)
from .utils_boltzmann import (
    sigmoid,
    bernoulli_sample,
    binarize,
    energy_rbm,
    energy_bm,
)

__all__ = [
    "RestrictedBoltzmannMachine",
    "BoltzmannMachine",
    "sigmoid",
    "bernoulli_sample",
    "binarize",
    "energy_rbm",
    "energy_bm",
]
