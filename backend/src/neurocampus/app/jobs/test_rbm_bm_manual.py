# backend/src/neurocampus/app/jobs/test_rbm_bm_manual.py
"""
Prueba rápida de compatibilidad de RBM y BM manuales con los .parquet generados.
- Usa únicamente columnas numéricas
- Entrenamiento corto para validar que no rompe
"""

import pandas as pd
import numpy as np

from neurocampus.models.rbm_manual import RestrictedBoltzmannMachine
from neurocampus.models.bm_manual import BoltzmannMachine

DATASET = "data/prep_auto/dataset_ejemplo.parquet"  # puedes cambiarlo a Evaluacion.parquet

def main():
    df = pd.read_parquet(DATASET)
    # Solo numéricos, NaN->0
    X = df.select_dtypes(include=[np.number]).fillna(0).to_numpy(dtype=np.float32)

    print(f"[Test] Datos cargados: {X.shape} (solo numéricos)")

    # Escalado opcional (0-1) si tus features no están ya en rango razonable
    # X = (X - X.min(axis=0, keepdims=True)) / (X.ptp(axis=0, keepdims=True) + 1e-9)

    # RBM
    rbm = RestrictedBoltzmannMachine(
        n_visible=X.shape[1],
        n_hidden=16,
        learning_rate=0.05,
        seed=123,
        l2=0.0,
        clip_grad=1.0,
        binarize_input=False,  # True si quieres RBM estrictamente binaria
    )
    rbm.fit(X, epochs=5, batch_size=64, verbose=1)
    H = rbm.transform_hidden(X[:128])
    print(f"[Test][RBM] Hidden batch shape: {H.shape}")

    # BM
    bm = BoltzmannMachine(
        n_visible=X.shape[1],
        n_hidden=16,
        learning_rate=0.05,
        seed=123,
        l2=0.0,
        clip_grad=1.0,
        binarize_input=False,
    )
    bm.fit(X, epochs=5, batch_size=64, verbose=1)
    H2 = bm.transform_hidden(X[:128])
    print(f"[Test][BM] Hidden batch shape: {H2.shape}")

    print("[Test] Entrenamiento manual de RBM y BM completado correctamente.")

if __name__ == "__main__":
    main()
