# scripts/test_unificacion_quick.py
"""
Smoke test rápido de UnificacionStrategy.
- Crea (si no existen) datasets de ejemplo en datasets/2024-1 y 2024-2
- Ejecuta acumulado() y verifica que se genera historico/unificado.parquet
"""
from pathlib import Path
import io
import pandas as pd

from backend.src.neurocampus.data.strategies.unificacion import UnificacionStrategy

ROOT = Path(".")
DATASETS = ROOT / "datasets"
(DATASETS / "2024-1").mkdir(parents=True, exist_ok=True)
(DATASETS / "2024-2").mkdir(parents=True, exist_ok=True)

# Semillas de ejemplo
for periodo in ("2024-1", "2024-2"):
    df = pd.DataFrame({
        "codigo_materia": ["MAT101", "MAT101"],
        "grupo": ["A", "A"],
        "cedula_profesor": ["123", "123"],
        "pregunta_1": [4, 5],
    })
    out = DATASETS / periodo / "data.csv"
    if not out.exists():
        df.to_csv(out, index=False)

# Ejecutar unificación
u = UnificacionStrategy(base_uri="localfs://.")
out_uri, meta = u.acumulado()
print("[OK] generado:", out_uri, meta)

assert Path(out_uri).exists(), "No se generó el parquet unificado"
