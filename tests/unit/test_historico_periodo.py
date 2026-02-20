"""tests.unit.test_historico_periodo

Pruebas de contrato para la unificación histórica:

Fase A1 del plan de trabajo del Dashboard:
- El histórico (historico/unificado*.parquet) debe contener una columna `periodo`
  consistente por fila.

Esta prueba valida la responsabilidad de `UnificacionStrategy._ensure_periodo`:
- Si un dataset no trae `periodo`, se debe inyectar usando el `dataset_id`.
- Si `periodo` existe pero viene vacío/None/NaN, se debe rellenar con `dataset_id`.

Importante:
- El Dashboard (fases posteriores) solo consumirá histórico, así que la
  consistencia de `periodo` es un pre-requisito para filtros y agregaciones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neurocampus.data.strategies.unificacion import UnificacionStrategy


# Nota: el proyecto declara `pyarrow` como dependencia del backend.
# Aun así, este importorskip permite que la suite de tests degrade con gracia
# en entornos donde todavía no se han instalado los extras de Parquet.
pytest.importorskip("pyarrow")


def _write_dataset(path: Path, df: pd.DataFrame) -> None:
    """Escribe un parquet en disco de forma determinística para tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_unificacion_inyecta_y_rellena_periodo(tmp_path: Path) -> None:
    """El histórico acumulado debe garantizar `periodo` por fila.

    Escenarios cubiertos:
    1) Dataset 2024-1 sin columna `periodo`.
    2) Dataset 2024-2 con columna `periodo` pero valores vacíos/None.

    Resultado esperado:
    - `historico/unificado.parquet` contiene `periodo`.
    - No existen valores vacíos/None/NaN en `periodo`.
    - Las filas provenientes de cada dataset quedan etiquetadas con su dataset_id.
    """

    # Estructura mínima para que UnificacionStrategy NO haga fallback al repo real.
    # (Su constructor verifica existencia de 'data/' y 'datasets/' en base_uri)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "datasets").mkdir(parents=True)

    # Dataset 2024-1 (SIN columna periodo)
    df_2024_1 = pd.DataFrame(
        {
            "codigo_materia": ["MAT101", "MAT101"],
            "grupo": ["A", "B"],
            "cedula_profesor": ["100", "200"],
            "docente": ["ALICE", "BOB"],
            "asignatura": ["CALCULO", "CALCULO"],
            "rating": [40.0, 35.0],
        }
    )

    # Dataset 2024-2 (CON columna periodo, pero inválida)
    df_2024_2 = pd.DataFrame(
        {
            "periodo": [None, "   "],
            "codigo_materia": ["FIS201", "FIS201"],
            "grupo": ["A", "A"],
            "cedula_profesor": ["300", "300"],
            "docente": ["CARLOS", "CARLOS"],
            "asignatura": ["FISICA", "FISICA"],
            "rating": [38.0, 42.0],
        }
    )

    _write_dataset(tmp_path / "datasets" / "2024-1.parquet", df_2024_1)
    _write_dataset(tmp_path / "datasets" / "2024-2.parquet", df_2024_2)

    # Ejecutar unificación sobre el FS temporal
    strat = UnificacionStrategy(base_uri=f"localfs://{tmp_path}")
    out_uri, meta = strat.acumulado()

    assert out_uri == "historico/unificado.parquet"
    assert meta["datasets"] == ["2024-1", "2024-2"]

    out_path = tmp_path / "historico" / "unificado.parquet"
    assert out_path.exists(), "No se generó el histórico unificado"

    out = pd.read_parquet(out_path)

    # Contrato: columna periodo siempre presente
    assert "periodo" in out.columns

    # Contrato: periodo sin vacíos/None/NaN
    assert out["periodo"].isna().sum() == 0
    assert out["periodo"].astype(str).str.strip().ne("").all()

    # Contrato: cada dataset queda etiquetado con su dataset_id
    assert out.loc[out["codigo_materia"].eq("MAT101"), "periodo"].unique().tolist() == ["2024-1"]
    assert out.loc[out["codigo_materia"].eq("FIS201"), "periodo"].unique().tolist() == ["2024-2"]
