# backend/tests/unit/test_datos_dashboard_helpers.py
import pandas as pd

from neurocampus.data.datos_dashboard import (
    build_dataset_resumen,
    build_sentimientos_resumen,
)


def test_build_dataset_resumen_basic():
    df = pd.DataFrame(
        {
            "periodo": ["2024-1", "2024-1", "2024-2"],
            "docente": ["Alice", "Bob", "Alice"],
            "asignatura": ["MAT101", "MAT101", "FIS201"],
            "nota": [4.5, 3.8, 4.0],
        }
    )

    res = build_dataset_resumen(df, "demo_ds")

    assert res.dataset_id == "demo_ds"
    assert res.n_rows == 3
    assert res.n_cols == 4
    assert set(res.periodos) == {"2024-1", "2024-2"}
    assert res.n_docentes == 2
    assert res.n_asignaturas == 2

    names = {c.name for c in res.columns}
    assert {"periodo", "docente", "asignatura", "nota"}.issubset(names)


def test_build_sentimientos_resumen_basic():
    df = pd.DataFrame(
        {
            "docente": ["Alice", "Alice", "Bob", "Bob"],
            "asignatura": ["MAT101", "MAT101", "MAT101", "FIS201"],
            "comentario": ["x", "y", "z", "w"],
            "sentiment_label_teacher": ["pos", "neu", "neg", "pos"],
        }
    )

    res = build_sentimientos_resumen(df, "demo_ds")

    assert res.dataset_id == "demo_ds"
    assert res.total_comentarios == 4

    # global
    by_label = {b.label: b.count for b in res.global_counts}
    assert by_label["pos"] == 2
    assert by_label["neu"] == 1
    assert by_label["neg"] == 1

    # por docente
    docentes = {g.group: {b.label: b.count for b in g.counts} for g in res.por_docente}
    assert docentes["Alice"]["pos"] == 1
    assert docentes["Alice"]["neu"] == 1
    assert docentes["Bob"]["neg"] == 1
    assert docentes["Bob"]["pos"] == 1
