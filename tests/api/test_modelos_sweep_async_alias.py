from __future__ import annotations

"""Tests de compatibilidad para sweeps asíncronos.

Este archivo cubre un caso reportado en el checklist P2:

- ``POST /modelos/entrenar/sweep`` históricamente espera el campo ``modelos``.
- El endpoint síncrono determinístico (``POST /modelos/sweep``) usa ``models``.

Para evitar confusión y errores 422 en clientes, el schema del endpoint
asíncrono debe aceptar **ambos** nombres: ``modelos`` y ``models``.

Nota importante:
- En estos tests se *parchea* la función interna ``_run_sweep_training``
  para evitar entrenamiento real y acceso a artifacts/datasets.
"""

import pytest


def _patch_fast_sweep(monkeypatch):
    """Parchea el runner del sweep para hacerlo instantáneo.

    El endpoint ``/modelos/entrenar/sweep`` agenda un ``BackgroundTask``.
    En ``TestClient`` dicho task se ejecuta inmediatamente al finalizar
    la respuesta.  Este parche reemplaza la lógica pesada por un update
    simple del estado en memoria (``_ESTADOS``).

    Se mantiene el contrato básico de ``EstadoResponse`` (status/progress).
    """
    from neurocampus.app.routers import modelos as m

    def _fake_run_sweep_training(sweep_id: str, req) -> None:  # pragma: no cover
        st = m._ESTADOS.get(sweep_id)
        if not isinstance(st, dict):
            st = {"job_id": sweep_id, "job_type": "sweep"}

        # Mantener params intacto (incluye `modelos` ya parseado por Pydantic)
        st.update(
            {
                "status": "completed",
                "progress": 1.0,
                "elapsed_s": 0.0,
                "job_type": "sweep",
                # Campos opcionales de sweep
                "sweep_best_overall": None,
                "sweep_best_by_model": None,
            }
        )
        m._ESTADOS[sweep_id] = st

    monkeypatch.setattr(m, "_run_sweep_training", _fake_run_sweep_training)


@pytest.mark.parametrize(
    "field_name",
    [
        "modelos",  # legacy
        "models",  # alias (nuevo)
    ],
)
def test_entrenar_sweep_accepts_model_list_aliases(client, monkeypatch, field_name: str):
    """El endpoint async debe aceptar tanto `modelos` como `models`.

    Criterio de aceptación:
    - El request es válido (HTTP 200)
    - El estado del job refleja la lista en ``params.modelos``

    Esto evita que el frontend (que usa `models` en /modelos/sweep)
    falle al integrarse con /modelos/entrenar/sweep.
    """
    _patch_fast_sweep(monkeypatch)

    payload = {
        "dataset_id": "ds_dummy",
        "family": "sentiment_desempeno",
        field_name: ["rbm_general", "dbm_manual"],
        # Mantener low-impact: el runner está parcheado, pero el schema debe validar.
        "auto_promote_champion": False,
        "max_total_runs": 2,
    }

    r = client.post("/modelos/entrenar/sweep", json=payload)
    assert r.status_code == 200, r.text
    sweep_id = r.json().get("sweep_id")
    assert sweep_id

    st = client.get(f"/modelos/estado/{sweep_id}")
    assert st.status_code == 200, st.text

    state = st.json()
    assert state["status"] == "completed"
    assert state.get("job_type") == "sweep"

    params = state.get("params") or {}
    assert params.get("modelos") == ["rbm_general", "dbm_manual"]
