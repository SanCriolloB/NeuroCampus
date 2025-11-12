
# README — Uso rápido modelos RBM (Día 11)

**Última actualización:** 2025-11-05 03:26

Este documento resume cómo **generar datos**, **entrenar modelos** y **auditar** resultados en NeuroCampus después de los cambios del Día 11.

---

## 1) Requisitos previos

- Tener el entorno listo con los requirements de `backend/requirements.txt`.
- Estar en la raíz del repo (donde está `configs/rbm_audit.yaml`).

```bash
python -m pip install -r backend/requirements.txt
```

> En Windows con venv: activa primero `.\.venv\Scriptsctivate`

---

## 2) Generar datos sintéticos (opcional pero recomendado)

Crea un dataset simulado coherente con el pipeline (columnas `calif_*`, `p_*`, `sentiment_label_teacher`).

```bash
# Makefile (atajo)
make sim-data

# o directamente con Python (elige .parquet o .csv)
python tools/sim/generate_synthetic.py --n 5000 --out data/simulated/evals_sim_5k.parquet
```

**Salida esperada:** `data/simulated/evals_sim_5k.parquet`

---

## 3) Entrenar modelos de forma directa

### 3.1 RBM Pura (no supervisada)
```bash
PYTHONPATH=backend/src python backend/scripts/train_rbm_pura.py   --data data/labeled/evaluaciones_2025_teacher.parquet   --n_hidden 64 --epochs 5 --cd_k 1 --batch_size 64 --lr 0.01
```
**Artefactos:** `artifacts/jobs/rbm_pura_*/H_sample.npy` y `train_meta.json`

### 3.2 RBM General (cabeza supervisada mínima)
```bash
PYTHONPATH=backend/src python backend/scripts/train_rbm_general.py   --data data/labeled/evaluaciones_2025_teacher.parquet   --target sentiment_label_teacher   --n_hidden 64 --epochs 10 --epochs_rbm 1 --cd_k 1   --batch_size 128 --lr_rbm 0.01 --lr_head 0.01
```
**Artefactos:** `artifacts/jobs/rbm_general_*/` con `rbm.pt`, `head.pt`, `vectorizer.json`, `job_score.json`

> Windows PowerShell: usa `$env:PYTHONPATH="backend/src"` antes del comando Python.

---

## 4) Auditoría k-fold (todos los modelos del YAML)

```bash
make rbm-audit
```

- Lee: `configs/rbm_audit.yaml` (en la **raíz**)
- Escribe: `artifacts/runs/rbm_audit_*/metrics.json`

Alias útiles (en CI/logs):
```bash
make rbm-pura-audit
make rbm-general-audit
make rbm-restringido-audit
```

> **Nota:** Por defecto ejecutan lo mismo que `make rbm-audit` (los 3 modelos). Si más adelante quieres que filtren por modelo, se puede añadir una bandera `--model` o generar un YAML temporal.

---

## 5) Tests

### 5.1 Tests de auditoría (schema/consistencia)
```bash
PYTHONPATH=backend/src pytest -q tests/unit/test_rbm_audit_schema.py
```

### 5.2 Tests de API de RBM General
```bash
PYTHONPATH=backend/src pytest -q tests/unit/test_rbm_general_api.py
```

---

## 6) Rutas importantes

- **Config auditoría:** `configs/rbm_audit.yaml`
- **Script auditor:** `backend/src/neurocampus/models/audit_kfold.py`
- **Estrategias:** `backend/src/neurocampus/models/strategies/`
  - `modelo_rbm_general.py`
  - `modelo_rbm_restringida.py`
  - `rbm_pura.py`
- **Simulación:** `tools/sim/generate_synthetic.py`
- **Entrenamiento directo:** `backend/scripts/train_rbm_pura.py`, `backend/scripts/train_rbm_general.py`
- **Artefactos:** `artifacts/runs/` y `artifacts/jobs/`

---

## 7) Errores comunes y soluciones

- **`ModuleNotFoundError: numpy` al usar `make rbm-audit`:** el target intenta instalar los requirements en tu intérprete; verifica que `backend/requirements.txt` instale sin errores.
- **`mat1 and mat2 shapes cannot be multiplied ...`**: el auditor intenta **reparar shapes**; asegúrate de que tus features sean **numéricas** y que el modelo reciba `visible_units` correcto.
- **`ValueError: Mix of label input types (string and number)`**: el auditor fuerza etiquetas a enteros; revisa que el target se detecte y que no haya strings raros en `y`.

---

## 8) Commits sugeridos (resumen Día 11)

```bash
git add backend/src/neurocampus/models/audit_kfold.py         backend/src/neurocampus/models/strategies/modelo_rbm_general.py         backend/src/neurocampus/models/strategies/modelo_rbm_restringida.py         backend/src/neurocampus/models/strategies/rbm_pura.py         backend/scripts/train_rbm_pura.py         backend/scripts/train_rbm_general.py         tools/sim/generate_synthetic.py         configs/rbm_audit.yaml         tests/unit/test_rbm_general_api.py         Makefile         docs/modelos/README_D11_uso_rapido.md
git commit -m "D11: Uso rápido RBM — simulación de datos, entrenamiento directo y auditoría k-fold (doc + scripts + tests + make)"
```
