# P2 — Predicciones (Backend) — Runbook + Contratos

Este documento describe la fase P2 del backend enfocada en **resolver y validar el predictor bundle** (P2.2)
usando artifacts generados en P0/P1 (feature-pack, runs, champions).

> Nota: en P2.2 el endpoint `/predicciones/predict` **no ejecuta inferencia real**; solo resuelve el `run_id`,
> valida el bundle y retorna metadata. La inferencia real está prevista para P2.4+.

## Conceptos

### Rutas lógicas vs rutas físicas
- **Ruta lógica**: `artifacts/...` (portable entre ambientes).
- **Ruta física**: carpeta real en disco, controlada por `NC_ARTIFACTS_DIR`.

El backend debe exponer/aceptar preferentemente rutas lógicas. Para resolverlas usa:

- `neurocampus.utils.paths.abs_artifact_path(ref)`
- `neurocampus.utils.paths.rel_artifact_path(path)`

### Entidades
- **Feature-pack**: artifacts/features/<dataset_id>/
- **Run**: artifacts/runs/<run_id>/
- **Champion**: artifacts/champions/<family>/<dataset_id>/champion.json (nuevo)
  - fallback: artifacts/champions/<dataset_id>/champion.json (legacy/mirror)

## Layout esperado de artifacts

### Feature-pack
Directorio: `artifacts/features/<dataset_id>/`

Archivos (mínimos):
- `train_matrix.parquet`
- `meta.json`

Opcional (pair-level):
- `pair_matrix.parquet`
- `pair_meta.json`

### Run
Directorio: `artifacts/runs/<run_id>/`

Archivos (mínimos para auditoría):
- `metrics.json`
- `history.json`

Archivos (recomendados P2 para inferencia):
- `model.bin` (o nombre equivalente: pesos serializados)
- `predictor.json` (metadatos de inferencia / configuración)
- `preprocess.json` (mapeos/normalizaciones si aplican)

> Nota: en P2 se formaliza qué debe existir para inferencia (ver sección “Contrato de inferencia”).

### Champion
Archivo preferido:
- `artifacts/champions/<family>/<dataset_id>/champion.json`

Campos relevantes P2:
- `source_run_id` (run fuente de verdad)
- `model_name`
- `metrics` (auditoría offline)
- `paths.run_dir`, `paths.run_metrics` (refs lógicas)
- `score` (tier/value para comparar)

## Contrato de inferencia (P2)

La inferencia se soporta por **run_id** o por **champion**.

### Resolución por run_id
1) Resolver directorio:
- `artifacts/runs/<run_id>/`
2) Leer:
- `metrics.json` (para conocer `model_name`, `dataset_id`, `task_type`, etc.)
- `model.bin` (pesos/estado del modelo)
- `predictor.json` (cómo inferir: input level, target mode, thresholds, etc.)

### Resolución por champion
1) Resolver `champion.json` (layout nuevo y fallback legacy).
2) Leer `source_run_id`
3) Continuar como run_id

## Endpoints implementados (P2.2)

En P2.2 hay dos endpoints estables:
- `GET /predicciones/health`
- `POST /predicciones/predict` (**resolve/validate**, sin inferencia)

### GET /predicciones/health
Devuelve estado y la ruta efectiva de artifacts.

Ejemplo:
```bash
curl -s "http://127.0.0.1:8000/predicciones/health"
```

### POST /predicciones/predict
**Objetivo:** resolver `run_id` y validar el bundle mínimo de inferencia.

Request (modo run_id directo):
```json
{ "run_id": "2025-1__rbm_general__20260216T003747Z__b56428df" }
```

Request (modo champion):
```json
{ "use_champion": true, "dataset_id": "2025-1", "family": "sentiment_desempeno" }
```

#### Códigos de respuesta esperados
- `200`: bundle resuelto y válido.
- `404`: no existe el champion o no existe el bundle del run (por ejemplo faltan `predictor.json`/`model.bin`).
- `422`: request inválido o predictor “no listo”.
  - Ejemplos:
    - `champion.json` existe pero no incluye `source_run_id`.
    - `model.bin` existe pero es placeholder/no listo.
- `500`: solo para errores inesperados.

## Variables de entorno (P2)

- `NC_ARTIFACTS_DIR`:
  - Ruta física base de artifacts.
  - Si no existe: default `<repo>/artifacts`.

- `NC_PROJECT_ROOT`:
  - Ruta física del repo (ayuda a resolver paths relativos si corres desde otro cwd).

## Verificación (manual)

### 1) Levantar backend
Ejemplo (ajusta según tu entorno):
```bash
uvicorn neurocampus.app.main:app --reload --app-dir backend/src
```

### 2) Health
```bash
curl -s "http://127.0.0.1:8000/predicciones/health"
```

### 3) Resolver por run_id
```bash
curl -s -X POST "http://127.0.0.1:8000/predicciones/predict" \
  -H "Content-Type: application/json" \
  -d '{ "run_id": "<run_id>" }'
```

### 4) Resolver por champion
Primero promueve un run a champion (P0/P1):
```bash
curl -s -X POST "http://127.0.0.1:8000/modelos/champion/promote" \
  -H "Content-Type: application/json" \
  -d '{ "dataset_id": "<dataset_id>", "family": "<family>", "run_id": "<run_id>" }'
```

Luego resuelve:
```bash
curl -s -X POST "http://127.0.0.1:8000/predicciones/predict" \
  -H "Content-Type: application/json" \
  -d '{ "use_champion": true, "dataset_id": "<dataset_id>", "family": "<family>" }'
```

### 5) Verificar bundle en disco
```bash
ls -la artifacts/runs/<run_id>
cat artifacts/runs/<run_id>/predictor.json
cat artifacts/runs/<run_id>/preprocess.json
```

> Si al resolver por champion obtienes 404 indicando que faltan `predictor.json`/`model.bin`,
> es muy probable que el champion esté apuntando a un run “viejo” o incompleto.
> La solución recomendada es promover como champion un run nuevo que sí tenga bundle completo.
