# P2 — Predicciones (Backend) — Runbook + Contratos

Este documento describe la fase P2 del backend: **predicción/inferencia** usando artifacts generados en P0/P1
(feature-pack, runs, champions). No toca frontend.

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

## Endpoints previstos (P2)

> Se implementarán en P2.x. Esta sección es el contrato inicial para integración.

### POST /predicciones/predict
Entrada posible:
- Por run: `{ "run_id": "...", "dataset_id": "...", "input_uri": "...", "data_source": "feature_pack" }`
- Por champion: `{ "family": "...", "dataset_id": "...", "input_uri": "...", "use_champion": true }`

Salida:
- `predictions_uri` o `predictions` (según tamaño; para datasets grandes preferir artifact parquet)
- `run_id` resuelto
- `model_name`, `task_type`
- `summary` (n_rows, columnas output, warnings)

### GET /predicciones/health
Retorna:
- estado del servicio
- ruta artifacts base
- versión del contrato P2

## Variables de entorno (P2)

- `NC_ARTIFACTS_DIR`:
  - Ruta física base de artifacts.
  - Si no existe: default `<repo>/artifacts`.

- `NC_PROJECT_ROOT`:
  - Ruta física del repo (ayuda a resolver paths relativos si corres desde otro cwd).

## Verificación (manual)

### Resolver run_id desde entrenamiento (P0/P1)
1) Entrenar: `POST /modelos/entrenar` => `job_id`
2) Estado: `GET /modelos/estado/<job_id>` => `run_id`

### Verificar artifacts de run
```bash
ls -1 artifacts/runs/<run_id>
