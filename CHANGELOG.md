# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **Predicciones (P2.2)**: endpoints `/predicciones/health` y `/predicciones/predict` (modo *resolve/validate* del predictor bundle; sin inferencia real).

### Changed
- **Rutas de predicción**: se reemplazan endpoints legacy `/prediccion/*` por `/predicciones/*` (P2.2).
- **Champions**: nueva convención `artifacts/champions/<family>/<dataset_id>/champion.json` con `source_run_id` (reemplaza `CHAMPION_WITH_TEXT` y `artifacts/champions/with_text/current/`).

### Fixed
- **/predicciones/predict**: cuando el bundle del run no existe/incompleto, responde `404` (en lugar de `500`).
- **Tests**: `NC_ARTIFACTS_DIR` se fuerza a un directorio temporal en `tests/conftest.py` para ejecuciones deterministas.

## [0.7.0] - 2025-10-14 — Día 7 (Release Closing)
### Added
- **Pipeline NLP (Teacher, BETO)**: `cmd_preprocesar_beto` con `--beto-mode (probs|simple)`, `--threshold`, `--margin`, `--neu-min`, `--min-tokens`, `--batch-size`.
  - Genera columnas: `_texto_clean`, `_texto_lemmas`, `token_count`, `has_text`, `p_neg`, `p_neu`, `p_pos`, `sentiment_label_teacher`, `sentiment_conf`, `accepted_by_teacher`.
  - Escribe meta: `*.meta.json` con parámetros y tasa de aceptación.
- **Entrenamiento Student**: `neurocampus.models.train_rbm` con soporte:
  - `--type (general|restringida)`, `--use-text-probs`, `--scale-mode (minmax|standard|scale_0_5)`.
  - Hiperparámetros: `--n-hidden`, `--epochs`, `--cd-k`, `--epochs-rbm`, `--batch-size`, `--lr-rbm`, `--lr-head`, `--seed`.
  - Persistencia en `artifacts/jobs/<JOB_ID>`: `vectorizer.json`, `rbm.pt`, `head.pt`, `job_meta.json`, `metrics.json`.
- **Promoción de champion (legacy 0.7.0)**: convención `artifacts/champions/with_text/current/` (binarios + `CHAMPION.json`). (Reemplazado en P2.2 por `artifacts/champions/<family>/<dataset_id>/champion.json`.)
- **Endpoint de predicción (legacy 0.7.0)**: reglas costo-sensibles en *facade* (prioriza `pos` si `p_pos≥0.55`, `neg` si `p_neg≥0.35` o `p_neg−p_neu≥0.05`, si no `neu`). (En P2.2, `/predicciones/predict` opera en modo *resolve/validate*; inferencia real prevista para P2.4+.)
- **Reporte agregado**: `cmd_score_docente` para estimar `prob_bueno_pct` por (`codigo materia`, `grupo`). Intervalo Jeffreys y score combinado (sentimiento + calificaciones).
- **Documentación**: `README.md`, `Preprocesamiento.md`, `Entrenamiento.md`, `Inferencia_API.md`, `Reporte_Docente.md`.
- **Dummies versionables**: guía para `examples/reports/` y artefactos ignorados en `artifacts/`.

### Changed
- **Cargador de dataset**: `cmd_cargar_dataset` ahora detecta preguntas con **espacios o guion bajo** (`pregunta 1` / `pregunta_1`) y mapea a `calif_1..calif_10`. Soporta `--meta-list` para conservar metadatos existentes.
- **Router de predicción**: parsing robusto de `pregunta_#` con espacio o `_` y normalización del comentario.
- **RBM General/Restringida**: mejoras menores de estabilidad y vectorización (escala/normalización alineada con `vectorizer.json`).

### Fixed
- **Argparse** en `cmd_preprocesar_beto`: corrección del nombre de argumento `--beto-mode` (bug de `Namespace`).
- **Windows Git Bash**: instrucción `printf` con comillas simples para evitar `event not found` al generar `.gitignore` con `!.gitkeep`.
- **Validación FastAPI**: documentación del body con clave `input` para evitar 422.

### Deprecated
- **Teacher simulado**: reemplazado por versión con `transformers` (BETO/robertuito). Mantener solo para pruebas puntuales.

### Migration notes
- **API online**: enviar `{"input": {...}}` (antes algunos clientes enviaban body plano).
- **Datasets**: re-ejecutar `cmd_preprocesar_beto` para obtener `has_text` y `accepted_by_teacher` antes de entrenar.
- **Campeón**: copiar el mejor `JOB_ID` a `artifacts/champions/with_text/current/` para habilitar inferencia.
- **Git**: añadir `.gitignore` para `__pycache__`, `node_modules`, `artifacts/*`, `data/**/*.parquet` y mantener solo `examples/` versionables.

## [0.6.0] - 2025-10-13 — Integración BETO inicial y entrenamiento RBM
### Added
- `teacher_labeling.py` con soporte de `transformers` (BETO/robertuito) y etiquetas `neg/neu/pos`.
- Primera versión de `train_rbm` (general/restringida) y guardado de artefactos.

### Changed
- Normalización de columnas de texto y limpieza/lematización base.

## [0.5.0] - 2025-10-12 — Estandarización de datasets
### Added
- `cmd_cargar_dataset` con heurística numérica y mapeo a `calif_#`.
- Soporte de `--meta-list` para conservar columnas (`codigo_materia`, `docente`, `grupo`, `periodo`).

### Fixed
- Filtros para evitar tomar columnas numéricas ajenas a preguntas (IDs, etc.).

## [0.4.0] - 2025-10-11 — API base
### Added
- Backend FastAPI con routers legacy (`/prediccion/online`, `/prediccion/batch`). (Reemplazados en P2.2 por `/predicciones/health` y `/predicciones/predict`.)

## [0.1.0] - 2025-10-07 — Bootstrap
### Added
- Estructura inicial del repositorio (backend + frontend), CI local mínima, mockups.

---

## Notas
- Fechas aproximadas por hitos del proyecto **Día 1–7**.
- Si cambian los contratos (endpoints, columnas), actualizar este changelog junto con la documentación.
