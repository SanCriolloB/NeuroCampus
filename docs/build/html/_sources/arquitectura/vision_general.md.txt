# Visión general de la arquitectura

NeuroCampus está organizado en varias capas bien separadas:

- **Frontend** (React + TypeScript, Vite):
  - Ubicación: `frontend/`.
  - Páginas principales:
    - `DataUpload.tsx` (pestaña Datos),
    - `Models.tsx` (pestaña Modelos),
    - `Prediction.tsx` (pestaña Predicciones),
    - `Dashboard.tsx` (pestaña Dashboard),
    - `AdminCleanup.tsx`, `Jobs.tsx`, etc.
  - Navegación con React Router:
    - Definida en `src/routes/Router.tsx`.
    - Layout principal en `src/layout/MainLayout.tsx` con `Sidebar` y `Topbar`.

- **Backend API** (FastAPI + Python):
  - Ubicación: `backend/src/neurocampus/`.
  - Punto de entrada de la API:
    - `neurocampus/app/main.py`.
  - Routers:
    - `neurocampus/app/routers/datos.py`,
    - `neurocampus/app/routers/modelos.py`,
    - `neurocampus/app/routers/prediccion.py`,
    - `neurocampus/app/routers/jobs.py`,
    - `neurocampus/app/routers/admin_cleanup.py`.
  - Esquemas Pydantic:
    - `neurocampus/app/schemas/datos.py`,
    - `neurocampus/app/schemas/modelos.py`,
    - `neurocampus/app/schemas/prediccion.py`,
    - `neurocampus/app/schemas/jobs.py`.

- **Capa de datos**:
  - Ubicación: `neurocampus/data/`.
  - Componentes principales:
    - Adaptadores (`data/adapters/*.py`) para trabajar con DataFrames y rutas
      de almacenamiento.
    - Cadena de validación y preprocesamiento (`data/chain/*.py`).
    - Facade de datos (`data/facades/datos_facade.py`).
    - Utilidades para el dashboard de la pestaña Datos
      (`data/datos_dashboard.py`).

- **Capa de modelos** (BM, RBM, DBM):
  - Ubicación: `neurocampus/models/`.
  - Modelos base y manuales:
    - `bm_manual.py`, `rbm_manual.py`, `dbm_manual.py`.
  - Estrategias de entrenamiento:
    - `models/strategies/bm_manual_strategy.py`,
    - `models/strategies/rbm_manual_strategy.py`,
    - `models/strategies/dbm_manual_strategy.py`,
    - otros modelos relacionados (`modelo_rbm_general`, `modelo_rbm_restringida`, etc.).
  - Registros y facades:
    - `models/registry.py`,
    - `models/facades/modelos_facade.py`.
  - Utilidades:
    - `models/audit_kfold.py`,
    - `models/hparam_search.py`,
    - `models/utils_boltzmann.py`,
    - plantillas de entrenamiento (`models/templates/plantilla_entrenamiento.py`).

- **Orquestación de jobs**:
  - Ubicación: `neurocampus/app/jobs/`.
  - Comandos para CLI / backend batch:
    - `cmd_preprocesar_beto.py`,
    - `cmd_preprocesar_batch.py`,
    - `cmd_cargar_dataset.py`,
    - `cmd_train_rbm_manual.py`,
    - `cmd_train_dbm_manual.py`,
    - `cmd_eval_confusion.py`,
    - `cmd_score_docente.py`,
    - `cmd_autoretrain.py`.

- **Observabilidad y logging**:
  - Ubicación: `neurocampus/observability/`.
  - Middleware de correlación (`middleware_correlation.py`),
    filtros de logging (`logging_filters.py`),
    bus de eventos (`bus_eventos.py`) y destinos de logging (`destinos/log_handler.py`).
  - Configuración de logging:
    - `neurocampus/app/logging_config.py`.

- **Makefile y scripts**:
  - En la raíz (`Makefile`) se definen comandos para:
    - ejecutar tests,
    - arrancar backend y frontend,
    - lanzar pipelines de preparación y entrenamiento.
  - Scripts adicionales en `backend/scripts/` para experimentos y utilidades.

Esta organización permite:

- Mantener la API limpia y desacoplada de la lógica de modelos.
- Reutilizar la capa de datos y modelos desde jobs batch o desde los endpoints.
- Escalar el frontend sin mezclar lógica de negocio con componentes de UI.
