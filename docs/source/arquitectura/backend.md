# Arquitectura del backend

El backend de NeuroCampus está construido con **FastAPI** y sigue una
organización por “contextos de dominio”:

- **Contextos principales**:
  - `datos`: ingesta, validación y resumen de datasets.
  - `modelos`: entrenamiento y gestión de modelos.
  - `prediccion`: inferencias y scoring.
  - `jobs`: orquestación de procesos de larga duración.
  - `admin`: tareas administrativas y limpieza.

---

## Estructura de directorios

Ubicación base del código del backend:

```text
backend/
  requirements.txt
  requirements-dev.txt
  src/
    neurocampus/
      app/
      data/
      models/
      observability/
      validation/
      ...
```

### 1. Módulo principal de la API

- Archivo: `neurocampus/app/main.py`.
- Responsabilidades:
  - Crear la instancia de `FastAPI`.
  - Configurar CORS.
  - Registrar middlewares (incluido el de Correlation-Id).
  - Registrar routers:
    - `/datos` → `app/routers/datos.py`
    - `/jobs` → `app/routers/jobs.py`
    - `/modelos` → `app/routers/modelos.py`
    - `/prediccion` → `app/routers/prediccion.py`
    - `/admin` → `app/routers/admin_cleanup.py`
  - Exponer un endpoint de salud (`/health`).

### 2. Routers (rutas agrupadas por dominio)

- Carpeta: `neurocampus/app/routers/`.

Principales routers:

- `datos.py`
  - Endpoints como:
    - `/datos/ping`
    - `/datos/esquema`
    - `/datos/validar`
    - `/datos/upload`
    - `/datos/resumen`
    - `/datos/sentimientos`
  - Usa la capa de datos (`data/`) para leer y procesar datasets.

- `modelos.py`
  - Endpoints para:
    - Lanzar entrenamientos de modelos.
    - Consultar resultados y métricas.
  - Usa la capa de modelos (`models/`) y las estrategias de entrenamiento.

- `prediccion.py`
  - Endpoints para generar:
    - Predicciones manuales por docente/asignatura.
    - Predicciones por lote.
  - Se apoya en el modelo campeón registrado.

- `jobs.py`
  - Expone endpoints para disparar o consultar jobs (por ejemplo, preprocesamiento
    con BETO, reentrenamientos, evaluaciones).

- `admin_cleanup.py`
  - Rutas de administración para limpieza de artefactos, datasets temporales, etc.

### 3. Esquemas Pydantic

- Carpeta: `neurocampus/app/schemas/`.

Ficheros relevantes:

- `datos.py`:
  - Inputs/outputs de endpoints `/datos/*`.
  - Esquemas para resumen de dataset, schema de columnas, etc.
- `jobs.py`:
  - Definición de payloads para jobs de preprocesamiento y entrenamiento.
- `modelos.py`:
  - Esquemas para parámetros de entrenamiento y resultados de métricas.
- `prediccion.py`:
  - Esquemas para petición de predicciones y respuestas del modelo.

Esto permite tener:

- Validación de entrada consistente.
- Documentación automática de la API a través del esquema OpenAPI de FastAPI.

### 4. Capa de datos

- Carpeta: `neurocampus/data/`.

Componentes:

- `adapters/`:
  - `almacen_adapter.py`: abstracción sobre el almacenamiento (rutas locales,
    S3 u otros, dependiendo de la configuración).
  - `dataframe_adapter.py`: utilidades para trabajar con pandas u objetos de
    DataFrame.
  - `formato_adapter.py`: manejo de formatos (`csv`, `xlsx`, `parquet`, etc.).

- `chain/`:
  - Implementa cadenas de procesamiento:
    - validación,
    - limpieza,
    - unificación de formatos.
  - Ejemplo: `data/chain/validadores.py`.

- `facades/datos_facade.py`:
  - Expone operaciones de alto nivel:
    - cargar dataset,
    - obtener resumen,
    - persistir versiones procesadas.

- `datos_dashboard.py`:
  - Funciones para construir el resumen utilizado en la pestaña **Datos**:
    - conteo de filas/columnas,
    - periodos,
    - métricas de docentes/asignaturas,
    - agregados de sentimientos.

- `validation_wrapper.py`:
  - Wrapper para ejecutar las validaciones de `neurocampus/validation/`.

### 5. Capa de modelos

- Carpeta: `neurocampus/models/`.

Incluye:

- Modelos manuales:
  - `bm_manual.py`,
  - `rbm_manual.py`,
  - `dbm_manual.py`.

- Estrategias (`models/strategies/`):
  - `bm_manual_strategy.py`,
  - `rbm_manual_strategy.py`,
  - `dbm_manual_strategy.py`,
  - `modelo_rbm_general.py`,
  - `modelo_rbm_restringida.py`,
  - `rbm_pura.py`,
  - `metodologia.py` (agrupa configuraciones y pipelines).

- Auditorías y búsqueda de hiperparámetros:
  - `audit_kfold.py` (validación cruzada),
  - `hparam_search.py` (búsqueda de hiperparámetros).

- Registro y facades:
  - `registry.py` (registro de modelos disponibles),
  - `facades/modelos_facade.py` (interfaz de alto nivel para entrenar y
    obtener resultados).

- Utilidades:
  - `utils_boltzmann.py` (funciones auxiliares para BM/RBM/DBM).
  - `templates/plantilla_entrenamiento.py` (estructura reutilizable para
    entrenamientos).

### 6. Jobs y scripts de backend

- Carpeta: `neurocampus/app/jobs/`.

Principales comandos:

- `cmd_preprocesar_beto.py`:
  - Implementa la lógica de preprocesamiento con BETO para análisis de
    sentimientos.
- `cmd_preprocesar_batch.py`:
  - Preprocesamiento de lotes de archivos de entrada.
- `cmd_cargar_dataset.py`:
  - Carga un dataset desde un origen específico hacia la carpeta de datos.
- `cmd_train_rbm_manual.py` y `cmd_train_dbm_manual.py`:
  - Entrenamiento de modelos RBM/DBM manuales.
- `cmd_eval_confusion.py`:
  - Genera matrices de confusión y métricas de evaluación.
- `cmd_score_docente.py`:
  - Calcula scores o predicciones para docentes concretos.
- `cmd_autoretrain.py`:
  - Lógica de reentrenamiento automático basado en nuevas evaluaciones.

Estos comandos pueden ser invocados:

- Desde línea de comandos (CLI).
- Desde endpoints del router `jobs`.

### 7. Observabilidad y logging

- Carpeta: `neurocampus/observability/`.
- Configuración en `app/logging_config.py`.

Funciones clave:

- Incluir un **Correlation-Id** en todas las peticiones (middleware).
- Enviar eventos de entrenamiento y predicción al bus de eventos.
- Loguear en formato estructurado para facilitar el monitoreo (incluyendo
  `correlation_id`, timestamps y nivel de log).

---

En conjunto, el backend se organiza para que:

- Los routers no contengan lógica pesada (delegan en capas `data` y `models`).
- La lógica de modelos sea reusable desde jobs y endpoints.
- La observabilidad permita seguir el rastro de un dataset o predicción a través
de los distintos componentes del sistema.
