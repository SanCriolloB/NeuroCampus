# Endpoints de jobs

La API de **jobs** gestiona procesos que pueden tomar más tiempo que una
petición HTTP típica, como:

- Preprocesamiento de textos con BETO.
- Entrenamiento de modelos (RBM/BM/DBM y variantes).
- Auditorías, evaluaciones y otros procesos batch.

Todos los endpoints aquí descritos cuelgan del prefijo base:

- `/jobs`

---

## Resumen de endpoints principales

| Método | Ruta                        | Descripción                                          |
| ------ | --------------------------- | ---------------------------------------------------- |
| POST   | `/jobs/preproc/beto/run`    | Lanza un job de preprocesamiento BETO               |
| GET    | `/jobs/preproc/beto/status` | Consulta el estado agregado de jobs BETO recientes  |
| GET    | `/jobs/{job_id}`            | Consulta el estado de un job específico             |
| GET    | `/jobs`                     | Lista jobs con filtros opcionales (tipo, estado…)   |

> **Nota**: los nombres exactos de rutas pueden variar ligeramente según la
> implementación interna. Esta sección describe el objetivo funcional de la
> API de jobs.

---

## `POST /jobs/preproc/beto/run`

### Descripción

Lanza un job de **preprocesamiento con BETO** sobre un dataset concreto.
Se utiliza desde la pestaña **Datos** cuando se marca la opción de ejecutar
BETO, o al usar el botón explícito para análisis de sentimientos.

### Entrada

- Cuerpo JSON típico:

  - `dataset_id`: identificador del dataset sobre el que se quiere ejecutar
    BETO.
  - Parámetros adicionales opcionales:
    - `text_col`: nombre de la columna de texto (por defecto `comentario`).
    - `keep_empty_text`: bandera para conservar filas sin texto.
    - `text_feats`, `tfidf_min_df`, `tfidf_max_df`, etc., si están soportados.

### Comportamiento esperado

- El backend crea un job de tipo “preprocesamiento BETO”:
  - Coloca el trabajo en la cola (si existe un sistema de colas),
  - o lo ejecuta de forma asíncrona/secuencial según la configuración.
- Registra un `job_id` para poder consultar su progreso.

### Respuesta

- Código `200 OK` o `202 Accepted` (según el diseño del backend).
- Cuerpo JSON típico:

  - `job_id`: identificador único del job.
  - `status`: estado inicial (por ejemplo, `created` o `running`).
  - `dataset_id`: dataset objetivo.

---

## `GET /jobs/preproc/beto/status`

### Descripción

Devuelve una vista resumida del estado de los jobs de BETO recientes, útil para
mostrar información rápida en la pestaña **Datos** (por ejemplo, cuándo se
ejecutó BETO por última vez y sobre qué dataset).

### Entrada

- Parámetros de query opcionales:
  - `dataset_id`: filtrar por dataset.
  - `limit`: número máximo de registros.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON típico:

  - `jobs`: lista de jobs con campos como:
    - `job_id`,
    - `dataset_id`,
    - `status` (`created`, `running`, `completed`, `failed`, etc.),
    - `started_at`, `finished_at`,
    - información de salida (ruta de archivos generados, métricas, etc.).

---

## `GET /jobs/{job_id}`

### Descripción

Devuelve el estado detallado de un job concreto, identificado por `job_id`.
Es útil cuando el frontend necesita:

- Mostrar una barra de progreso o estado detallado.
- Diagnosticar un fallo en un job específico.

### Entrada

- `job_id` en la ruta.

### Respuesta

- Código `200 OK` si el job existe.
- Cuerpo JSON típico:

  - `job_id`
  - `type` (por ejemplo, `preproc_beto`, `train_rbm`, etc.).
  - `status`
  - `created_at`, `started_at`, `finished_at`
  - `progress` (si está disponible).
  - `result` o `output` (por ejemplo, rutas a artefactos, métricas, etc.).
  - `error` (si el job falló).

---

## `GET /jobs`

### Descripción

Lista los jobs existentes, filtrando opcionalmente por tipo, estado u otros
criterios. Es la base para ofrecer vistas tipo “monitor de jobs” en la pestaña
**Jobs** del frontend.

### Entrada

- Parámetros de query opcionales:
  - `type`: tipo de job (`preproc_beto`, `train_rbm`, etc.).
  - `status`: estado (`running`, `completed`, `failed`, etc.).
  - `dataset_id`: filtrar por dataset asociado.
  - `limit`, `offset` o parámetros de paginación.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON típico:

  - `items`: lista de jobs con campos resumidos (id, tipo, estado, fechas).
  - `total`: total de jobs que cumplen el filtro (si se implementa paginación).
