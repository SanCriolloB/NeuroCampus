# Endpoints de datos

La API de datos agrupa los endpoints relacionados con la **ingesta**, 
**validación**, **preprocesamiento** y **resumen** de los datasets usados por 
NeuroCampus.

Todos los endpoints aquí descritos cuelgan del prefijo base:

- `/datos`

---

## Resumen de endpoints principales

| Método | Ruta                  | Descripción                                                 |
| ------ | --------------------- | ----------------------------------------------------------- |
| GET    | `/datos/ping`         | Verificación rápida de salud del módulo de datos.          |
| GET    | `/datos/esquema`      | Devuelve el esquema esperado de columnas para los datasets |
| POST   | `/datos/validar`      | Valida un archivo sin persistirlo en el sistema            |
| POST   | `/datos/upload`       | Sube y registra un nuevo dataset                           |
| GET    | `/datos/resumen`      | Devuelve el resumen estructural del dataset activo         |
| GET    | `/datos/sentimientos` | Devuelve agregados de análisis de sentimientos (BETO)      |

> **Nota**: los nombres concretos de los parámetros pueden variar según la 
> implementación, pero esta sección resume la intención funcional de cada ruta.

---

## `GET /datos/ping`

### Descripción

Endpoint de comprobación rápida del módulo de datos. Útil para:

- Verificar la conectividad entre frontend y backend.
- Checks básicos en tests automáticos o sistemas de monitorización.

### Respuesta

- Código `200 OK` con un pequeño JSON indicando estado (por ejemplo,
  `{ "status": "ok" }`).

---

## `GET /datos/esquema`

### Descripción

Devuelve el **esquema esperado de columnas** para los datasets de evaluaciones
docentes. Esta información se utiliza:

- En la pestaña **Datos** para mostrar la plantilla de columnas.
- Como referencia al preparar archivos para carga o predicción por lote.

### Entrada

- Parámetros típicos (opcional):
  - `tipo` o `dataset_id` para seleccionar una variante de esquema, si aplica.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON con campos similares a:

  - `version`: versión del esquema.
  - `columns`: lista de columnas esperadas con información como:
    - nombre interno,
    - tipo de dato,
    - si es requerida u opcional,
    - descripción.

Ejemplo simplificado:

```json
{
  "version": "0.3.0",
  "columns": [
    {"name": "docente", "type": "string", "required": true},
    {"name": "asignatura", "type": "string", "required": true},
    {"name": "p1", "type": "number", "required": true},
    "..."
  ]
}
```

---

## `POST /datos/validar`

### Descripción

Valida un archivo de datos **sin persistirlo** en el sistema. Se utiliza para:

- Comprobar que el archivo cumple con el esquema esperado.
- Obtener una vista previa de filas y posibles errores antes de cargarlo.

### Entrada

- `multipart/form-data` con campos típicos:

  - `file`: archivo a validar (`.csv`, `.xlsx`, `.xls`, `.parquet`, etc.).
  - `dataset_id` (opcional): identificador que se planea asignar al dataset.

### Comportamiento esperado

- El backend:
  - Lee el archivo con un adaptador de formato.
  - Aplica la cadena de validación definida en `neurocampus.data` / `validation`.
  - Devuelve un informe de validación y, opcionalmente, una muestra de filas.

### Respuesta

- Código `200 OK` si el archivo es legible (aunque tenga errores de validación).
- Cuerpo JSON típico:

  - `ok`: `true` / `false` según el resultado de validación.
  - `errors`: lista de errores.
  - `warnings`: lista de advertencias.
  - `sample`: muestra de filas (por ejemplo, primeras 5–10).
  - `dataset_id`: identificador sugerido o inferido (si aplica).

---

## `POST /datos/upload`

### Descripción

Carga y registra un nuevo dataset en el sistema. Este endpoint suele engancharse
a la acción **«Cargar y procesar»** de la pestaña **Datos**.

### Entrada

- `multipart/form-data` con campos típicos:

  - `file`: archivo con las evaluaciones docentes.
  - `dataset_id`: identificador del dataset (por ejemplo, `2024-2`).
  - `overwrite`: booleano que indica si se debe sobrescribir un dataset ya 
    existente con el mismo `dataset_id`.
  - Flags adicionales (según implementación):
    - `run_preproc`: aplicar preprocesamiento al cargar.
    - `run_beto`: lanzar job de análisis de sentimientos tras el preprocesamiento.

### Comportamiento esperado

- El backend:
  - Valida el archivo.
  - Lo transforma al formato interno estándar.
  - Lo guarda en la ubicación configurada (`data/`).
  - Genera y persiste un **resumen de dataset**.
  - Si se indica, lanza el job de BETO.

### Respuesta

- En caso de éxito, código `201 Created` o `200 OK`.
- Cuerpo JSON típico:

  - `ok`: `true`.
  - `dataset_id`: identificador efectivamente registrado.
  - `rows_ingested`: número de filas ingestadas.
  - `stored_as`: ruta o URI donde quedó almacenado el dataset.
  - `preproc_job_id`: id del job de preprocesamiento, si aplica.
  - `beto_job_id`: id del job BETO lanzado, si aplica.

---

## `GET /datos/resumen`

### Descripción

Devuelve un resumen estructural del dataset activo o del `dataset_id`
especificado. Es la base de la sección **«Resumen del dataset»** en la pestaña
**Datos**.

### Entrada

- Parámetros de query típicos:

  - `dataset_id` (opcional): si no se indica, puede usarse un dataset por 
    defecto o el último cargado.

### Respuesta

- Código `200 OK` en caso de éxito.
- Cuerpo JSON con información como:

  - `dataset_id`
  - `n_rows`, `n_columns`
  - `columns`: lista de columnas con tipos y estadísticas básicas.
  - `docentes_count`, `asignaturas_count`
  - `periodos`: listado o rango de periodos detectados.
  - Otros agregados de interés para el dashboard de datos.

Esta información se utiliza para poblar las tarjetas, tablas y gráficos del
**resumen de dataset**.

---

## `GET /datos/sentimientos`

### Descripción

Devuelve agregados de **análisis de sentimientos** derivados del preprocesamiento
con BETO. Se usa tanto en la pestaña **Datos** como, potencialmente, en otras
vistas analíticas.

### Entrada

- Parámetros de query típicos:

  - `dataset_id`: dataset sobre el que se quiere consultar.
  - Otros filtros opcionales (por ejemplo, `docente`, `asignatura`), según la
    implementación.

### Respuesta

- Código `200 OK` en caso de éxito.
- Cuerpo JSON con campos típicos:

  - `global_counts`: lista de objetos resumen global (pos/neu/neg), con recuentos 
    y proporciones.
  - `top_docentes`: agregados por docente ordenados según alguna métrica.
  - `coverage` y otros indicadores auxiliares (porcentaje de filas con texto,
    tasa de éxito del modelo, etc.).

Estos datos alimentan:

- Gráfico de barras global (positivo / neutro / negativo).
- Tablas y gráficos por docente/asignatura.
