# NeuroCampus API — v0.3.0

---

## Convenciones
- **Base URL**: `http://127.0.0.1:8000`
- **Auth**: (TBD Día 3+)
- **Formato**: `application/json; charset=utf-8`
- **Fechas**: ISO-8601 (`YYYY-MM-DDTHH:mm:ssZ`)
- **Errores**: cuerpo `{ "error": string, "code"?: string }`
- **Números**: `float64` (salvo que se indique lo contrario)
- **Nombres**: `snake_case` en claves de JSON

### Códigos de estado (uso común)
- `200 OK` → operación síncrona exitosa (devuelve resultado)
- `201 Created` → recurso creado (ej. upload)
- `202 Accepted` → operación encolada/asíncrona (devuelve `job_id`)
- `400 Bad Request` → validación/entrada inválida
- `404 Not Found` → recurso o `job_id` inexistente
- `409 Conflict` → conflicto lógico (p.ej. dataset existente con overwrite=false)
- `500 Internal Server Error` → error no controlado

---

## 1 /datos

### 1.1 GET `/datos/esquema`

Devuelve el esquema del dataset esperado para carga inicial.  
La respuesta se construye a partir de `schemas/plantilla_dataset.schema.json` (con fallback en memoria).

**Ejemplo de respuesta:**
```json
{
  "version": "v0.3.0",
  "columns": [
    { "name": "periodo", "dtype": "string", "required": true },
    { "name": "codigo_materia", "dtype": "string", "required": true },
    { "name": "grupo", "dtype": "integer", "required": true },
    { "name": "pregunta_1", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_2", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_3", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_4", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_5", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_6", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_7", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_8", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_9", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "pregunta_10", "dtype": "number", "required": true, "range": [0, 50] },
    { "name": "Sugerencias:", "dtype": "string", "required": false, "max_len": 5000 }
  ]
}
```

---

### 1.2 POST `/datos/upload`

Permite subir un archivo CSV o XLSX con los datos de evaluación docente.  
La carga se valida según `schemas/plantilla_dataset.schema.json` (nivel mínimo, mock hasta persistencia real).

**Body (multipart/form-data)**
- `file`: Archivo CSV/XLSX a subir.
- `periodo`: Cadena que indica el periodo académico (ej. `"2024-2"`).
- `overwrite`: Booleano, indica si se debe sobrescribir un dataset existente.

**Importante:**  
Los campos derivados de PLN — `comentario.sent_pos`, `comentario.sent_neg`, `comentario.sent_neu` — **no deben incluirse** en el archivo cargado.  
Estos valores se **calcularán** durante la etapa de análisis de sentimientos (Día 6).

**201 — Response**
```json
{
  "dataset_id": "2024-2",
  "rows_ingested": 1250,
  "stored_as": "s3://neurocampus/datasets/2024-2.parquet",
  "warnings": ["col 'grupo' vacío en 32 filas"]
}
```

**409 — Response** (overwrite=false y ya existe)
```json
{
  "error": {
    "code": "CONFLICT",
    "message": "Dataset 2024-2 ya existe"
  }
}
```

---

### 1.3 POST `/datos/validar`

Valida un archivo de datos **sin** almacenarlo. Ejecuta cadena: **esquema → tipos → dominio → duplicados → calidad**.  
Soporta **CSV/XLSX/Parquet** y funciona con **Pandas** o **Polars** (configurable).

**Body (multipart/form-data)**
- `file`: Archivo `CSV | XLSX | Parquet`.
- `fmt` (opcional): fuerza el lector, uno de `"csv" | "xlsx" | "parquet"`.

**200 — Response**
```json
{
  "summary": { "rows": 1250, "errors": 5, "warnings": 18, "engine": "pandas" },
  "issues": [
    { "code": "MISSING_COLUMN", "severity": "error", "column": "pregunta_7", "row": null, "message": "Columna requerida ausente: pregunta_7" },
    { "code": "BAD_TYPE", "severity": "warning", "column": "codigo_materia", "row": null, "message": "Tipo esperado string vs observado int64" },
    { "code": "DOMAIN_VIOLATION", "severity": "error", "column": "periodo", "row": null, "message": "Valor fuera de dominio en periodo: 2024-13" }
  ]
}
```

**Códigos**
- `200` validado con éxito
- `400` error al procesar archivo / formato inválido

**Notas**
- El engine por defecto es **`pandas`**, configurable con `NC_DF_ENGINE=polars`.
- No persiste ni transforma el archivo; es solo validación para reporte en UI.
- El contrato de respuesta corresponde a `DatosValidarResponse` (Pydantic).

#### 🆕 Normalización de encabezados y equivalencias
La validación **tolera nombres de columnas con espacios o con `_`**, acentos y signos como `:`.  
Internamente se normalizan (minúsculas, sin acentos, sin puntuación) y se aplican sinónimos básicos.

**Ejemplos de equivalencias admitidas:**
- `codigo_materia` ≡ `codigo materia` ≡ `Código Materia` ≡ `codigo_asignatura`
- `cedula_profesor` ≡ `cedula profesor` ≡ `cédula docente` ≡ `docente_id`
- `Sugerencias:` ≡ `sugerencias` ≡ `Sugerencias`  
*(Se ignoran columnas `Unnamed: N` y se mantienen los nombres canónicos en el reporte).*

#### 🆕 Coerción de tipos antes de validar
Si el esquema define el tipo de una columna, se intenta **coaccionar** el dato antes de reportar `BAD_TYPE`:
- `string`: convierte enteros/números a cadena (IDs como `codigo_materia`, `cedula_profesor`).
- `integer`: `to_numeric(...).astype("Int64")` (pandas) o `Int64` (polars).
- `number`: `float`.
- `boolean`: mapeo automático (`true/false/1/0`).
- `date/datetime`: `to_datetime(..., errors="coerce")`.
- **Nullables**: se respetan tipos del estilo `["string","null"]` (JSON Schema).

#### 🆕 Lectura de archivos (robusta)
- **CSV**: reintentos con `utf-8-sig` y `latin-1`; autodetección de separador (`; | tab | ,`) y manejo de BOM.
- **XLSX** con Polars: lectura vía **pandas** y conversión a **polars**.
- **Parquet**: lectura nativa.

#### 🆕 Esquema aceptado
- **JSON Schema** (`properties`, `required`, `enum`, `minimum/maximum`) **o**
- Formato nativo: `{"columns":[{"name","dtype","domain{allowed|min|max}"}]}`.

#### 🆕 Resolución del schema en runtime
- El backend intenta localizar `schemas/plantilla_dataset.schema.json` subiendo directorios desde el código.
- Se puede forzar la ruta con `NC_SCHEMA_PATH=/ruta/absoluta/plantilla_dataset.schema.json`.

---

## 2 /modelos

<!-- Entrenamiento, estado y publicación de modelos. -->

### 2.1 POST `/modelos/entrenar`

**Body**
```json
{
  "nombre": "rbm_r1_2024_2",
  "tipo": "rbm_restringida",
  "metodologia": "PeriodoActual",
  "dataset_id": "2024-2",
  "params": {
    "hidden_units": 128,
    "learning_rate": 0.01,
    "epochs": 30,
    "batch_size": 256,
    "regularization": "l2",
    "seed": 42
  }
}
```

**202 — Response**
```json
{ "job_id": "job_train_01H8ZK...", "status": "QUEUED" }
```

---

### 2.2 GET `/modelos/estado`

**Query**
- `limit`: int (default 20)

**200 — Response**
```json
{
  "items": [
    {
      "nombre": "rbm_r1_2024_2",
      "tipo": "rbm_restringida",
      "dataset_id": "2024-2",
      "created_at": "2025-09-09T10:00:00Z",
      "status": "COMPLETED",
      "metrics": { "f1_macro": 0.81, "accuracy": 0.86 },
      "artifact_uri": "s3://neurocampus/models/rbm_r1_2024_2/"
    },
    {
      "nombre": "rbm_g1_2024_1",
      "tipo": "rbm_general",
      "dataset_id": "2024-1",
      "status": "FAILED",
      "error": "Divergencia en entrenamiento"
    }
  ]
}
```

---

### 2.3 POST `/modelos/publicar`

**Body**
```json
{ "nombre": "rbm_r1_2024_2", "canal": "produccion" }
```

**200 — Response**
```json
{ "published": true, "canal": "produccion", "nombre": "rbm_r1_2024_2" }
```

---

## 3 /prediccion

### 3.1 POST `/prediccion/online`

**Body**
```json
{
  "modelo": "rbm_r1_2024_2",
  "payload": {
    "periodo": "2024-2",
    "docente_id": "DOC123",
    "asignatura_id": "QCH-201",
    "grupo": "A",
    "score_global": 4.3,
    "comentario": "Buena comunicación, carga alta."
  }
}
```

**200 — Response**
```json
{
  "modelo": "rbm_r1_2024_2",
  "pred": { "label": "alto_desempeno", "score": 0.82, "confianza": 0.77 },
  "explicacion": { "features_top": ["score_global", "comentario.sentimiento_pos"] },
  "metadata": { "latencia_ms": 42 }
}
```

---

### 3.2 POST `/prediccion/batch`

**Body (multipart/form-data)**
- `file`: CSV/XLSX (opcional si se pasa `dataset_id`)
- `dataset_id`: string (opcional)
- `modelo`: string (opcional; si no, usa el publicado en canal `produccion`)
- `download`: enum(`csv`,`parquet`) — default `csv`

**202 — Response**
```json
{ "job_id": "job_pred_01H8ZL...", "status": "QUEUED" }
```

---

## 4 /jobs

*(Reservado para próximos días de desarrollo)*

### 4.1 POST `/jobs/run`

**Body**
```json
{
  "command": "entrenamiento_completo",
  "args": { "dataset_id": "2024-2", "tipo": "rbm_restringida" }
}
```

**202 — Response**
```json
{ "job_id": "job_pipe_01H8ZM...", "status": "QUEUED" }
```

---

### 4.2 GET `/jobs/status/{id}`

**200 — Response**
```json
{
  "job_id": "job_pipe_01H8ZM...",
  "status": "RUNNING",
  "started_at": "2025-09-09T10:05:00Z",
  "steps": [
    { "name": "cargar_dataset", "status": "DONE", "duration_s": 5 },
    { "name": "unificar_historico", "status": "DONE", "duration_s": 11 },
    { "name": "entrenar", "status": "RUNNING", "progress": 0.6 }
  ],
  "logs_tail": [
    "epoch=18/30 loss=0.42 acc=0.85",
    "epoch=19/30 loss=0.41 acc=0.86"
  ]
}
```

---

### 4.3 GET `/jobs/list`

**Query**
- `kind`: enum(`train`,`predict`,`pipeline`) (opcional)
- `limit`: int (default 50)

**200 — Response**
```json
{
  "items": [
    { "job_id": "job_train_01H8ZK...", "status": "COMPLETED", "ended_at": "2025-09-09T10:18:12Z" },
    { "job_id": "job_pred_01H8ZL...", "status": "FAILED", "error": "Archivo inválido" }
  ]
}
```

---

## 5 Esquemas Pydantic (resumen)

- `DatosUploadResponse`: `dataset_id:string`, `rows_ingested:int`, `stored_as:string`, `warnings:string[]`
- `DatosValidarResponse`: `summary{rows:int,errors:int,warnings:int,engine:string}`, `issues[ValidIssue]`
- `ValidIssue`: `code,severity("error"|"warning"),column?,row?,message`
- `EntrenarRequest`: `nombre,tipo,metodologia,dataset_id,params{...}`
- `ModeloEstadoItem`: `nombre,tipo,dataset_id,status,metrics?,error?,artifact_uri?`
- `PublicarRequest`: `nombre,canal`
- `PredOnlineRequest`: `modelo?,payload{...}`
- `PredOnlineResponse`: `modelo,pred{label,score,confianza},explicacion?,metadata?`
- `JobRunRequest`: `command,args{...}`
- `JobStatus`: `job_id,status,steps[],logs_tail[]`

---

## 6 Ejemplos de uso (curl)

**Validar sin almacenar (CSV)**
```bash
curl -X POST http://127.0.0.1:8000/datos/validar   -F "file=@examples/dataset_ejemplo.csv"
```

**Validar forzando formato XLSX**
```bash
curl -X POST http://127.0.0.1:8000/datos/validar   -F "file=@examples/plantilla.xlsx"   -F "fmt=xlsx"
```

**Subir datos**
```bash
curl -X POST http://localhost:8000/datos/upload   -F "file=@./examples/dataset_ejemplo.csv"   -F periodo=2024-2
```

**Entrenar RBM**
```bash
curl -X POST http://localhost:8000/modelos/entrenar   -H "Content-Type: application/json"   -d '{
    "nombre":"rbm_r1_2024_2","tipo":"rbm_restringida","metodologia":"PeriodoActual","dataset_id":"2024-2",
    "params":{"hidden_units":128,"learning_rate":0.01,"epochs":30,"batch_size":256,"regularization":"l2","seed":42}
  }'
```

**Predicción online**
```bash
curl -X POST http://localhost:8000/prediccion/online   -H "Content-Type: application/json"   -d '{
    "modelo":"rbm_r1_2024_2",
    "payload":{"periodo":"2024-2","docente_id":"DOC123","asignatura_id":"QCH-201","grupo":"A","score_global":4.3,"comentario":"Buena comunicación"}
  }'
```

---

## 7 Notas de versión v0.3.0

- **Nuevo:** `POST /datos/validar` (multipart) para validar CSV/XLSX/Parquet sin almacenar.
- **Contrato:** respuesta con `summary{rows,errors,warnings,engine}` e `issues[]`.
- **Config:** soporte de engine `pandas`/`polars` (env `NC_DF_ENGINE`).
- **Docs:** se documenta **normalización de encabezados**, **coerción de tipos**, **lectura CSV robusta**, soporte de **JSON Schema** y `NC_SCHEMA_PATH`.
- **Mantiene**: endpoints de `/modelos`, `/prediccion` y `/jobs` documentados para MVP.

---

## 8 Referencias internas

- Estructura del repositorio y módulos de backend/frontend.
- Mockups/pestaña **Datos** para presentar KPIs y tabla de issues de validación.
