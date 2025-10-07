<!--
NeuroCampus API — v0.1.0 (borrador Día 1)
Contratos HTTP para el MVP del backend (FastAPI).
Versionado semántico: v0.y.z (minor inestable hasta 1.0.0).
Base URL recomendada (local): http://localhost:8000
-->

# NeuroCampus API — v0.1.0

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
- `202 Accepted` → operación encolada/asíncrona (devuelve `job_id`)
- `400 Bad Request` → validación/entrada inválida
- `404 Not Found` → recurso o `job_id` inexistente
- `500 Internal Server Error` → error no controlado

---

## 1 /datos

<!--
Módulo de ingesta y validación de datasets. Primero se consulta un esquema,
luego se pueden validar datos sin persistir o subir definitivamente.
-->

### 1.1 GET `/datos/esquema`

<!-- Devuelve el esquema de la plantilla de datos (columnas, tipos, dominios, requeridos). -->

**Query (opcional)**
- `version`: string (p.ej. `v0.1.0`) — si se omite, retorna la activa.

**200 — Response**
```json
{
  "version": "v0.1.0",
  "columns": [
    { "name": "periodo", "dtype": "string", "required": true, "domain": ["2024-1", "2024-2"] },
    { "name": "docente_id", "dtype": "string", "required": true },
    { "name": "asignatura_id", "dtype": "string", "required": true },
    { "name": "grupo", "dtype": "string", "required": false },
    { "name": "score_global", "dtype": "number", "required": true, "range": [0, 5] },
    { "name": "comentario", "dtype": "string", "required": false, "max_len": 5000 }
  ]
}
```

---

### 1.2 POST `/datos/upload`

<!-- Sube un dataset para ingesta/almacenamiento (validación básica de formato). -->

**Body (multipart/form-data)**
- `file`: CSV/XLSX
- `periodo`: string (p.ej. `2024-2`)
- `overwrite`: boolean (default `false`)

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

<!-- Ejecuta validaciones de calidad (esquema→tipos→dominio→duplicados→calidad) sin almacenar. -->

**Body**
```json
{
  "periodo": "2024-2",
  "inline_data_csv": "base64-CSV-o-URL-opcional",
  "rules": { "strict_types": true, "duplicate_keys": ["docente_id", "asignatura_id", "grupo"] }
}
```

**200 — Response**
```json
{
  "summary": { "rows": 1250, "errors": 5, "warnings": 18 },
  "checks": [
    { "name": "schema", "status": "PASS", "details": {} },
    { "name": "dtype", "status": "WARN", "details": { "score_global": "coercidos 12 valores" } },
    { "name": "domain", "status": "PASS", "details": {} },
    { "name": "duplicates", "status": "FAIL", "details": { "rows": [10, 87, 344] } },
    { "name": "quality", "status": "WARN", "details": { "comentario": "vacíos 35%" } }
  ],
  "recommendations": [
    "Eliminar duplicados por docente_id+asignatura_id+grupo",
    "Revisar coerción de score_global"
  ]
}
```

---

## 2 /modelos

<!--
Entrenamiento, estado y publicación de modelos. Se soportan variantes RBM (general y restringida) como MVP.
-->

### 2.1 POST `/modelos/entrenar`

<!-- Lanza entrenamiento con configuración dada. -->

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

<!-- Consulta últimos modelos y su estado (con métricas si aplica). -->

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

<!-- Marca un modelo como activo en un canal (p.ej. produccion) para predicción online. -->

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

<!--
Predicción unitaria para UI y por lotes para procesamiento offline.
-->

### 3.1 POST `/prediccion/online`

<!-- Predicción unitaria (uso interactivo/UI). -->

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

<!-- Predicción por lotes (carga CSV/XLSX o referencia a dataset). -->

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

<!--
Ejecución y monitoreo de trabajos asíncronos (pipelines de datos, entrenamiento, predicciones batch).
-->

### 4.1 POST `/jobs/run`

<!-- Ejecuta comandos predefinidos de orquestación. -->

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

<!-- Consulta estado y logs de un job, con progreso de pasos cuando aplique. -->

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

<!-- Lista los últimos jobs con filtro opcional por tipo. -->

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

<!--
Solo un resumen para guiar la implementación; las definiciones completas
vivirán en backend/src/neurocampus/app/schemas/.
-->

- `DatosUploadResponse`: `dataset_id:string`, `rows_ingested:int`, `stored_as:string`, `warnings:string[]`
- `ValidacionRespuesta`: `summary:{rows,int errors,int warnings}`, `checks:[{name,status,details}]`
- `EntrenarRequest`: `nombre,tipo,metodologia,dataset_id,params{...}`
- `ModeloEstadoItem`: `nombre,tipo,dataset_id,status,metrics?,error?,artifact_uri?`
- `PublicarRequest`: `nombre,canal`
- `PredOnlineRequest`: `modelo?,payload{...}`
- `PredOnlineResponse`: `modelo,pred{label,score,confianza},explicacion?,metadata?`
- `JobRunRequest`: `command,args{...}`
- `JobStatus`: `job_id,status,steps[],logs_tail[]`

---

## 6 Ejemplos de uso (curl)

<!--
Snippets rápidos para probar los contratos desde terminal.
-->

**Subir datos**
```bash
curl -X POST http://localhost:8000/datos/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./examples/dataset_ejemplo.csv" \
  -F periodo=2024-2
```

**Entrenar RBM**
```bash
curl -X POST http://localhost:8000/modelos/entrenar \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "nombre":"rbm_r1_2024_2","tipo":"rbm_restringida","metodologia":"PeriodoActual","dataset_id":"2024-2",
    "params":{"hidden_units":128,"learning_rate":0.01,"epochs":30,"batch_size":256,"regularization":"l2","seed":42}
  }'
```

**Predicción online**
```bash
curl -X POST http://localhost:8000/prediccion/online \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "modelo":"rbm_r1_2024_2",
    "payload":{"periodo":"2024-2","docente_id":"DOC123","asignatura_id":"QCH-201","grupo":"A","score_global":4.3,"comentario":"Buena comunicación"}
  }'
```

---

## 7 Notas de versión v0.1.0

<!--
Qué incluye esta versión del contrato y advertencia de cambios antes del release.
-->

- Incluye contratos base de `/datos`, `/modelos`, `/prediccion`, `/jobs`.
- `prediccion/online` retorna `explicacion.features_top` como placeholder.
- `jobs/*` expone `steps` y `logs_tail` mínimos para UI de monitoreo.
- **Sujeto a cambios** antes de `release/0.1.0`.

---

## 8 Referencias internas (alineación con repo y UI)

<!--
Estas referencias son para mantener trazabilidad con otros artefactos del proyecto
(estructura de repo, plan de trabajo y mockups). No requieren enlaces externos.
-->

- Estructura del repositorio y rutas planificadas en backend y frontend.
- Árbol base con `docs/api.md` y módulos `/datos`, `/modelos`, `/prediccion`, `/jobs`.
- Mockups/pestañas de la UI que consumen estos endpoints (Datos, Modelos, Predicciones, Jobs).
