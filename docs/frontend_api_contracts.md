# Frontend API Contracts — NeuroCampus (v1)

Este documento define **contratos mínimos** esperados por el frontend (rebuild UI 1:1 “Data Flow and Visualization”).  
Objetivo: asegurar **trazabilidad por `dataset_id` / `periodo`** y **actualización transversal** entre pestañas (Datos → Modelos → Predicciones → Dashboard).

> Nota: Si el backend ya tiene endpoints equivalentes con otros nombres, este documento debe mapearlos 1:1.  
> Si faltan endpoints, este documento actúa como “source of truth” para implementarlos.

---

## Convenciones

- **Base URL**: `VITE_API_BASE` (frontend env)
- **Formato**: JSON para requests/responses, excepto uploads batch (multipart).
- **Errores**: preferible `{ "detail": "..." }` o `{ "message": "..." }` con status 4xx/5xx.
- **Trazabilidad**:
  - `dataset_id` es el identificador estable del dataset (idealmente UUID o key estable).
  - `periodo` es una dimensión lógica (ej. `2025-1`), puede coexistir con `dataset_id`.
- **Estados**:
  - Dataset: `uploaded | processing | ready | failed`
  - Jobs: `created | running | done | failed`

---

# 1) Datos

## 1.1 GET /datos/esquema

Devuelve el esquema esperado (campos requeridos/opcionales) y metadata útil para validación.

**200**
```json
{
  "version": "1",
  "required": ["col_a", "col_b"],
  "optional": ["col_c"],
  "fields": [
    {
      "name": "col_a",
      "dtype": "string",
      "required": true,
      "desc": "Descripción del campo"
    }
  ],
  "examples": {
    "col_a": "texto",
    "col_b": 4.5
  }
}
```

---

## 1.2 POST /datos/validar (multipart)

Valida estructura antes de subir/procesar.
Debe retornar issues (info/warning/error), conteos y muestra.

**form-data**
- `file`: csv/xlsx/parquet
- `dataset_id`: string (compat)
- `fmt` (opcional): `csv|xlsx|parquet`

**200**
```json
{
  "ok": true,
  "dataset_id": "2025-1",
  "missing": [],
  "extra": [],
  "sample": [{"col_a": "x"}],
  "issues": [{"level": "warning", "msg": "...", "col": "col_a"}],
  "n_rows": 1000,
  "n_cols": 20
}
```

**422 (ejemplo)**
```json
{
  "detail": "Missing required columns: docente, asignatura"
}
```

---

## 1.3 POST /datos/upload (multipart)

Sube dataset para procesamiento y/o persistencia.
Debe retornar `dataset_id` y `status`.

**form-data**
- `file`
- `periodo` (string)  // compat actual
- `dataset_id` (string) // compat
- `overwrite` ("true"|"false")

**200**
```json
{
  "ok": true,
  "dataset_id": "2025-1",
  "stored_as": "evaluaciones_2025_1.csv",
  "status": "ready",
  "rows_read": 1050,
  "rows_valid": 1000
}
```

**200 (si el procesamiento queda asíncrono)**
```json
{
  "ok": true,
  "dataset_id": "2025-1",
  "stored_as": "evaluaciones_2025_1.csv",
  "status": "processing",
  "job_id": "job_abc123"
}
```

---

## 1.4 GET /datos/resumen?dataset=2025-1

KPIs para DataTab y metadata para llenar tablas/charts.

**200**
```json
{
  "dataset_id": "2025-1",
  "n_rows": 1000,
  "n_cols": 20,
  "periodos": ["2025-1"],
  "fecha_min": "2025-01-15",
  "fecha_max": "2025-05-20",
  "n_docentes": 60,
  "n_asignaturas": 120,
  "columns": [
    {
      "name": "docente",
      "dtype": "string",
      "non_nulls": 1000,
      "sample_values": ["A", "B"]
    }
  ]
}
```

---

## 1.5 GET /datos/sentimientos?dataset=2025-1

Agregados para charts en DataTab. (Global + por docente + por asignatura).

**200**
```json
{
  "dataset_id": "2025-1",
  "total_comentarios": 980,
  "global_counts": [
    {"label": "neg", "count": 200, "proportion": 0.204},
    {"label": "neu", "count": 500, "proportion": 0.510},
    {"label": "pos", "count": 280, "proportion": 0.286}
  ],
  "por_docente": [
    {
      "group": "Docente X",
      "counts": [{"label": "pos", "count": 10, "proportion": 0.50}]
    }
  ],
  "por_asignatura": [
    {
      "group": "Matemáticas",
      "counts": [{"label": "neg", "count": 5, "proportion": 0.20}]
    }
  ]
}
```

---

## 1.6 (Recomendado) GET /datos/list

Lista datasets disponibles para selector global y navegación histórica.

**200**
```json
[
  {
    "dataset_id": "2025-1",
    "periodo": "2025-1",
    "nombre": "Encuesta 2025-1",
    "created_at": "2025-06-01T10:20:30Z",
    "status": "ready",
    "rows": 1000,
    "cols": 20
  }
]
```

---

# 2) Jobs

> Solo aplica si el procesamiento es asíncrono.

## 2.1 POST /jobs/preproc/beto/run

Dispara el pre-procesamiento y/o análisis de sentimientos (BETO).

**Body**
```json
{
  "dataset": "2025-1",
  "text_col": null,
  "keep_empty_text": true
}
```

**200**
```json
{
  "id": "job_abc123",
  "dataset": "2025-1",
  "status": "created",
  "created_at": "2025-12-19T13:00:00Z",
  "meta": {}
}
```

---

## 2.2 GET /jobs/preproc/beto/:job_id

Consulta estado y progreso.

**200**
```json
{
  "id": "job_abc123",
  "dataset": "2025-1",
  "src": "evaluaciones_2025_1.csv",
  "dst": "evaluaciones_2025_1_beto.csv",
  "status": "running",
  "created_at": "2025-12-19T13:00:00Z",
  "started_at": "2025-12-19T13:01:00Z",
  "finished_at": null,
  "error": null,
  "meta": {
    "progress": 0.35,
    "stage": "tokenizing"
  }
}
```

**200 (done)**
```json
{
  "id": "job_abc123",
  "dataset": "2025-1",
  "status": "done",
  "created_at": "2025-12-19T13:00:00Z",
  "started_at": "2025-12-19T13:01:00Z",
  "finished_at": "2025-12-19T13:10:00Z",
  "error": null,
  "meta": {
    "progress": 1.0,
    "stage": "completed"
  }
}
```

**200 (failed)**
```json
{
  "id": "job_abc123",
  "dataset": "2025-1",
  "status": "failed",
  "created_at": "2025-12-19T13:00:00Z",
  "started_at": "2025-12-19T13:01:00Z",
  "finished_at": "2025-12-19T13:03:00Z",
  "error": "Model not available",
  "meta": {
    "progress": 0.1,
    "stage": "loading-model"
  }
}
```

---

# 3) Modelos

## 3.1 GET /modelos/runs?model_name=...&dataset_id=...&periodo=...

Lista runs entrenados. Debe soportar filtrado por dataset/periodo para trazabilidad.

**200**
```json
[
  {
    "run_id": "run_123",
    "model_name": "rbm_general",
    "created_at": "2025-12-19T12:00:00Z",
    "metrics": {
      "accuracy": 0.85,
      "f1_macro": 0.81,
      "loss": 0.22
    }
  }
]
```

---

## 3.2 GET /modelos/champion?model_name=...&dataset_id=...&periodo=...

Devuelve el modelo “champion” actual. La política debe estar definida (global o por dataset/periodo).

**200**
```json
{
  "model_name": "rbm_general",
  "metrics": {"accuracy": 0.88, "f1_macro": 0.84},
  "path": "models/rbm_general/champion.pkl"
}
```

---

## 3.3 POST /modelos/entrenar

Dispara entrenamiento (posible job).

**Body**
```json
{
  "modelo": "rbm_general",
  "data_ref": "2025-1",
  "epochs": 10,
  "hparams": {}
}
```

**200**
```json
{
  "job_id": "train_job_001",
  "status": "running"
}
```

---

## 3.4 GET /modelos/estado/:job_id

Consulta estado del entrenamiento.

**200**
```json
{
  "job_id": "train_job_001",
  "status": "running",
  "metrics": {"accuracy": 0.70, "f1_macro": 0.65},
  "history": [{"epoch": 1, "loss": 0.50}]
}
```

---

# 4) Predicción

## 4.1 POST /prediccion/online

Predicción online (usa champion global o por dataset según política).

**Body**
```json
{
  "family": "sentiment_desempeno",
  "input": {
    "calificaciones": {"p1": 4.5, "p2": 4.0},
    "comentario": "Buen profesor, explica claro."
  }
}
```

**200**
```json
{
  "label_top": "pos",
  "scores": {"neg": 0.05, "neu": 0.20, "pos": 0.75},
  "confidence": 0.75,
  "latency_ms": 120,
  "correlation_id": "corr_001"
}
```

---

## 4.2 POST /prediccion/batch  (multipart)

Batch actual por archivo. Recomendación futura: permitir `dataset_id` para operar sobre dataset ya ingestada.

**multipart**
- `file`

**(Recomendado futuro)**  
`POST /prediccion/batch?dataset_id=2025-1`

**200**
```json
{
  "batch_id": "batch_1",
  "summary": {
    "rows": 1000,
    "labels": {"neg": 200, "neu": 500, "pos": 300}
  },
  "sample": [{"id": "1", "label": "pos"}],
  "artifact": "/artifacts/batch_1_results.csv",
  "correlation_id": "corr_002"
}
```

---

## 4.3 (Opcional) GET /prediccion/results?dataset_id=...

Si el backend persiste resultados de batch, este endpoint permite consultarlos.

**200**
```json
{
  "dataset_id": "2025-1",
  "batch_id": "batch_1",
  "summary": {},
  "sample": [],
  "artifact": "/artifacts/batch_1_results.csv"
}
```

---

# 5) Dashboard (pendiente de payload final)

Se requieren endpoints **agregados** filtrables por `dataset_id` o `periodo`.

Ejemplos (nombres referenciales):
- `GET /dashboard/kpis?periodo=2025-1`
- `GET /dashboard/riesgo-asignatura?periodo=2025-1`
- `GET /dashboard/ranking-docentes?periodo=2025-1`
- `GET /dashboard/historico?periodo=2025-1`
- `GET /dashboard/real-vs-predicho?periodo=2025-1`
- `GET /dashboard/wordcloud?periodo=2025-1`

**Requisito clave:** todos deben respetar filtros globales (dataset/periodo) para que el dashboard muestre información consistente con la última ingesta.

---

# 6) Requisitos de “perfección” (cross-tab refresh)

Para considerar el sistema completamente funcional:

1. **Ingesta** retorna `dataset_id` y deja dataset en estado `ready` o `processing` (con job).
2. Frontend setea `activeDatasetId` y `activePeriodo` y persiste en storage.
3. Modelos/Predicciones/Dashboard consultan usando esos filtros y actualizan sus visualizaciones.
4. Si hay jobs, la UI refleja estado y refresca automáticamente al completarse.
