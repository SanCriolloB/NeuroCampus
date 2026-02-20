# Dashboard API (histórico-only)

> **Propósito:** Documentar el contrato HTTP del Dashboard de NeuroCampus.  
> **Regla de negocio:** el Dashboard **solo** consulta histórico (no datasets individuales).  
> **Fuentes de datos:**
> - `historico/unificado.parquet` (processed histórico)
> - `historico/unificado_labeled.parquet` (labeled histórico, opcional)

## Base URL

- Desarrollo (Makefile): `http://localhost:8000`
- Prefijo: `/dashboard`

## Filtros estándar (query params)

La mayoría de endpoints aceptan el mismo set de filtros:

- `periodo` *(string, opcional)*: periodo exacto, p. ej. `2024-2`. Si viene, **prioriza** sobre rango.
- `periodo_from` *(string, opcional)*: inicio de rango, incluyente (p. ej. `2024-1`)
- `periodo_to` *(string, opcional)*: fin de rango, incluyente (p. ej. `2025-1`)
- `docente` *(string, opcional)*: nombre del docente tal como aparece en histórico.
- `asignatura` *(string, opcional)*: nombre de la asignatura tal como aparece en histórico.
- `programa` *(string, opcional)*: programa (si existe en histórico).

> **Nota:** si no se envía ningún filtro, el backend responde con agregados del histórico completo (siempre que exista manifest + parquet).

---

## 1) `GET /dashboard/status`

**Descripción:** estado del histórico (liviano; no carga parquets completos).

**200 OK — response**
```json
{
  "manifest_exists": true,
  "manifest_updated_at": "2026-02-16T15:40:00Z",
  "manifest_corrupt": false,
  "periodos_disponibles": ["2024-1","2024-2","2025-1"],
  "processed": {"path":"historico/unificado.parquet","exists":true,"mtime":"2026-02-16T15:39:00Z"},
  "labeled": {"path":"historico/unificado_labeled.parquet","exists":true,"mtime":"2026-02-16T15:39:30Z"},
  "ready_processed": true,
  "ready_labeled": true
}
```

---

## 2) `GET /dashboard/periodos`

**Descripción:** lista de periodos disponibles (desde `historico/manifest.json`).

**200 OK — response**
```json
{ "items": ["2024-1","2024-2","2025-1"] }
```

---

## 3) `GET /dashboard/catalogos`

**Descripción:** catálogos para poblar dropdowns (docentes / asignaturas / programas) basados en histórico **processed**.

**Params:** filtros estándar.

**Ejemplo**
```bash
curl -s "http://localhost:8000/dashboard/catalogos?periodo=2024-2"
```

**200 OK — response**
```json
{
  "docentes": ["DOCENTE 1","DOCENTE 2"],
  "asignaturas": ["ASIGNATURA A","ASIGNATURA B"],
  "programas": ["PROGRAMA X","PROGRAMA Y"]
}
```

---

## 4) `GET /dashboard/kpis`

**Descripción:** KPIs básicos basados en histórico **processed**.

**Params:** filtros estándar.

**Contrato y escala**
- `evaluaciones`: conteo de filas en histórico filtrado.
- `docentes`, `asignaturas`: cardinalidades (si existen columnas).
- `score_promedio`: **escala canónica del Dashboard 0–50**.  
  Si no hay datos para el filtro, se devuelve `null` (no `NaN`).

**Ejemplo**
```bash
curl -s "http://localhost:8000/dashboard/kpis?periodo_from=2024-1&periodo_to=2025-1"
```

**200 OK — response**
```json
{
  "evaluaciones": 534,
  "docentes": 120,
  "asignaturas": 80,
  "score_promedio": 46.1
}
```

---

## 5) `GET /dashboard/series`

**Descripción:** serie agregada por periodo desde histórico **processed**.

**Params**
- `metric` *(string, requerido)*: métrica solicitada. Ejemplos:
  - `evaluaciones`
  - `score_promedio`
  - `docentes`
  - `asignaturas`
- filtros estándar

**Ejemplo**
```bash
curl -s "http://localhost:8000/dashboard/series?metric=score_promedio&periodo_from=2024-1&periodo_to=2025-1"
```

**200 OK — response**
```json
{
  "metric": "score_promedio",
  "points": [
    {"periodo":"2024-1","value":40.0},
    {"periodo":"2024-2","value":46.1},
    {"periodo":"2025-1","value":47.0}
  ]
}
```

---

## 6) `GET /dashboard/rankings`

**Descripción:** ranking derivado del histórico **processed**.

**Params**
- `by` *(string, requerido)*: `docente` | `asignatura`
- `metric` *(string, opcional)*: `score_promedio` | `evaluaciones` (default: `score_promedio`)
- `order` *(string, opcional)*: `asc` | `desc` (default: `desc`)
- `limit` *(int, opcional)*: 1..200 (default: 8)
- filtros estándar

**Ejemplo (Top mejores docentes por score)**
```bash
curl -s "http://localhost:8000/dashboard/rankings?by=docente&metric=score_promedio&order=desc&limit=8&periodo_from=2024-1&periodo_to=2025-1"
```

**Ejemplo (A intervenir: peores docentes por score)**
```bash
curl -s "http://localhost:8000/dashboard/rankings?by=docente&metric=score_promedio&order=asc&limit=8&periodo_from=2024-1&periodo_to=2025-1"
```

**200 OK — response**
```json
{
  "by": "docente",
  "metric": "score_promedio",
  "order": "desc",
  "items": [
    {"name":"DOCENTE A","value":50.0},
    {"name":"DOCENTE B","value":49.2}
  ]
}
```

---

## 7) `GET /dashboard/radar`

**Descripción:** radar de indicadores (promedio de preguntas 1..10), basado en histórico **processed**.

**Params:** filtros estándar.

**Contrato y escala**
- Retorna 10 items con keys: `pregunta_1` ... `pregunta_10`
- `value` está en **0–50** (misma escala del histórico).  
  El frontend puede convertir a 0–5 dividiendo entre 10.

**Ejemplo**
```bash
curl -s "http://localhost:8000/dashboard/radar?periodo_from=2024-1&periodo_to=2025-1"
```

**200 OK — response**
```json
{
  "items": [
    {"key":"pregunta_1","value":46.4},
    {"key":"pregunta_2","value":45.9}
  ]
}
```

---

## 8) `GET /dashboard/wordcloud`

**Descripción:** top términos (wordcloud) desde histórico **labeled**.

**Params**
- `limit` *(int, opcional)*: 1..500 (default: 80)
- filtros estándar

**Ejemplo**
```bash
curl -s "http://localhost:8000/dashboard/wordcloud?limit=30&periodo_from=2024-1&periodo_to=2025-1"
```

**200 OK — response**
```json
{
  "items": [
    {"text":"docente","value":23},
    {"text":"excelente","value":10}
  ]
}
```

**Errores**
- `404`: no existe `historico/unificado_labeled.parquet`
- `400`: el labeled existe pero no hay columna compatible para tokens

---

## 9) `GET /dashboard/sentimiento`

**Descripción:** distribución de sentimiento desde histórico **labeled**.

**Params:** filtros estándar.

**Ejemplo**
```bash
curl -s "http://localhost:8000/dashboard/sentimiento?periodo=2024-2"
```

**200 OK — response**
```json
{
  "buckets": [
    {"label":"neg","value":0.12},
    {"label":"neu","value":0.70},
    {"label":"pos","value":0.18}
  ]
}
```

**Errores**
- `404`: no existe `historico/unificado_labeled.parquet`
- `400`: el labeled existe pero faltan columnas compatibles

---

## Códigos de error comunes

- **404 Not Found**
  - No existe el histórico requerido (`unificado.parquet` o `unificado_labeled.parquet`)
- **400 Bad Request**
  - `metric` inválida en `/series` o `/rankings`
  - Columnas requeridas no existen en el histórico filtrado
- **500 Internal Server Error**
  - Debe evitarse; si aparece, revisar logs del backend (p. ej. NaN serializándose).

