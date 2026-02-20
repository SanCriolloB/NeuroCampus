# Dashboard - Troubleshooting

Esta guía recopila los errores más comunes al desarrollar/depurar la pestaña **Dashboard** (backend + frontend)
y cómo verificarlos rápidamente.

> Principio del Dashboard: **solo histórico** (`historico/unificado.parquet` y `historico/unificado_labeled.parquet`).

---

## 1) `ModuleNotFoundError: No module named 'neurocampus'`

### Causa
Ejecutar Python desde la raíz sin `PYTHONPATH` apuntando a `backend/src`.

### Solución
En Windows (Git Bash):

```bash
PYTHONPATH=backend/src python -c "import neurocampus; print('ok')"
```

Si usas `make be-dev`, ya configura `PYTHONPATH="src;$PYTHONPATH"` dentro de `backend/`.

---

## 2) `/dashboard/kpis` responde 500 con `Out of range float values are not JSON compliant: nan`

### Causa
Se retornó `NaN` en `score_promedio` (por ejemplo, al calcular `mean()` sobre una serie vacía o toda NaN).
Starlette serializa JSON con `allow_nan=False`, por lo que `NaN/inf` hacen fallar la respuesta.

### Verificación rápida (sin servidor)

```bash
PYTHONPATH=backend/src python -c "from neurocampus.dashboard.queries import load_processed, apply_filters, DashboardFilters, compute_kpis; df=apply_filters(load_processed(), DashboardFilters(periodo_from='2024-1',periodo_to='2025-1',docente='ROMERO MENDEZ OMAR AUGUSTO',asignatura='ACTIVIDADES COMPLEMENTARIAS I')); print('rows=', len(df)); print(compute_kpis(df))"
```

### Fix esperado
`compute_kpis()` debe convertir `NaN/inf` a `None` (y mantener respuesta 200).

---

## 3) Wordcloud devuelve `[]` aunque exista texto en labeled

### Causa típica
`sugerencias_lemmatizadas` existe pero está completamente vacía (`None`) para el subconjunto filtrado.
Si el código elige esa columna solo por existencia, termina contando 0 tokens.

### Verificaciones

1) Confirmar si hay texto en labeled:

```bash
PYTHONPATH=backend/src python -c "import pandas as pd; df=pd.read_parquet('historico/unificado_labeled.parquet', columns=['periodo','sugerencias_lemmatizadas','texto_lemmas']); print('shape=', df.shape); print('nonnull sugerencias=', df['sugerencias_lemmatizadas'].notna().sum()); print('nonnull texto_lemmas=', df['texto_lemmas'].notna().sum())"
```

2) Probar la agregación:

```bash
PYTHONPATH=backend/src python -c "from neurocampus.dashboard.aggregations import wordcloud_terms; from neurocampus.dashboard.queries import DashboardFilters; print(wordcloud_terms(DashboardFilters(periodo_from='2024-1', periodo_to='2024-2'), limit=15))"
```

### Fix esperado
El selector de columna debe elegir la *primera* columna con contenido tokenizable (fallback a `texto_lemmas`).

---

## 4) Radar devuelve ceros al probar el router directamente

### Causa
Al llamar funciones del router directamente (sin FastAPI), los parámetros con `Query(...)` pueden quedar como
objetos `Query` en vez de `None`, lo que afecta filtros.

### Solución
Al probar, pasar `None` explícitamente en los parámetros opcionales:

```bash
PYTHONPATH=backend/src python -c "from neurocampus.app.routers.dashboard import dashboard_radar; r=dashboard_radar(periodo=None, docente=None, asignatura=None, programa=None, periodo_from='2024-1', periodo_to='2024-2'); print([(i.key, i.value) for i in r.items[:3]])"
```

---

## 5) Escalas inconsistentes en UI (0–100 / “normalización”)

### Causa
El backend entrega métricas de score en escala **0–50** para Dashboard (score_promedio, rankings).
Si el frontend normaliza a 0–100 (o 70–95) se distorsionan comparaciones y rankings.

### Verificación (API)
Probar que rankings y series estén en 0–50:

```bash
curl -s "http://localhost:8000/dashboard/series?metric=score_promedio&periodo_from=2024-1&periodo_to=2025-1"
curl -s "http://localhost:8000/dashboard/rankings?by=docente&metric=score_promedio&order=desc&limit=5&periodo_from=2024-1&periodo_to=2025-1"
```

### Fix esperado (UI)
- Labels `/50`
- Domains de charts en `[0, 50]`
- Radar: convertir a 0–5 con `/10`

---

## 6) “Top Mejores” y “A intervenir” muestran lo mismo

### Causa
El frontend estaba reordenando localmente o no enviaba `order` al backend.

### Fix esperado
- Top Mejores: `order=desc`
- A intervenir: `order=asc`

Verifica en logs del backend que el query cambie el `order`.

---

## 7) Checklist rápido de archivos históricos

Verifica que existan los parquet:

```bash
ls -la historico/unificado.parquet
ls -la historico/unificado_labeled.parquet
```

Verifica columnas relevantes:

```bash
python -c "import pandas as pd; df=pd.read_parquet('historico/unificado.parquet'); print([c for c in df.columns if c.startswith('pregunta_')][:15])"
python -c "import pandas as pd; df=pd.read_parquet('historico/unificado_labeled.parquet'); print([c for c in df.columns if 'lem' in c or 'text' in c][:30])"
```

---

## 8) Smoke test recomendado (backend)

Con el backend corriendo en `http://localhost:8000`:

```bash
curl -s "http://localhost:8000/dashboard/status"
curl -s "http://localhost:8000/dashboard/periodos"
curl -s "http://localhost:8000/dashboard/catalogos?periodo_from=2024-1&periodo_to=2025-1"
curl -s "http://localhost:8000/dashboard/kpis?periodo_from=2024-1&periodo_to=2025-1"
curl -s "http://localhost:8000/dashboard/radar?periodo_from=2024-1&periodo_to=2025-1"
curl -s "http://localhost:8000/dashboard/wordcloud?periodo_from=2024-1&periodo_to=2025-1&limit=20"
```

Si todos responden 200, el Dashboard está listo para consumo por frontend.
