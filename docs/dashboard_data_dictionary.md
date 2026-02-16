# Dashboard — Diccionario de datos (histórico)

Este documento describe las columnas y métricas principales usadas por la pestaña **Dashboard** en NeuroCampus.

> Alcance: **histórico-only** (archivos en `historico/`).
>
> - Processed: `historico/unificado.parquet`
> - Labeled: `historico/unificado_labeled.parquet`

---

## 1) `historico/unificado.parquet` (processed)

### Identificadores y llaves
- `id`: identificador del registro (evaluación).
- `periodo`: periodo/semestre en formato `YYYY-<n>` (ej: `2024-2`).
- `codigo_materia`: código de la asignatura.
- `materia`: nombre de la asignatura.
- `grupo`: grupo/sección.
- `cedula_profesor`: identificador del docente.
- `profesor`: nombre del docente.

### Encuesta (preguntas 1–10)
- `pregunta_1` … `pregunta_10`

**Escala y semántica**
- En el histórico procesado estas columnas suelen venir en una escala **0–50** (por ejemplo `47` representa `4.7/5.0`).
- En el dashboard:
  - **Radar**: escala 0–5 → se obtiene dividiendo por `10`.
  - **Agregaciones/series/rankings**: se mantienen en **0–50**.

> Nota: si en algún histórico estas columnas vienen en 1–5, el backend debe normalizarlas antes de exponerlas.

### Texto libre
- `observaciones`: comentario textual (si existe).
- `unnamed_17` … `unnamed_25`: columnas residuales (normalmente deben considerarse ruido/legacy).

---

## 2) `historico/unificado_labeled.parquet` (labeled)

Este parquet corresponde a histórico con procesamiento de texto y señales de sentimiento/clasificación.

### Identificadores y llaves (alineadas con processed)
- `id`
- `periodo`
- `profesor`, `docente` (aliases posibles)
- `materia` / `asignatura` (aliases posibles)
- `codigo_materia`, `grupo`, `cedula_profesor`

### Texto y features
- `comentario`: texto original del comentario (si existe).
- `has_text`: indicador de si existe texto (0/1).
- `has_text_processed`: indicador de si pasó por pipeline de limpieza (0/1).
- `texto_raw_concat`: concatenación de texto bruto (cuando aplica).
- `texto_clean`: texto limpiado.
- `texto_lemmas`: lematización (texto tokenizado/normalizado).
- `sugerencias_lemmatizadas`: sugerencias/lemmas (puede existir pero estar vacío).

### Sentimiento
- `p_neg`, `p_neu`, `p_pos`: probabilidades por clase.
- `sentiment_label_teacher`: etiqueta final (ej: `neg/neu/pos`).
- `sentiment_conf`: confianza.
- `accepted_by_teacher`: indicador/flag de aceptación (cuando aplica).

### Scores
- `score_total_0_50`: score total en 0–50 (si viene del pipeline).
- `rating`: rating (puede ser 0–5 u otra escala según pipeline).
- `score_total`: score total usado por el Dashboard (escala **0–50**).
- `pregunta_1` … `pregunta_10`, `calif_1` … `calif_10`: calificaciones por ítem (dependiendo del origen).

---

## 3) Métricas del Dashboard (contrato)

### `score_promedio`
- Métrica agregada que representa el promedio del score en escala **0–50**.
- Se usa en:
  - `/dashboard/kpis` → `score_promedio`
  - `/dashboard/series?metric=score_promedio`
  - `/dashboard/rankings?metric=score_promedio`

### Radar (perfil de indicadores)
- Endpoint: `/dashboard/radar`
- Devuelve 10 ítems (`pregunta_1`…`pregunta_10`) con valores en escala **0–50**.
- El frontend los convierte a 0–5 para el chart dividiendo por `10`.

### Wordcloud
- Endpoint: `/dashboard/wordcloud`
- Fuente recomendada:
  1) `sugerencias_lemmatizadas` (si tiene contenido)
  2) `texto_lemmas` (fallback)
- Respuesta: lista de `{text, value}` donde `value` es frecuencia.

---

## 4) Reglas de compatibilidad (aliases)
El backend intenta resolver variaciones de nombres entre históricos:

- Docente: `docente` o `profesor`
- Asignatura: `asignatura` o `materia`

Si un histórico no trae una de estas columnas, el filtro debe usar el alias disponible.

---

## 5) Checklist rápido de verificación

### Columns processed
```bash
python -c "import pandas as pd; df=pd.read_parquet('historico/unificado.parquet'); print(df.columns.tolist())"
```

### Columns labeled
```bash
python -c "import pandas as pd; df=pd.read_parquet('historico/unificado_labeled.parquet'); print(df.columns.tolist())"
```

### Validar escala de preguntas (ejemplo)
```bash
python -c "import pandas as pd; df=pd.read_parquet('historico/unificado.parquet', columns=['pregunta_1','pregunta_10']); print(df[['pregunta_1','pregunta_10']].describe())"
```
