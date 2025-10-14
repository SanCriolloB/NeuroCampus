# Reporte por Docente/Materia/Grupo (NeuroCampus)

Este documento describe cómo **generar, leer e interpretar** el reporte agregado que estima, por combinación de columnas (p. ej., **código de materia** y **grupo**), la probabilidad de que “le vaya bien” a un espacio académico/curso. El reporte se basa en el **sentimiento** extraído de comentarios con BETO y en las **calificaciones** numéricas de las preguntas `pregunta_1..pregunta_10`.

> El reporte se genera con el job CLI `neurocampus.app.jobs.cmd_score_docente`. Los artefactos resultantes se guardan en `artifacts/reports/` (ignorados por Git) y los dummies en `examples/reports/` (versionables).

---

## 1) Objetivo

1. **Agrupar** el dataset etiquetado por columnas de interés (p. ej., `codigo materia`, `grupo`).
2. **Contar positivos** (vía etiqueta `sentiment_label_teacher` o umbral sobre `p_pos`).
3. Estimar **porcentaje positivo** (`pct_pos`) y su **intervalo de confianza** (Jeffreys).
4. Combinar señal de **sentimiento** y **calificaciones medias** para un **score 0–100**: `prob_bueno_pct`.

---

## 2) Requisitos y entrada

- Haber corrido el **preprocesamiento + BETO** y disponer de un parquet en `data/labeled/`, típicamente:
  - `data/labeled/evaluaciones_2025_beto.parquet`
  - (opcional) filtrado a texto aceptado: `data/labeled/evaluaciones_2025_beto_textonly.parquet`
- Columnas necesarias en la entrada:
  - `sentiment_label_teacher` **o** `p_pos` (probabilidad de positivo).
  - `calif_1..calif_10`
  - columnas de **grupo** (p. ej., `codigo materia`, `grupo`). También se soportan variantes con guion bajo: `codigo_materia`.

---

## 3) Comando para generar el reporte

Desde la **raíz del repositorio**:

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_score_docente   --in data/labeled/evaluaciones_2025_beto.parquet   --out artifacts/reports/docente_score.parquet   --group-cols "codigo materia,grupo"   --pos-th 0.55 --alpha 0.05 --mix-w 0.4
```

Parámetros clave:
- `--group-cols`: lista separada por comas de columnas para agrupar (respetar el nombre exacto; si hay espacios, envolver todo en comillas).
- `--pos-th`: si no existe `sentiment_label_teacher`, se considera **positivo** cuando `p_pos ≥ pos_th` (default 0.55).
- `--alpha`: nivel para intervalo Jeffreys (default 0.05 → ~95%).
- `--mix-w`: peso del **componente de calificaciones** frente al de sentimiento en el score combinado (default 0.4).

> Si tus columnas son con guion bajo, usa: `--group-cols "codigo_materia,grupo"`.

---

## 4) Metodología

1. **Selección de positivos**
   - Si existe `sentiment_label_teacher`, se usa **`pos_count = (# filas con label == "pos")`** en cada grupo.
   - Si no, se usa **`pos_count = (# filas con p_pos ≥ pos_th)`**.
2. **Porcentajes**  
   - `pct_pos = pos_count / n` donde `n` es el total de filas por grupo.
3. **Intervalo de confianza (Jeffreys)**  
   - Se usa el intervalo binomial de **Jeffreys**:  
     `lo = Beta.ppf(alpha/2, pos_count + 0.5, n - pos_count + 0.5)`  
     `hi = Beta.ppf(1 - alpha/2, pos_count + 0.5, n - pos_count + 0.5)`
4. **Calificaciones**  
   - Se calculan `calif_i_mean` por grupo y `calif_mean` global (promedio simple de las diez).
   - Se normaliza a `[0,1]`: `calif_mean_0_1 = clip(calif_mean / 5, 0, 1)` (si la rúbrica es 0–5).
5. **Score combinado**  
   - Señal de sentimiento: `S = pct_pos`
   - Señal de calificación: `C = calif_mean_0_1`
   - **Score 0–1**: `score_combinado_0_1 = (1 - mix_w) * S + mix_w * C`  
   - **Prob. bueno 0–100**: `prob_bueno_pct = 100 * score_combinado_0_1`

---

## 5) Esquema de salida (parquet)

| Columna                | Descripción |
|------------------------|-------------|
| columnas de grupo      | p. ej., `codigo materia`, `grupo` |
| `n`                    | total de filas en el grupo |
| `pos_count`            | conteo de positivos en el grupo |
| `pct_pos`              | proporción de positivos (0–1) |
| `pct_pos_lo`/`pct_pos_hi` | intervalo Jeffreys (0–1) |
| `calif_1_mean..calif_10_mean` | medias por pregunta |
| `calif_mean`           | media global de calificaciones |
| `calif_mean_0_1`       | media global escalada a [0,1] |
| `p_pos_mean`           | media de `p_pos` en el grupo (si existe `p_pos`) |
| `score_combinado_0_1`  | score mixto sentimiento+calificaciones (0–1) |
| `prob_bueno_pct`       | score en 0–100 |

> La tabla puede incluir columnas extra según disponibilidad en la entrada (p. ej., metadatos).

---

## 6) Lectura y exportación

Ver los primeros registros y contar grupos:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("artifacts/reports/docente_score.parquet")
print(df.head(10))
print("n_groups:", len(df))
PY
```

Exportar a CSV (si necesitas compartirlo):
```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("artifacts/reports/docente_score.parquet")
df.to_csv("artifacts/reports/docente_score.csv", index=False, encoding="utf-8-sig")
print("OK CSV -> artifacts/reports/docente_score.csv")
PY
```

---

## 7) Interpretación de resultados

- **`prob_bueno_pct`** es un **ranking** útil para priorizar cursos/grupos con mayor probabilidad de “salir bien”.  
- Usa **`pct_pos_lo`** (límite inferior del intervalo) para comparaciones **conservadoras** entre grupos.
- Revisa **`n`**: grupos con pocos datos tienen más incertidumbre (intervalos más anchos).
- Si `calif_mean` y `pct_pos` divergen, investiga **casos atípicos** (comentarios muy polarizados).

Ejemplo de lectura rápida:
- `pct_pos = 0.60`, `calif_mean_0_1 = 0.62`, `mix_w = 0.4` → `score ≈ 0.6*0.6 + 0.4*0.62 = 0.608` → `prob_bueno_pct ≈ 60.8`

---

## 8) Dummies versionables

Para demos y CI, puedes crear un **dummy** con mismas columnas de la salida real y guardarlo en `examples/reports/`:

```bash
python - <<'PY'
import math
import pandas as pd
from pathlib import Path

cols = pd.read_parquet("artifacts/reports/docente_score.parquet").columns.tolist()

def make_row(cm, grupo, n, pos, calif_mean_val=3.0):
    row = {c: pd.NA for c in cols}
    if "codigo materia" in row: row["codigo materia"] = cm
    if "codigo_materia" in row: row["codigo_materia"] = cm
    if "grupo" in row: row["grupo"] = str(grupo)
    if "n" in row: row["n"] = int(n)
    if "pos_count" in row: row["pos_count"] = int(pos)
    pct = pos / n if n else 0.0
    if "pct_pos" in row: row["pct_pos"] = pct
    if "pct_pos_lo" in row: row["pct_pos_lo"] = max(0.0, pct - 0.12)
    if "pct_pos_hi" in row: row["pct_pos_hi"] = min(1.0, pct + 0.12)
    means = []
    for i in range(1, 11):
        c = f"calif_{i}_mean"
        if c in row:
            v = calif_mean_val + (0.05 * (i % 3) - 0.05)
            row[c] = v
            means.append(v)
    if "calif_mean" in row and means:
        row["calif_mean"] = sum(means)/len(means)
    if "calif_mean_0_1" in row:
        base = float(row.get("calif_mean", calif_mean_val))
        row["calif_mean_0_1"] = min(1.0, max(0.0, base/5.0))
    if "p_pos_mean" in row:
        row["p_pos_mean"] = pct
    score = 0.6*float(row.get("calif_mean_0_1", pct)) + 0.4*float(row.get("pct_pos", pct))
    if "score_combinado_0_1" in row: row["score_combinado_0_1"] = score
    if "prob_bueno_pct" in row: row["prob_bueno_pct"] = round(100*score, 1)
    return row

rows = [
    make_row("MAT101", "1", n=50, pos=35, calif_mean_val=3.4),
    make_row("MAT101", "2", n=30, pos=18, calif_mean_val=3.2),
    make_row("FIS201", "1", n=40, pos=22, calif_mean_val=3.0),
    make_row("FIS201", "2", n=20, pos=12, calif_mean_val=3.6),
]

df = pd.DataFrame(rows)
Path("examples/reports").mkdir(parents=True, exist_ok=True)
out = "examples/reports/docente_score_example.parquet"
df.to_parquet(out, index=False)
print("OK ->", out)
PY
```

> Commitea solo los **dummies** en `examples/`. El reporte real en `artifacts/reports/` se mantiene **fuera** del repo.

---

## 9) Troubleshooting

- **No encuentra columnas de grupo** → imprime columnas disponibles y ajusta `--group-cols` (con comillas si hay espacios):  
  ```bash
  python - <<'PY'
  import pandas as pd
  df = pd.read_parquet("data/labeled/evaluaciones_2025_beto.parquet")
  print(list(df.columns))
  PY
  ```
- **Todo `prob_bueno_pct=60`** → parámetros por defecto (`mix_w` y valores medios) pueden armonizar grupos. Ajusta `--pos-th` y revisa variedad en `calif_mean`.
- **Pocos datos por grupo** → intervalos amplios; considera agrupar más grueso (p. ej., solo por `codigo materia`).
- **Rangos de calificación diferentes a 0–5** → adapta la normalización a tu escala.

---

## 10) Checklist de publicación

- [x] `artifacts/reports/docente_score.parquet` generado sin errores.
- [x] Revisión rápida de `n`, `pct_pos`, intervalos y `prob_bueno_pct`.
- [x] (Opcional) Dummy en `examples/reports/` para documentación/demos.
- [x] README/Docs actualizados con comando y criterios de interpretación.

