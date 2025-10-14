# Preprocesamiento (NeuroCampus)

Este documento describe el **pipeline de preprocesamiento** de textos y calificaciones para el proyecto **NeuroCampus**, desde el CSV crudo hasta el dataset etiquetado con **BETO** que consumen los modelos **RBM Student**.

> Todo lo aquí descrito es reproducible mediante **jobs CLI** incluidos en el backend. Los artefactos resultantes se guardan en `data/processed/`, `data/labeled/` y `artifacts/` (estos últimos ignorados por Git).

---

## Objetivo

1. **Estandarizar** columnas de entrada (comentario y `pregunta_1..pregunta_10`).
2. **Limpiar y lematizar** el texto.
3. **Etiquetar sentimiento** con **BETO** (transformers) y generar **probabilidades** por clase.
4. **Aplicar reglas de aceptación** (gating) para quedarnos con etiquetas confiables.
5. Dejar un **Parquet final** listo para entrenamiento/inferencia con RBM.

---

## Entradas y salidas

### Entrada mínima (CSV crudo)
- 1 columna de texto (p. ej., `Sugerencias`, `comentario`, `observaciones`).
- 10 columnas de calificaciones: **`pregunta_1..pregunta_10`** _(también se aceptan con espacio: `pregunta 1` … `pregunta 10`)_.
- (Opcional) Metadatos: `codigo_materia`, `docente`, `grupo`, `periodo`, etc.

### Salidas principales
- `data/processed/evaluaciones_2025.parquet` → **estandarizado** (`comentario`, `calif_1..calif_10`, metadatos).  
- `data/labeled/evaluaciones_2025_beto.parquet` → **etiquetado con BETO**:
  - Texto limpio y lematizado: `_texto_clean`, `_texto_lemmas`, `token_count`, `has_text`.
  - Probabilidades: `p_neg`, `p_neu`, `p_pos`.
  - Etiqueta teacher: `sentiment_label_teacher` (`neg`, `neu`, `pos`) + `sentiment_conf`.
  - Señal de aceptación: `accepted_by_teacher` ∈ {0,1}.
  - `comentario` (trazabilidad humana) + metadatos (si existían).
- (Opcional) `data/labeled/evaluaciones_2025_beto_textonly.parquet` → **subset aceptado** (`has_text==1` y `accepted_by_teacher==1`).

Además se genera `data/labeled/evaluaciones_2025_beto.parquet.meta.json` con resumen del job (modelo BETO, thresholds, etc.).

---

## Paso 0 — Requisitos previos

Instala dependencias del backend (FastAPI + ML + NLP). En Windows se recomienda **Git Bash** o **PowerShell** para comandos.

```bash
python -m venv .venv && source .venv/bin/activate           # Linux/macOS
# .\.venv\Scripts\Activate.ps1                              # Windows PowerShell

pip install -r backend/requirements.txt
```

> En Windows es normal ver un **warning** de HuggingFace sobre *symlinks*; no bloquea la ejecución (usa caché “degradada”).

---

## Paso 1 — Carga y estandarización del CSV

**Script:** `neurocampus.app.jobs.cmd_cargar_dataset`

- Detecta **columna de texto** (busca candidatos comunes; se puede forzar con `--text-col`, si existiera).
- Normaliza nombres de **preguntas**: acepta `pregunta_1..10` **y** `pregunta 1..10`. Las mapea a `calif_1..calif_10`.
- Filtra automáticamente columnas numéricas **no preguntas** (ID, etc.).
- Permite **preservar metadatos** con `--meta-list` (solo los que existan realmente).

**Ejemplo:**

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_cargar_dataset   --in examples/Evaluacion.csv   --out data/processed/evaluaciones_2025.parquet   --meta-list "codigo_materia,docente,grupo,periodo"
```

**Salida:** Parquet con columnas: `comentario`, `calif_1..calif_10` y los metadatos disponibles.

---

## Paso 2 — Limpieza y lematización

**Script:** `neurocampus.app.jobs.cmd_preprocesar_beto` (internamente usa `neurocampus.services.nlp.preprocess`)

Operaciones típicas de `limpiar_texto`:
- Normaliza espacios, minúsculas, quita ruido común, conserva caracteres relevantes.
- Opcionalmente remueve URLs, correos y tokens de poco valor semántico.

`tokenizar_y_lematizar_batch` produce `_texto_lemmas`, que es la cadena **lematizada** (p. ej., “explicar duda clase” en vez de “explicó dudas en las clases”).  
También se calcula `token_count` y se derivan `has_text = 1{token_count >= min_tokens}`.

---

## Paso 3 — Etiquetado con BETO (Teacher)

**Script:** `neurocampus.app.jobs.cmd_preprocesar_beto`

Dos modos:
- `--beto-mode probs` (**recomendado**): calcula probabilidades `p_neg`, `p_neu`, `p_pos` y etiqueta `sentiment_label_teacher` por **argmax** + **gating** (ver abajo).
- `--beto-mode simple`: replica el pipeline *simple* del notebook (top-1 + score) **sin** probas completas (no recomendado para producción).

**Parámetros clave:**

- `--beto-model` (default `finiteautomata/beto-sentiment-analysis`)  
- `--batch-size` (default 32; ajusta según memoria)  
- **Gating de aceptación**:
  - `--threshold` (confianza mínima del top-1, p. ej., 0.90)
  - `--margin` (margen entre top-1 y segundo, p. ej., 0.25)
  - `--neu-min` (mínimo absoluto para aceptar `neu`, p. ej., 0.90)
- `--min-tokens` (mínimo de tokens lematizados para considerar el texto, p. ej., 1 o 2)

**Lógica de aceptación (probs):**  
Sea `P = [p_neg, p_neu, p_pos]` y `c = argmax(P)`:
- Si `c=pos`: aceptar si `p_pos ≥ threshold` **y** `p_pos - max(p_neg,p_neu) ≥ margin`.
- Si `c=neg`: aceptar si `p_neg ≥ threshold` **y** `p_neg - max(p_pos,p_neu) ≥ margin`.
- Si `c=neu`: aceptar si `p_neu ≥ neu_min`.

Si se acepta ⇒ `accepted_by_teacher=1`, si no ⇒ `0`.

**Ejemplo (probs + filtros):**

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_preprocesar_beto   --in data/processed/evaluaciones_2025.parquet   --out data/labeled/evaluaciones_2025_beto.parquet   --beto-mode probs   --threshold 0.90 --margin 0.25 --neu-min 0.90   --min-tokens 1
```

**Consola esperada:**
```
{'out': 'data/labeled/evaluaciones_2025_beto.parquet',
 'n_rows': 938,
 'accept_rate': 0.95...,           # fracción aceptada por teacher
 'text_coverage': 0.99...,         # fracción con has_text==1
 'text_col': 'comentario'}
```

---

## Paso 4 — Subset “texto aceptado” (opcional, recomendado)

Para entrenar un Student **más estable** con texto, crea el subset:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/labeled/evaluaciones_2025_beto.parquet")
df[(df["has_text"]==1) & (df["accepted_by_teacher"]==1)]   .to_parquet("data/labeled/evaluaciones_2025_beto_textonly.parquet", index=False)
print("OK -> data/labeled/evaluaciones_2025_beto_textonly.parquet")
PY
```

---

## Esquema de columnas (dataset etiquetado)

| Columna                    | Tipo       | Descripción |
|---------------------------|------------|-------------|
| `comentario`              | string     | Texto original limpio para trazabilidad humana. |
| `_texto_clean`            | string     | Texto normalizado previo a lematización. |
| `_texto_lemmas`           | string     | Texto lematizado para el Teacher. |
| `token_count`             | int        | Conteo de tokens en `_texto_lemmas`. |
| `has_text`                | {0,1}      | 1 si `token_count ≥ min_tokens`. |
| `p_neg`, `p_neu`, `p_pos` | float      | Probabilidades por clase (modo `probs`). |
| `sentiment_label_teacher` | string     | `neg`, `neu`, `pos` (argmax). |
| `sentiment_conf`          | float      | `max(p_neg,p_neu,p_pos)`. |
| `accepted_by_teacher`     | {0,1}      | Señal de aceptación por reglas. |
| `calif_1..calif_10`       | float      | Calificaciones estandarizadas. |
| metadatos (opc.)          | varios     | p. ej., `codigo_materia`, `docente`, `grupo`, `periodo`. |

---

## Validación rápida

Distribución de etiquetas y aceptación:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/labeled/evaluaciones_2025_beto.parquet")
print("n_rows:", len(df))
print("text_coverage:", df["has_text"].mean())
print("accept_rate:", df["accepted_by_teacher"].mean())
print("labels:", df["sentiment_label_teacher"].value_counts(dropna=False).to_dict())
PY
```

---

## Conexión con entrenamiento (RBM Student)

El entrenamiento soporta incluir las probabilidades de texto (`--use-text-probs`) como **3 features adicionales**. Ejemplo recomendado (modelo estable actual):

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.train_rbm   --type general   --data data/labeled/evaluaciones_2025_beto_textonly.parquet   --job-id auto   --seed 42   --epochs 100 --n-hidden 64   --cd-k 1 --epochs-rbm 1   --batch-size 128   --lr-rbm 5e-3 --lr-head 1e-2   --scale-mode minmax   --use-text-probs
```

> El job guarda `job_meta.json` y `metrics.json` y puede promoverse a campeón copiando a `artifacts/champions/with_text/current`.

---

## Problemas comunes y soluciones

- **“No se encuentra la columna de texto”** → usa `--text-col` para forzarla.  
- **Muchas filas con texto vacío** → sube `--min-tokens` (p. ej., 2) para filtrar `“ok”, “ninguna”`, etc.  
- **Pocas muestras aceptadas** → baja `--threshold` o `--margin` ligeramente (cuidando la calidad).  
- **Memoria insuficiente** con BETO → reduce `--batch-size`.  
- **Windows/HF symlinks warning** → es esperado; la caché funciona igualmente.  
- **Rutas relativas** → ejecuta desde la **raíz** del repo o usa paths absolutos.

---

## Ejemplo end-to-end (completo)

```bash
# 1) CSV → parquet estandarizado
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_cargar_dataset   --in examples/Evaluacion.csv   --out data/processed/evaluaciones_2025.parquet   --meta-list "codigo_materia,docente,grupo,periodo"

# 2) Preprocesar + BETO (probs) con filtros
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_preprocesar_beto   --in data/processed/evaluaciones_2025.parquet   --out data/labeled/evaluaciones_2025_beto.parquet   --beto-mode probs   --threshold 0.90 --margin 0.25 --neu-min 0.90   --min-tokens 1

# 3) Subset de texto aceptado
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/labeled/evaluaciones_2025_beto.parquet")
df[(df["has_text"]==1) & (df["accepted_by_teacher"]==1)]   .to_parquet("data/labeled/evaluaciones_2025_beto_textonly.parquet", index=False)
print("OK subset textonly")
PY
```

---

## Buenas prácticas

- Mantén datasets reales fuera del repo (solo **examples/** versionados).  
- Documenta los parámetros usados (quedan en el `.meta.json`).  
- Usa **semillas** (`--seed`) para reproducibilidad en entrenamiento.  
- Promueve campeón solo cuando **mejore macro-F1** de referencia.

