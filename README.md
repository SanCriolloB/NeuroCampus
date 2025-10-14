# NeuroCampus

MVP para analizar evaluaciones estudiantiles con **FastAPI (backend)**, **RBM Student** y **NLP (BETO)**.  
Incluye pipeline de preprocesamiento, entrenamiento y endpoints de predicción con regla costo-sensible.

---

## Requisitos

- Python 3.10+ (recomendado 3.10–3.12)
- Node 18+ (frontend)
- Git Bash / WSL (Windows) o shell POSIX
- Dependencias Python (backend):
  - `torch`, `transformers`, `pandas`, `pyarrow`, `fastapi`, `uvicorn`, `scikit-learn`, `scipy` (para el reporte)

> **Windows (Git Bash):** usa **comillas simples** en `printf`/`echo` para evitar `event not found` por `!`.

---

## Estructura de carpetas (resumen)

```
backend/
  src/neurocampus/
    app/              # FastAPI, routers, jobs CLI
    models/           # estrategias RBM, entrenamiento
    prediction/       # fachada de predicción
    services/nlp/     # preprocesamiento y teacher (BETO)
artifacts/
  jobs/               # corridas de entrenamiento (salida)   ← ignorado por Git
  champions/          # modelo “campeón” activo              ← ignorado por Git
  reports/            # reportes agregados                    ← ignorado por Git
data/
  processed/          # datasets estandarizados               ← evitar versionar reales
  labeled/            # etiquetados (teacher/BETO)            ← evitar versionar reales
examples/
  reports/            # artefactos dummy versionables
frontend/             # app web (Vite + React + TS)
```

---

## Setup rápido

### 1) Backend

```bash
# Linux/macOS
python -m venv .venv && source .venv/bin/activate
# Windows PowerShell
# python -m venv .venv ; .\.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
```

Crea ignores para artefactos:

```bash
mkdir -p artifacts/{jobs,champions,reports}
printf '*\n!.gitkeep\n' > artifacts/jobs/.gitignore
printf '*\n!.gitkeep\n' > artifacts/champions/.gitignore
printf '*\n!.gitkeep\n' > artifacts/reports/.gitignore
touch artifacts/jobs/.gitkeep artifacts/champions/.gitkeep artifacts/reports/.gitkeep
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Pipeline de datos (end-to-end)

> Todos los comandos asumen que ejecutas desde la **raíz del repo**.  
> Cuando uses módulos Python, define `PYTHONPATH="$PWD/backend/src"`.

### A) Cargar CSV crudo → parquet estandarizado

Convierte tu CSV de evaluaciones a un parquet con:
- `comentario`
- `calif_1..calif_10` (solo columnas `pregunta_1..10` o `pregunta 1..10`)
- (opcional) metadatos que quieras preservar

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_cargar_dataset   --in examples/Evaluacion.csv   --out data/processed/evaluaciones_2025.parquet   --meta-list "codigo_materia,docente,grupo,periodo"
```

> El cargador admite nombres de pregunta con **espacio o guion bajo** (ej. `pregunta 1` / `pregunta_1`).

### B) Preprocesamiento + BETO (teacher)

Limpia, lematiza y etiqueta con **BETO** (modo **probs** recomendado).  
Filtra por número mínimo de tokens y aplica “gating” por confianza.

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_preprocesar_beto   --in data/processed/evaluaciones_2025.parquet   --out data/labeled/evaluaciones_2025_beto.parquet   --beto-mode probs   --threshold 0.90 --margin 0.25 --neu-min 0.90   --min-tokens 1
```

Genera un subset **texto-válido** (aceptado por el teacher):

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/labeled/evaluaciones_2025_beto.parquet")
df[(df["has_text"]==1) & (df["accepted_by_teacher"]==1)]   .to_parquet("data/labeled/evaluaciones_2025_beto_textonly.parquet", index=False)
print("OK -> data/labeled/evaluaciones_2025_beto_textonly.parquet")
PY
```

### C) Entrenamiento RBM (Student)

Modelo recomendado (estable actual): **texto + num**, `minmax`, 100 épocas.

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.train_rbm   --type general   --data data/labeled/evaluaciones_2025_beto_textonly.parquet   --job-id auto   --seed 42   --epochs 100 --n-hidden 64   --cd-k 1 --epochs-rbm 1   --batch-size 128   --lr-rbm 5e-3 --lr-head 1e-2   --scale-mode minmax   --use-text-probs
```

El job crea una carpeta `artifacts/jobs/<JOB_ID>` con:
- `vectorizer.json`, `rbm.pt`, `head.pt`
- `job_meta.json`, `metrics.json`

### D) Promover “campeón”

```bash
JOB="artifacts/jobs/<JOB_ID>"              # ← pon aquí el último job
DEST="artifacts/champions/with_text/current"

rm -rf "$DEST" && mkdir -p "$DEST"
cp -r "$JOB"/* "$DEST"/

# (opcional) descriptor
cat > "$DEST/CHAMPION.json" <<'JSON'
{ "family":"with_text", "selected_at":"(UTC)", "reason":"best macro-F1", "notes":"text+num,minmax,100ep" }
JSON

# (opcional) variable de entorno para runtime
echo 'CHAMPION_WITH_TEXT=artifacts/champions/with_text/current' >> .env
```

---

## Backend (FastAPI)

Levanta el API:

```bash
uvicorn neurocampus.app.main:app --reload --app-dir backend/src
# Docs en: http://127.0.0.1:8000/docs
```

### Predicción online (con regla costo-sensible)

El endpoint espera **`input`** con `calificaciones` y `comentario`.  
El servidor devuelve `proba: [p_neg,p_neu,p_pos]` y `label` ajustada por reglas:

- si `p_pos ≥ 0.55` ⇒ **pos**
- si no y `p_neg ≥ 0.35` o `p_neg - p_neu ≥ 0.05` ⇒ **neg**
- en otro caso ⇒ **neu**

**Ejemplo (Git Bash, heredoc):**
```bash
curl -s -X POST "http://127.0.0.1:8000/prediccion/online"   -H "Content-Type: application/json; charset=utf-8"   --data-binary @- <<'JSON'
{"input":{
  "calificaciones":{"pregunta_1":4.5,"pregunta_2":4.0,"pregunta_3":3.8,"pregunta_4":4.2,"pregunta_5":4.6,
                    "pregunta_6":4.3,"pregunta_7":4.1,"pregunta_8":4.4,"pregunta_9":4.0,"pregunta_10":4.5},
  "comentario":"La metodología fue clara y el profesor resolvió dudas con paciencia."
}}
JSON
```

> Si quieres aceptar también el formato “plano” sin `input`, puedes agregar un endpoint alterno `/prediccion/online_v2` que envuelva el body.

---

## Reporte “¿le irá bien?” por (docente/materia/grupo)

Job batch que agrega por columnas de grupo (usa `sentiment_label_teacher` o `p_pos`):

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.app.jobs.cmd_score_docente   --in data/labeled/evaluaciones_2025_beto.parquet   --out artifacts/reports/docente_score.parquet   --group-cols "codigo materia,grupo"   --pos-th 0.55 --alpha 0.05 --mix-w 0.4
```

Salida (parquet) con:
- `n`, `pos_count`, `pct_pos`, `pct_pos_lo/hi` (Jeffreys CI),
- medias `calif_*_mean`, `prob_bueno_pct` (score combinado 0–100).

> Para versionar ejemplos sin datos reales, copia un **dummy** a `examples/reports/`.

---

## Buenas prácticas de Git

Ignora artefactos/datos reales y caches:

```
**/__pycache__/
.venv/
.env
artifacts/*
!artifacts/**/.gitkeep
!artifacts/**/.gitignore
data/**/*.parquet
data/**/*.csv
frontend/node_modules/
```

> Versiona **solo** ejemplos sintéticos (`examples/`), código y documentación.

---

## Solución de problemas (rápido)

- **Git Bash**: usa comillas **simples** en `printf '*
!.gitkeep
'`.
- **Rutas**: ejecuta desde la **raíz**. Si estás en subcarpetas, usa rutas relativas correctas.
- **HuggingFace symlinks (warning)** en Windows: es solo aviso; funciona con caché “degradada”.
- **422 “Field required: input”**: el body del endpoint debe ir envuelto en `{"input": {...}}`.
- **Pocos comentarios útiles**: sube `--min-tokens`, ajusta `threshold/margin/neu-min` o entrena con `--use-text-probs`.

---

## Roadmap corto

- Random Search 3-fold para hparams (15–25 trials).
- Exponer reportes en UI (ranking por `prob_bueno_pct`).
- Endpoint “batch” con subida de CSV y barra de progreso.
- Dockerización (backend y volumen de `artifacts/`).

---
