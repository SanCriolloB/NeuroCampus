# Entrenamiento (NeuroCampus)

Este documento describe **cómo entrenar** el modelo *Student* basado en **RBM** (Restricted Boltzmann Machine) con las calificaciones numéricas y, opcionalmente, con las **probabilidades de sentimiento** generadas por BETO en el preprocesamiento.

> Requisitos previos: haber ejecutado el pipeline de **preprocesamiento** y contar con los parquet en `data/labeled/` (ver `Preprocesamiento.md`).

---

## 1) Datos de entrada al entrenamiento

- **Parquet etiquetado** por el *Teacher* (BETO), típico:
  - `data/labeled/evaluaciones_2025_beto.parquet`
  - `data/labeled/evaluaciones_2025_beto_textonly.parquet` (opcional **recomendado**: filas con `has_text==1` y `accepted_by_teacher==1`)
- **Columnas requeridas**:
  - `calif_1..calif_10` (numéricas).
  - **Target**: etiqueta de sentimiento `sentiment_label_teacher` en {`neg`,`neu`,`pos`} (proviene del teacher).
- **Columnas opcionales** (si se usa texto en el Student):
  - `p_neg`, `p_neu`, `p_pos` → se agregan como **3 features adicionales** con `--use-text-probs`.

> Sugerencia: Entrenar con **`*_textonly.parquet`** mejora estabilidad cuando el dataset original tiene muchos textos vacíos o ruido.

---

## 2) Comando base de entrenamiento

El módulo de entrenamiento es `neurocampus.models.train_rbm` y expone opciones para el tipo de RBM, escalado de features y uso de probabilidades de texto.

> Ejecuta siempre desde la **raíz del repo** y define `PYTHONPATH` para el backend.

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.train_rbm   --type general   --data data/labeled/evaluaciones_2025_beto_textonly.parquet   --job-id auto   --seed 42   --epochs 100 --n-hidden 64   --cd-k 1 --epochs-rbm 1   --batch-size 128   --lr-rbm 5e-3 --lr-head 1e-2   --scale-mode minmax   --use-text-probs
```

**Esta es la configuración estable actual** (Día 7):
- `--type general`: RBM con cabeza de clasificación (softmax) para 3 clases.
- `--scale-mode minmax`: escalar `calif_1..10` a [0,1] (en datasets con rango 0–5 ayuda).
- `--use-text-probs`: añade `p_neg/p_neu/p_pos` como features 11–13.
- `--epochs 100`, `--cd-k 1`, `--epochs-rbm 1`: preentrenamiento RBM ligero en cada época + cabeza supervisada.
- Tasa de aprendizaje: `--lr-rbm 5e-3`, `--lr-head 1e-2`.
- `--seed 42`, `--batch-size 128` para reproducibilidad y rendimiento.

---

## 3) Otras variantes útiles

### 3.1 Solo calificaciones (sin texto)
```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.train_rbm   --type general   --data data/labeled/evaluaciones_2025_beto_textonly.parquet   --job-id auto   --seed 42   --epochs 80 --n-hidden 64   --cd-k 1 --epochs-rbm 1   --batch-size 128   --lr-rbm 5e-3 --lr-head 1e-2   --scale-mode minmax
```
> Quita `--use-text-probs` para entrenar con **10 features** (solo preguntas).

### 3.2 Tipo “restringida”
```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.train_rbm   --type restringida   --data data/labeled/evaluaciones_2025_beto_textonly.parquet   --job-id auto   --seed 42   --epochs 100 --n-hidden 64   --cd-k 1 --epochs-rbm 1   --batch-size 128   --lr-rbm 5e-3 --lr-head 1e-2   --scale-mode minmax   --use-text-probs
```
> Útil para comparar arquitecturas. El *general* ha mostrado desempeño más estable en este dataset.

### 3.3 Escalados alternos
- `--scale-mode standard` → z-score (media 0, var 1).  
- `--scale-mode scale_0_5` → mapea a [0,5] (solo si tu rúbrica lo exige; para RBM suele ser mejor **minmax**).

---

## 4) Qué se guarda en cada job

Cada corrida crea un directorio `artifacts/jobs/<YYYYMMDD_HHMMSS>` con:

```
job_meta.json         # hiperparámetros usados, semilla, dataset, tamaños
metrics.json          # f1_macro, accuracy, distribución de clases
vectorizer.json       # configuración de escalado/features
rbm.pt                # pesos de la RBM (binarios)
head.pt               # pesos de la cabeza clasificadora (binarios)
```

**Ejemplo de salida al finalizar:**
```
{'job_dir': 'artifacts\jobs\20251013_173312',
 'f1_macro': 0.28,
 'accuracy': 0.75,
 'classes': ['neg', 'neu', 'pos'],
 'n_val': 184,
 'n_labeled_used': 917,
 'n_features': 13,
 'type': 'general',
 'seed': 42}
```

> **Interpretación:** reportamos métricas en **validación**. En datasets desbalanceados, prioriza **macro-F1** frente a accuracy.

---

## 5) Promoción de “campeón” (modelo activo)

Una vez identifiques la mejor corrida, **promuévela** a `artifacts/champions/with_text/current`:

```bash
JOB="artifacts/jobs/<JOB_ID>"
DEST="artifacts/champions/with_text/current"

rm -rf "$DEST" && mkdir -p "$DEST"
cp -r "$JOB"/* "$DEST"/

# Descriptor (opcional)
cat > "$DEST/CHAMPION.json" <<'JSON'
{
  "family": "with_text",
  "selected_at": "(UTC)",
  "reason": "best macro-F1",
  "notes": "text+num, minmax, 100 epochs"
}
JSON
```

> El backend lee este **campeón** para las predicciones online. Puedes apuntar con `CHAMPION_WITH_TEXT=artifacts/champions/with_text/current` en `.env`.

---

## 6) Evaluación y análisis rápido

### 6.1 Cargar métricas del job
```bash
python - <<'PY'
import json, os
job = "artifacts/jobs/<JOB_ID>"
with open(os.path.join(job,"metrics.json"), "r", encoding="utf-8") as f:
    m = json.load(f)
print(m)
PY
```

### 6.2 Distribución de etiquetas (train/val) y matriz de confusión
Si el job guardó predicciones de validación (opcional), puedes inspeccionar la matriz:
```bash
python - <<'PY'
import json, os, pandas as pd
job = "artifacts/jobs/<JOB_ID>"
ytrue = pd.read_parquet(os.path.join(job, "y_val.parquet"))["y"]
ypred = pd.read_parquet(os.path.join(job, "y_val_pred.parquet"))["yhat"]
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytrue, ypred, labels=[0,1,2]))
print(classification_report(ytrue, ypred, target_names=["neg","neu","pos"]))
PY
```
> Si tu versión no guarda `y_val*`, revisa `metrics.json` y logs de entrenamiento para señales de *underfitting/overfitting*.

---

## 7) Recomendaciones prácticas

- **Desbalance**: usa la **regla costo-sensible** en inferencia (ya integrada en el backend) para favorecer *neg* cuando la evidencia es suficiente y evitar falsos positivos de *pos*.
- **Texto ruidoso**: filtra con `--min-tokens 1..2` en el teacher; eleva `threshold/margin/neu-min` para mayor precisión.
- **Ajuste de hparams**: comienza con el **preset estable** y prueba variaciones pequeñas (hidden 32/128, lr 1e-3–2e-2, cd-k 1–2).
- **Semilla**: fija `--seed` para corrida reproducible.
- **Val split**: si tu dataset cambia drásticamente, considerar *stratified split* explícito (por ahora se usa split interno del job).

---

## 8) Entrenamiento sin texto (baseline de control)

Para validar que el beneficio viene del texto, entrena un baseline **solo con calificaciones** y compara:

```bash
PYTHONPATH="$PWD/backend/src" python -m neurocampus.models.train_rbm   --type general   --data data/labeled/evaluaciones_2025_beto_textonly.parquet   --job-id auto   --seed 42   --epochs 80 --n-hidden 64   --cd-k 1 --epochs-rbm 1   --batch-size 128   --lr-rbm 5e-3 --lr-head 1e-2   --scale-mode minmax
```

> Espera menor macro-F1 frente al modelo **con** `--use-text-probs` si los comentarios añaden señal útil.

---

## 9) (Opcional) Búsqueda de hiperparámetros

Próximo paso del proyecto:
- **Random Search** 3-fold (15–25 trials) variando: `n_hidden`, `cd_k`, `epochs_rbm`, `lr_head`, `lr_rbm`, `scale_mode`, `use_text_pros`.
- Selección por **macro-F1** en validación media.
- Persistir un `search_report.json` y promover automáticamente el mejor job.

---

## 10) Errores comunes

- **FileNotFoundError** → corre desde la **raíz** del repo o ajusta rutas relativas.
- **CUDA/memoria** → por defecto usamos CPU; si habilitas GPU, baja `batch-size` o cd-k.
- **Métricas extrañas (accuracy alto vs F1 bajo)** → dataset desbalanceado; usa macro-F1 y revisa la **matriz de confusión**.
- **Peor desempeño con texto** → revisa `accept_rate` y `text_coverage` del teacher; si son muy bajos, el texto puede no estar aportando.

---

## 11) Checklist de cierre (Día 7)

- [x] Entrenamiento con preset estable ejecutado y **job_dir** identificado.
- [x] `metrics.json` con métricas de validación.
- [x] Modelo **promovido** a `artifacts/champions/with_text/current`.
- [x] Backend leyendo el campeón y sirviendo `/prediccion/online`.
- [x] Documentación actualizada (este archivo + README + Preprocesamiento).

