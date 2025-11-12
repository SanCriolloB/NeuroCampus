
# Día 11 · Documentación de Modelos RBM — NeuroCampus

**Fecha de generación:** 2025-11-05 02:33

Este documento describe la estructura actual de modelos RBM dentro del proyecto **NeuroCampus**, con sus roles, diferencias técnicas y relaciones. Incluye los cambios implementados en el **Día 11**, en el marco del bloque “Mejora de rendimiento de RBM (Días 10–15)”.

---

## 1️⃣ Contexto general

El sistema de modelos predictivos de NeuroCampus utiliza distintas variantes de **Máquinas de Boltzmann Restringidas (RBM)** para procesar y clasificar información evaluativa (p. ej. calificaciones, sentimientos, embeddings textuales).

Durante el **Día 11**, se reestructuró y documentó la arquitectura de modelos para:

1. Corregir la definición de la **RBM general**, que antes era una **Boltzmann Machine completa**, no restringida.
2. Crear una implementación **matemáticamente pura de RBM** (sin dependencias cruzadas).
3. Establecer un **núcleo común y coherente** para reutilizar en futuros modelos (RBM, DBN, etc.).

---

## 2️⃣ Estructura actual del módulo `neurocampus.models.strategies`

```
backend/src/neurocampus/models/strategies/
├── modelo_rbm_general.py         ← RBM bipartita con cabeza supervisada mínima
├── modelo_rbm_restringida.py     ← RBM existente (optimizada para entrenamiento supervisado)
└── rbm_pura.py                   ← Nueva RBM matemática pura (sin capa supervisada)
```

---

## 3️⃣ Descripción de cada modelo

### 🧩 `modelo_rbm_general.py` (RBM General Bipartita)

**Antes:** era una Boltzmann Machine completa (con conexiones visibles–visibles y ocultas–ocultas).  
**Ahora:** se convierte en una **RBM real** bipartita (`W`, `b_v`, `b_h`), reutilizando el núcleo `_RBM` y añadiendo una **cabeza supervisada mínima**.

**Características principales:**
- Entrenamiento no supervisado inicial con **Contrastive Divergence (CD-k)**.
- Fine-tuning supervisado con **cross-entropy** para clasificación (`neg`, `neu`, `pos`).
- Entrada flexible: detecta columnas `calif_*`, `p_*`, `x_text_*` automáticamente.
- Normalización integrada con escalado `minmax` o `0–5`.
- Persistencia en formato `.pt` + `vectorizer.json` + `meta.json`.

**Métodos clave:**
- `fit(X, y)`: entrena la RBM y la cabeza supervisada.
- `predict_proba(X)`: devuelve probabilidades de clase (3 clases).
- `predict(X)`: devuelve etiquetas textuales (`neg`, `neu`, `pos`).
- `save(dir)` / `load(dir)`: guardado y carga completa del modelo.

---

### 🧮 `modelo_rbm_restringida.py` (RBM Restringida Supervisada)

**Propósito:** mantener la versión supervisada ya estable, con ajustes internos para compatibilidad y validación cruzada.

**Características:**
- CD-k configurable (`cd_k`).
- Entrenamiento supervisado incorporado en la arquitectura.
- Compatibilidad total con el auditor (`audit_kfold.py`).
- Soporta inicialización dinámica de pesos según el dataset (`n_visible` ajustable).

**Uso:** se conserva como baseline de referencia para las futuras comparaciones.

---

### ⚙️ `rbm_pura.py` (RBM Matemática Pura)

**Nueva implementación (Día 11)**

RBM mínima, bipartita, implementada desde las ecuaciones de energía y probabilidades condicionales. No depende de otras clases del proyecto.

**Estructura interna:**
- `_RBMCore`: implementa la lógica matemática (amostrado de `h|v` y `v|h`, CD-k, gradientes).
- `RBM`: envoltorio de alto nivel con API estándar (`fit`, `predict_proba`, `predict`).
- Normalizador interno (`_MinMax`) para escalar datos a [0,1].

**Entrenamiento:**
- Entrena únicamente la parte no supervisada (sin `head`).
- Calcula reconstrucción (`recon_error`) y norma de gradiente por batch.
- Devuelve probabilidades heurísticas de 3 clases para mantener compatibilidad con el auditor.

**Métodos principales:**
- `fit(X)`: entrenamiento CD-k simple.
- `transform(X)`: devuelve activaciones ocultas.
- `predict_proba(X)`: pseudo-probabilidades de clase.
- `predict(X)`: etiquetas inferidas (`neg`, `neu`, `pos`).

---

## 4️⃣ Integración con el auditor `audit_kfold.py`

El auditor ahora resuelve automáticamente las tres variantes de modelo:

```python
if name == "rbm_general":
    from neurocampus.models.strategies.modelo_rbm_general import ModeloRBMGeneral
elif name == "rbm_restringido":
    from neurocampus.models.strategies.modelo_rbm_restringida import ModeloRBMRestringida
elif name == "rbm_pura":
    from neurocampus.models.strategies.rbm_pura import RBM as RBMPura
```

Cada modelo recibe sus parámetros desde `configs/rbm_audit.yaml`, donde el auditor:

- Normaliza alias (`hidden_units → n_hidden`, `lr → lr_rbm/lr_head`).
- Inyecta la dimensión de entrada (`visible_units` o `n_visible`).
- Repara automáticamente cualquier mismatch de shape.
- Evalúa métricas k-fold (Accuracy, F1, AUC, Precision, Recall, MCC).

---

## 5️⃣ Archivo de configuración YAML

Ubicación: `configs/rbm_audit.yaml` (en la raíz del repo)

Ejemplo actual:

```yaml
dataset:
  path: data/labeled/evaluaciones_2025_teacher.parquet

evaluation:
  n_splits: 5
  shuffle: true
  stratify: true
  random_seed: 42
  metrics: ["accuracy", "f1", "roc_auc", "precision", "recall", "mcc"]

models:
  - name: "RBM_general"
    params:
      hidden_units: 128
      lr: 0.01
      batch_size: 128
      epochs: 10
      cd_k: 1

  - name: "RBM_restringido"
    params:
      hidden_units: 128
      lr: 0.01
      batch_size: 128
      epochs: 10
      cd_k: 1

  - name: "rbm_pura"
    params:
      hidden_units: 128
      lr: 0.01
      batch_size: 128
      epochs: 10
      cd_k: 1

artifacts:
  root: artifacts/runs
```

---

## 6️⃣ Relación entre modelos

| Tipo de modelo | Supervisado | Arquitectura | Fuente de inicialización | Propósito |
|----------------|-------------|---------------|---------------------------|------------|
| RBM General | ✅ | Bipartita | Núcleo `_RBM` | Modelo estándar con cabeza supervisada mínima |
| RBM Restringida | ✅ | Bipartita (optimizada) | Código previo | Baseline validado de producción |
| RBM Pura | ❌ | Bipartita matemática | Implementación base Día 11 | Modelo teórico base para comparaciones y DBN futura |

---

## 7️⃣ Próximos pasos (Día 12–13)

1. **Construir una DBN (Deep Belief Network)** apilando capas RBM (`rbm_pura`) para comparar rendimiento.  
2. **Ampliar el auditor** para registrar tiempos de entrenamiento e inferencia.  
3. **Generar datasets simulados** (Mockaroo / SDV / CTGAN) para probar escalabilidad.  
4. **Refinar visualización de resultados** (gráficos comparativos por métrica).

---

## 8️⃣ Commit relacionado

```bash
git add backend/src/neurocampus/models/strategies/modelo_rbm_general.py         backend/src/neurocampus/models/strategies/rbm_pura.py         backend/src/neurocampus/models/audit_kfold.py         configs/rbm_audit.yaml         Makefile         docs/modelos/RBM_diseno.md
git commit -m "D11: Documentación de estructura y comparativa de modelos RBM (general, restringida, pura) + integración auditoría K-Fold"
```

---

**Autor:** Equipo NeuroCampus  
**Bloque:** Mejora de rendimiento de RBM (Días 10–15)
