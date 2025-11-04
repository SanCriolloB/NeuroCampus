# Día 10 · Paso 6.4 — Informe de Auditoría RBM (K-Fold)

**Fecha de generación:** 2025-11-04 18:03

Este documento consolida los resultados y cambios técnicos realizados en los pasos 6.1–6.3, y deja la documentación final del **Paso 6.4**.


## 1) Contexto y objetivo

- **Objetivo**: Auditar el desempeño base de las variantes **RBM_general** y **RBM_restringido** sobre el dataset configurado, con validación **k-fold** estratificada, y dejar la línea base para las mejoras siguientes del Día 10.
- **Evidencia generada**: `metrics.json` bajo el run indicado por el auditor.

## 2) Configuración usada
- **Dataset**: `data/labeled/evaluaciones_2025_teacher.parquet`
- **Target**: `(auto-detect)`
- **Evaluación**:
  - `n_splits`: **5**
  - `shuffle`: **True**
  - `stratify`: **True**
  - `random_seed`: **42**
  - `metrics`: accuracy, f1, roc_auc, precision, recall, mcc

## 3) Cambios técnicos clave (6.1–6.3)
- **Auditor (`audit_kfold.py`)**:
  - Encoding automático del `target` (strings → enteros) y persistencia de `label_mapping`.
  - Métricas robustas para binario/multiclase (AUC macro OVR si aplica) y `zero_division=0`.
  - Manejo flexible de `predict`/`predict_proba` (1D/2D).
  - Inyección/ajuste de dimensión de entrada a partir de `X.shape[1]`.
- **Strategy `modelo_rbm_restringida.py`**:
  - Diferir la creación de la RBM hasta `fit(...)` para usar `n_visible = X.shape[1]` real.
  - Safeguards en `_RBM.sample_h(...)` y `_RBM.cd_step(...)` para re-sincronizar `W`/`b_v` si cambian las features.
- **Tests (6.3)**: `tests/unit/test_rbm_audit_schema.py` pasando (**1 passed**).

## 4) Resultados

### Modelo: **RBM_general**

**Parámetros clave**
- **hidden_units**: `128`
- **lr**: `0.01`
- **batch_size**: `128`
- **epochs**: `10`
- **cd_k**: `1`
- **target**: `sentiment_label_teacher`
- **label_mapping**: `neg`→0, `neu`→1, `pos`→2

**Resumen (k-fold)**
| métrica | media | std |
|--- | --- | ---|
| accuracy | 0.7708 | 0.0006 |
| f1 | 0.2902 | 0.0001 |
| mcc | 0.0000 | 0.0000 |
| precision | 0.2569 | 0.0002 |
| recall | 0.3333 | 0.0000 |
| roc_auc | 0.4116 | 0.0262 |


**Resultados por fold**
| fold | accuracy | f1 | mcc | precision | recall | roc_auc |
|--- | --- | --- | --- | --- | --- | ---|
| 1 | 0.7713 | 0.2903 | 0.0000 | 0.2571 | 0.3333 | 0.3769 |
| 2 | 0.7713 | 0.2903 | 0.0000 | 0.2571 | 0.3333 | 0.4261 |
| 3 | 0.7713 | 0.2903 | 0.0000 | 0.2571 | 0.3333 | 0.4255 |
| 4 | 0.7701 | 0.2900 | 0.0000 | 0.2567 | 0.3333 | 0.4448 |
| 5 | 0.7701 | 0.2900 | 0.0000 | 0.2567 | 0.3333 | 0.3847 |


### Modelo: **RBM_restringido**

**Parámetros clave**
- **hidden_units**: `128`
- **lr**: `0.01`
- **batch_size**: `128`
- **epochs**: `10`
- **cd_k**: `1`
- **target**: `sentiment_label_teacher`
- **label_mapping**: `neg`→0, `neu`→1, `pos`→2

**Resumen (k-fold)**
| métrica | media | std |
|--- | --- | ---|
| accuracy | 0.1588 | 0.0019 |
| f1 | 0.0914 | 0.0010 |
| mcc | 0.0000 | 0.0000 |
| precision | 0.0529 | 0.0006 |
| recall | 0.3333 | 0.0000 |
| roc_auc | 0.3685 | 0.0402 |


**Resultados por fold**
| fold | accuracy | f1 | mcc | precision | recall | roc_auc |
|--- | --- | --- | --- | --- | --- | ---|
| 1 | 0.1596 | 0.0917 | 0.0000 | 0.0532 | 0.3333 | 0.3870 |
| 2 | 0.1596 | 0.0917 | 0.0000 | 0.0532 | 0.3333 | 0.3615 |
| 3 | 0.1596 | 0.0917 | 0.0000 | 0.0532 | 0.3333 | 0.3596 |
| 4 | 0.1551 | 0.0895 | 0.0000 | 0.0517 | 0.3333 | 0.4291 |
| 5 | 0.1604 | 0.0922 | 0.0000 | 0.0535 | 0.3333 | 0.3056 |



## 5) Conclusiones y siguientes acciones

- **RBM_general** supera a **RBM_restringido** en la línea base.
- Los valores de **AUC** relativamente bajos sugieren margen de mejora en separabilidad: 
  normalización, selección/creación de features y ajuste de hiperparámetros serán el foco del resto del Día 10.

**Acciones inmediatas (Día 10 - siguientes subpasos):**
1. Afinar **preprocesamiento/normalización** específico por feature y revisar outliers.
2. Explorar **conjunto de features**: incluir/excluir `p_*` y/o `x_text_*` según disponibilidad y validación.
3. Tuning ligero de hiperparámetros en `n_hidden`, `cd_k`, tasas de aprendizaje y balanceo de clases.