# Pipeline de modelos (BM, RBM, DBM)

El pipeline de modelos de NeuroCampus describe el camino completo desde que un
dataset es cargado hasta que se entrena un modelo campeón y se usa para
predicciones y dashboard.

---

## Visión general del pipeline

1. **Ingesta y validación de datos** (pestaña Datos / router `datos`).
2. **Preprocesamiento y enriquecimiento** (incluyendo BETO).
3. **Entrenamiento de modelos BM/RBM/DBM**.
4. **Evaluación y selección del modelo campeón**.
5. **Predicción y scoring para docentes/asignaturas**.
6. **Exposición de resultados en Predicciones y Dashboard**.

---

## 1. Ingesta y validación

- Iniciada desde la pestaña **Datos**:
  - Botones:
    - “Validar sin guardar”
    - “Cargar y procesar”
- Endpoints:
  - `POST /datos/validar`:
    - Recibe el archivo (CSV/XLSX/Parquet).
    - Usa `neurocampus.data.validation_wrapper.run_validations` para validar
      el contenido.
  - `POST /datos/upload`:
    - Persiste el dataset preprocesado en la estructura de `data/`.
    - Actualiza el resumen del dataset (`datos_dashboard.build_dataset_resumen`).

El resultado de esta etapa es un dataset consistente y listo para usar en
modelos.

---

## 2. Preprocesamiento y enriquecimiento (incluyendo BETO)

Alcance típico:

- Limpieza de columnas.
- Codificación de variables categóricas.
- Normalización o escalado (según el modelo).
- Procesamiento del texto de comentarios mediante BETO.

### BETO (análisis de sentimientos)

- Job principal: `neurocampus/app/jobs/cmd_preprocesar_beto.py`.
- Responsabilidades:
  - Leer el dataset etiquetado con comentarios.
  - Pasar los textos por el modelo BETO (offline, CPU/GPU) para:
    - obtener scores de polaridad (pos/neu/neg),
    - generar labels o características adicionales.
  - Guardar un dataset enriquecido (ej. `*_beto.parquet`).

- Integración con la API:
  - Router `jobs.py` expone endpoints para lanzar y consultar el job de BETO.
  - Router `datos.py` expone:
    - `/datos/sentimientos` con agregados para:
      - distribución global,
      - distribución por docente,
      - métricas de cobertura de texto.

La pestaña **Datos** utiliza estos resultados para renderizar las gráficas de
sentimientos.

---

## 3. Entrenamiento de modelos BM/RBM/DBM

Una vez el dataset está preprocesado, el entrenamiento se puede lanzar desde:

- Pestaña **Modelos** (interacción UI).
- CLI / Makefile (para pipelines automatizados).

### 3.1 Modelos manuales

Ubicación: `neurocampus/models/`.

- `bm_manual.py`:
  - Implementación de Máquinas de Boltzmann completas.
- `rbm_manual.py`:
  - Implementación de RBM con codificación manual (sin dependencias externas
    de redes neuronales).
- `dbm_manual.py`:
  - Implementación de DBM manual, construida a partir de la lógica de RBM.

### 3.2 Estrategias de entrenamiento

Ubicación: `neurocampus/models/strategies/`.

- `bm_manual_strategy.py`
- `rbm_manual_strategy.py`
- `dbm_manual_strategy.py`
- Otros archivos relacionados (por ejemplo, `modelo_rbm_general`,
  `modelo_rbm_restringida`, `rbm_pura`, etc.).

Estas estrategias definen:

- Cómo se inicializa el modelo.
- Cómo se ejecuta el ciclo de entrenamiento:
  - epochs,
  - tamaño de batch,
  - tasa de aprendizaje.
- Qué métricas se calculan (loss, accuracy, etc.).
- Cómo se guarda el artefacto del modelo.

### 3.3 Orquestación por jobs

Ubicación: `neurocampus/app/jobs/`.

Comandos relevantes:

- `cmd_train_rbm_manual.py`:
  - Entrena uno o varios modelos RBM manuales sobre un dataset preprocesado.
- `cmd_train_dbm_manual.py`:
  - Entrena modelos DBM manuales.
- `cmd_eval_confusion.py`:
  - Genera matrices de confusión y métricas de evaluación.
- `cmd_autoretrain.py`:
  - Puede lanzar reentrenamientos periódicos según nuevas evaluaciones.

Estos comandos se invocan desde:

- CLI (por ejemplo, vía Makefile).
- Endpoints expuestos en el router `modelos` o `jobs`.

---

## 4. Evaluación y selección del modelo campeón

La evaluación se basa en:

- Métricas calculadas en:
  - `models/audit_kfold.py` (validación cruzada),
  - `models/hparam_search.py` (búsqueda de hiperparámetros).
- Resultados almacenados:
  - Rutas de artefactos,
  - ficheros de métricas (JSON/CSV),
  - logs de entrenamiento.

La **lógica de selección del modelo campeón** puede incluir:

- Regla básica:
  - modelo con mejor F1 o accuracy en el conjunto de validación.
- Criterios adicionales:
  - estabilidad entre folds (en `audit_kfold`),
  - tiempo de entrenamiento (preferir modelos más ligeros si el rendimiento es similar).

El modelo seleccionado se registra a través de:

- `models/registry.py`:
  - Permite resolver un “modelo activo” por dataset o contexto.

---

## 5. Predicción y scoring

La inferencia se realiza:

- Desde el router `prediccion.py`:
  - Para casos individuales (pestaña **Predicciones**, ingreso manual).
  - Para predicciones por lote (fichero de entrada con varios casos).
- Utilizando:
  - El modelo campeón cargado desde los artefactos,
  - el mismo pipeline de preprocesamiento aplicado al dataset de entrenamiento.

La salida típica:

- Probabilidad de alto/bajo rendimiento.
- Intervalos de confianza (si están implementados).
- Variables derivadas (por ejemplo, scores de sentimiento agregados).

---

## 6. Integración con Predicciones y Dashboard

- La pestaña **Predicciones**:
  - Consume los endpoints de `prediccion.py` y muestra:
    - probabilidad,
    - radar de indicadores,
    - barras comparativas,
    - proyecciones temporales para el caso o lote analizado.

- La pestaña **Dashboard**:
  - Consume datos agregados generados a partir de:
    - predicciones,
    - métricas del modelo,
    - histórico de evaluaciones reales.
  - Muestra:
    - KPIs globales,
    - históricos por docente/asignatura,
    - ranking de docentes,
    - distribuciones de riesgo,
    - comparaciones real vs predicho.

---

Este pipeline de modelos asegura que:

- Los datos pasan por un proceso controlado de validación y preprocesamiento.
- El entrenamiento de BM/RBM/DBM se realiza de forma reproducible.
- La selección del modelo campeón está basada en métricas cuantitativas.
- Las predicciones se integran de forma coherente con las vistas de usuario
  (Predicciones y Dashboard) para la toma de decisiones.
