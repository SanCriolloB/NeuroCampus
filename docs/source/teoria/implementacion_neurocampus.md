# Implementación de modelos de Boltzmann en NeuroCampus

## Objetivo

Esta sección conecta los conceptos teóricos de Máquinas de Boltzmann (BM),
Restricted Boltzmann Machines (RBM) y Deep Boltzmann Machines (DBM) con los
módulos concretos del proyecto NeuroCampus.

El propósito es que cualquier desarrollador pueda:

- Ubicar rápidamente dónde se implementa cada concepto teórico.
- Entender cómo se orquesta el entrenamiento, evaluación y el uso de los
  modelos en la aplicación (pestañas Modelos, Predicciones y Dashboard).

---

## Mapeo conceptual → módulos de código

### 1. BM (Máquinas de Boltzmann)

**Teoría**

- Modelo de energía con conexiones generales entre unidades visibles y
  ocultas.
- Probabilidad definida a partir de la energía y la función de partición.
- Entrenamiento mediante aproximaciones al gradiente (muestreo Monte Carlo,
  Gibbs, etc.).

**Implementación en NeuroCampus**

- Módulo principal:
  - `neurocampus.models.bm_manual`
- Responsabilidades:
  - Definir la estructura de la BM (número de visibles/ocultas, pesos, sesgos).
  - Implementar la función de energía y las probabilidades condicionales
    necesarias.
  - Proveer métodos de entrenamiento y actualización de parámetros.

La BM manual se integra en el ecosistema de modelos a través del registro de
modelos y las estrategias de entrenamiento, permitiendo compararla con RBM y
DBM desde la pestaña **Modelos**.

---

### 2. RBM (Restricted Boltzmann Machines)

**Teoría**

- Estructura bipartita: visibles–ocultas sin conexiones internas.
- Condicionales factorizadas que facilitan el muestreo:
  - \(P(h \mid v)\) y \(P(v \mid h)\).
- Entrenamiento mediante Contrastive Divergence (CD-k) y variantes.

**Implementación en NeuroCampus**

- Módulo principal:
  - `neurocampus.models.rbm_manual`
- Estrategias y variantes:
  - `neurocampus.models.strategies.rbm_manual_strategy`
  - `neurocampus.models.strategies.rbm_pura`
  - `neurocampus.models.strategies.modelo_rbm_general`
  - `neurocampus.models.strategies.modelo_rbm_restringida`

- Jobs de entrenamiento:
  - `neurocampus.app.jobs.cmd_train_rbm_manual`
  - Otros comandos auxiliares para auditoría y búsqueda de hiperparámetros.

- Auditoría y búsqueda de hiperparámetros:
  - `neurocampus.models.audit_kfold`
  - `neurocampus.models.hparam_search`

Estas piezas trabajan juntas para:

- Entrenar RBM bajo distintos escenarios y configuraciones.
- Evaluar su desempeño usando validación cruzada y métricas estándar.
- Seleccionar configuraciones adecuadas que luego se integran como modelos
  candidatos en la pestaña **Modelos**.

---

### 3. DBM (Deep Boltzmann Machines)

**Teoría**

- Extensión de RBM/BM a múltiples capas ocultas.
- Interacciones entre capas adyacentes.
- Entrenamiento combinado de todas las capas, a menudo precedido de
  pre-entrenamiento por capas (RBM apiladas).

**Implementación en NeuroCampus**

- Módulo principal:
  - `neurocampus.models.dbm_manual`
- Estrategias:
  - `neurocampus.models.strategies.dbm_manual_strategy`
- Jobs relacionados:
  - `neurocampus.app.jobs.cmd_train_dbm_manual` (nombre aproximado, según
    nomenclatura de jobs de entrenamiento).

En la práctica, la DBM se construye a partir de bloques RBM y comparte
infraestructura con otros modelos (registro, auditoría, etc.). Esto permite:

- Reutilizar código de RBM manual para configurar las capas.
- Integrar la DBM en los mismos pipelines de entrenamiento y evaluación.

---

## Estrategias, facades y registro de modelos

Para que los modelos de Boltzmann se integren de forma coherente con la API y
el frontend, NeuroCampus define varias capas adicionales.

### 1. Estrategias de entrenamiento (`strategies/`)

Ubicación: `neurocampus/models/strategies/`.

Responsabilidades:

- Encapsular la lógica de entrenamiento de cada modelo:
  - configuración de hiperparámetros,
  - ciclos de epochs,
  - cálculo de métricas (loss, accuracy, etc.).
- Permitir que la capa de aplicación (routers, jobs) invoque entrenamientos
  de forma homogénea.

Ejemplos:

- `bm_manual_strategy.py`
- `rbm_manual_strategy.py`
- `dbm_manual_strategy.py`
- Variantes específicas para distintos experimentos de RBM.

### 2. Registro de modelos (`registry.py`)

Ubicación: `neurocampus/models/registry.py`.

Responsabilidades:

- Mantener un registro de los modelos disponibles (BM, RBM, DBM, etc.).
- Resolver qué modelo se debe usar para un determinado contexto (por ejemplo,
  tipo de dataset o configuración de experimento).
- Facilitar la carga y el guardado de artefactos (pesos, configuraciones,
  métricas).

### 3. Facade de modelos (`facades/modelos_facade.py`)

Ubicación: `neurocampus/models/facades/modelos_facade.py`.

Responsabilidades:

- Proveer una interfaz de alto nivel para:
  - lanzar entrenamientos,
  - recuperar métricas,
  - seleccionar el modelo campeón,
  - preparar modelos para predicción.
- Ocultar detalles internos de:
  - cómo se instancian las clases de modelo,
  - cómo se gestionan archivos de artefactos,
  - cómo se integran resultados con otras capas del sistema.

Esta fachada es la que idealmente usan los routers de `modelos` y `prediccion`
para interactuar con los diferentes modelos.

---

## Integración con la API y la interfaz de usuario

### 1. Pestaña Modelos

- Usa endpoints de `neurocampus/app/routers/modelos.py` que delegan en:
  - `modelos_facade`,
  - estrategias y registro de modelos.
- Permite lanzar entrenamientos de BM, RBM, DBM y variantes, y obtener:
  - métricas,
  - curvas de entrenamiento,
  - matrices de confusión.

### 2. Pestaña Predicciones

- Usa endpoints de `neurocampus/app/routers/prediccion.py` para aplicar el
  **modelo campeón** a nuevos datos.
- Internamente:
  - recupera el modelo seleccionado a través del registro/facade,
  - aplica el mismo preprocesamiento que se usó en entrenamiento,
  - calcula probabilidades y clases finales.

### 3. Pestaña Dashboard

- Se nutre de métricas y predicciones agregadas generadas por los modelos
  de Boltzmann (y otros modelos que se integren en el futuro).
- Permite observar el impacto de las decisiones de modelado:
  - mejor o peor calibración del modelo,
  - distribución de riesgo,
  - evolución del desempeño modelado en el tiempo.

---

## Resumen

En NeuroCampus, la teoría de BM, RBM y DBM se concreta en:

- Módulos `bm_manual`, `rbm_manual`, `dbm_manual` para la implementación base.
- Estrategias de entrenamiento en `models/strategies/`.
- Auditorías y búsqueda de hiperparámetros (`audit_kfold`, `hparam_search`).
- Registro y facades (`registry`, `modelos_facade`).
- Integración con la API a través de routers de `modelos`, `prediccion` y
  `jobs`.

Esta organización permite evolucionar la familia de modelos de Boltzmann sin
romper la interfaz hacia el frontend ni los flujos de la aplicación.
