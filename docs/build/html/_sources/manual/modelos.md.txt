# Pestaña «Modelos»

## Objetivo

La pestaña **Modelos** permite entrenar y comparar diferentes arquitecturas de
Máquinas de Boltzmann (BM, RBM, DBM y variantes “puras”) sobre los datasets
preprocesados, para seleccionar el modelo con mejor desempeño y usarlo en las
predicciones posteriores.

Está pensada para usuarios técnicos o analistas de datos que necesiten:

- Ejecutar entrenamientos controlados.
- Revisar métricas y curvas de aprendizaje.
- Entender cómo se comportan los distintos modelos sobre un mismo dataset.

---

## Organización de la pantalla

La interfaz se organiza en:

1. **Barra superior**
   - Selector de **dataset activo** (por periodo o identificador).
   - Botón principal **«Entrenar todos los modelos»**.
   - Opcionalmente, acceso a ajustes avanzados (épocas, tamaño de batch, etc.).

2. **Pestañas internas**
   - **Comparativa de modelos**  
     Vista general para comparar resultados de BM, RBM, DBM y variantes.
   - **Detalles del mejor modelo**  
     Análisis en profundidad del modelo con mejor métrica global.

La navegación entre pestañas no reinicia los resultados; se reusa el último
experimento lanzado para el dataset actual.

---

## Flujo de trabajo básico

1. **Seleccionar dataset**
   - En la parte superior, escoge el periodo o identificador de dataset sobre el
     que quieras entrenar.
   - Idealmente es un dataset que ya pasó por:
     - Preprocesamiento (limpieza, codificación).
     - Enriquecimiento con BETO (si el modelo usa características de texto).

2. **Configurar parámetros de entrenamiento (opcional)**
   - En el panel de configuración rápida puedes ajustar:
     - Número de épocas.
     - Tamaño de batch.
     - Tasas de aprendizaje.
   - Si no se ajustan, se usan valores por defecto recomendados.

3. **Entrenar los modelos**
   - Pulsa **«Entrenar BM/RBM/DBM/BM pura/RBM pura»** o el botón principal
     equivalente.
   - El backend lanzará los jobs correspondientes para cada modelo:
     - BM (Máquina de Boltzmann completa),
     - RBM (Restricted Boltzmann Machine),
     - DBM (Deep Boltzmann Machine),
     - variantes “puras” según la configuración interna.

4. **Revisar la comparativa de modelos**
   - Al finalizar el entrenamiento, se actualiza la pestaña **Comparativa de
     modelos** con una tabla y gráficos de resumen.

5. **Seleccionar el modelo campeón**
   - El sistema resalta automáticamente la fila del **mejor modelo** según la
     métrica principal configurada (por ejemplo, F1 o accuracy).
   - Desde ahí puedes ir a **Detalles del mejor modelo** para analizarlo con
     mayor profundidad.

---

## Comparativa de modelos

En esta pestaña se muestra:

### 1. Configuración rápida

Pequeña card donde se visualizan:

- Dataset actual.
- Hiperparámetros globales usados en el entrenamiento.
- Fecha y hora del último experimento.

Sirve como contexto al analizar las métricas.

### 2. Tabla comparativa de métricas

Tabla principal con:

- **Filas**: cada modelo evaluado (BM, RBM, DBM, BM pura, RBM pura, etc.).
- **Columnas típicas**:
  - Accuracy (exactitud),
  - Precision,
  - Recall,
  - F1-score,
  - Tiempo total de entrenamiento,
  - Número de parámetros (si está disponible),
  - Identificador del modelo/artefacto guardado.

La **fila del mejor modelo** se destaca visualmente (por ejemplo, con un fondo
ligeramente diferente o un icono).

### 3. Gráficos comparativos

Según la configuración, se suelen mostrar:

- **Gráfico de barras comparativas**:
  - Eje X: modelos.
  - Eje Y: métrica seleccionada (p. ej., F1).
  - Permite ver rápidamente qué arquitectura ofrece mejor desempeño.

- **Gráfico de barras apiladas o agrupadas** (opcional):
  - Para visualizar varias métricas a la vez (por ejemplo, precision y recall).

---

## Detalles del mejor modelo

Al seleccionar la pestaña **Detalles del mejor modelo**, se muestran:

### 1. Resumen del modelo

Card con:

- Nombre del modelo (por ejemplo, `RBM_manual` o `DBM`).
- Métricas clave (Accuracy, F1, etc.).
- Hiperparámetros (número de neuronas ocultas, capas, tasa de aprendizaje,
  épocas, etc.).
- Ruta del artefacto guardado (por ejemplo, en `artifacts/`).

### 2. Curvas de entrenamiento

Gráficas que permiten entender el comportamiento del modelo durante el
entrenamiento:

- **Curva de pérdida (loss vs epoch)**:
  - Eje X: epochs.
  - Eje Y: valor de la función de pérdida.
  - Permite ver si el entrenamiento ha convergido o si hay sobreajuste.

- **Curva de accuracy (accuracy vs epoch)**:
  - Eje X: epochs.
  - Eje Y: exactitud en entrenamiento/validación (si aplica).

### 3. Tiempo por época

Gráfico adicional:

- **Gráfico de barras o de línea**:
  - Eje X: epochs.
  - Eje Y: tiempo de entrenamiento por epoch.
  - Útil para comparar modelos pesados (como DBM) frente a otros más ligeros.

### 4. Matriz de confusión

Visualización esencial para entender qué está clasificando bien o mal el
modelo:

- **Matriz de confusión** (idealmente en forma de mapa de calor):
  - Ejes: clases verdaderas vs clases predichas (por ejemplo, alto/bajo
    rendimiento).
  - Cada celda indica el número de casos.
- Métricas derivadas:
  - TP (True Positive),
  - TN (True Negative),
  - FP (False Positive),
  - FN (False Negative).

---

## Errores frecuentes y recomendaciones

- **No aparece ningún modelo en la tabla**  
  Verifica que:
  1. Hay un dataset seleccionado.
  2. Se ha ejecutado al menos un entrenamiento para ese dataset.

- **Las métricas son muy bajas para todos los modelos**  
  Puede indicar:
  - Problemas con el preprocesamiento del dataset.
  - Hiperparámetros muy poco adecuados (épocas insuficientes, learning rate
    muy alto, etc.).

- **El entrenamiento tarda demasiado**  
  - Considera reducir el tamaño de las capas ocultas o el número de epochs.
  - Revisa si es necesario entrenar todos los modelos o solo un subconjunto.

---

## Relación con otras secciones

- La pestaña **Modelos** se alimenta de:
  - Los datasets preparados en la pestaña **Datos**.
  - La teoría de BM/RBM/DBM descrita en la sección de **Teoría de modelos**.
- El modelo campeón seleccionado se usa posteriormente en:
  - **Predicciones** (para inferencias individuales o por lote).
  - **Dashboard** (para métricas agregadas y proyecciones).
