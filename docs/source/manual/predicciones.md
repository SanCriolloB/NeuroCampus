# Pestaña «Predicciones»

## Objetivo

La pestaña **Predicciones** permite aplicar el modelo seleccionado (modelo
campeón) para estimar el rendimiento esperado de docentes/asignaturas a partir
de nuevas evaluaciones.

Ofrece dos modos principales:

1. **Ingreso manual**: registro de un caso a la vez vía formulario.
2. **Predicción por lote**: carga de múltiples casos desde un archivo.

---

## Organización de la pantalla

La pantalla se organiza con:

- **Layout maestro**:
  - Columna izquierda: formulario de entrada.
  - Columna derecha: resultado y visualizaciones de la predicción.

- **Tabs internos**:
  - `Ingreso manual`.
  - `Predicción por lote`.

Ambas pestañas comparten la misma zona de resultados (tarjetas y gráficos), que
se actualiza cada vez que se genera una nueva predicción.

---

## Ingreso manual

### 1. Formulario (columna izquierda)

Campos típicos:

- **Nombre del docente** (texto).
- **Asignatura** (texto o desplegable).
- **Calificaciones para las 10 preguntas**:
  - Usualmente 10 sliders o selectores con escala 1–5.
  - Cada pregunta representa un aspecto de la evaluación (claridad, dominio de
    contenido, puntualidad, etc.).
- **Comentario** (textarea opcional):
  - Texto libre que será procesado por el pipeline de PLN si el modelo lo
    utiliza.

### 2. Generar predicción

- Botón principal: **«Generar predicción»**.
- Al pulsarlo:
  - El frontend prepara un payload con los campos del formulario.
  - El backend aplica el modelo campeón configurado.
  - Se obtiene una probabilidad de alto/bajo rendimiento y otras métricas
    asociadas.

---

## Resultados de la predicción (salida inmediata)

En la columna derecha se muestran una o varias tarjetas y gráficos:

### 1. Tarjeta de resultado principal

Incluye:

- **Probabilidad de alto/bajo rendimiento**  
  Ejemplo:
  - Alto: 78% (IC 95% [70–85]),
  - Bajo: 22% (IC 95% [15–30]).

- **Clasificación final**:
  - “Predicción: Alto rendimiento esperado”
  - o “Predicción: Bajo rendimiento esperado”.

- **Mensaje explicativo**:
  - Texto corto que ayuda a interpretar el resultado:
    - “El modelo sugiere que este docente tiene una alta probabilidad de
      mantener un rendimiento positivo, especialmente por las calificaciones
      en las preguntas X e Y.”

### 2. Gráfico tipo radar (indicadores)

- **Gráfico de radar**:
  - Cada eje representa una pregunta o indicador (P1, P2, …, P10).
  - El área representa el perfil del docente para ese caso concreto.
- Ayuda a visualizar fortalezas y debilidades relativas.

### 3. Barras comparativas

- **Gráfico de barras**:
  - Comparación del docente/curso actual vs:
    - promedio histórico del mismo docente,
    - promedio de la asignatura,
    - promedio global.
- Eje X: indicadores o dimensiones.
- Eje Y: puntuaciones normalizadas.

### 4. Proyección temporal (si hay histórico)

- **Gráfico de línea o barras en el tiempo**:
  - Eje X: semestres o periodos.
  - Eje Y: rendimiento predicho o indicador agregado.
- Permite ver si el rendimiento está mejorando, estable o empeorando.

---

## Predicción por lote

### 1. Ingesta de archivo

En la pestaña **Predicción por lote**:

- Se muestra un componente de subida de archivo similar al de la pestaña
  **Datos**.
- Formatos aceptados:
  - `.csv`, `.xlsx`, `.xls`, etc. (según se defina en el backend).
- El archivo debe seguir una **plantilla de columnas** compatible con el
  modelo:
  - nombre docente,
  - asignatura,
  - respuestas a las 10 preguntas,
  - comentario (opcional),
  - otros campos que sean necesarios.

### 2. Ejecución de predicciones

- Botón: **«Generar predicciones (lote)»**.
- El backend procesa cada fila, aplicando el mismo modelo campeón.
- Se devuelve:
  - Una tabla de resultados (probabilidad de alto/bajo rendimiento por fila).
  - Agregados globales (por ejemplo, porcentaje de casos en riesgo alto).

### 3. Visualizaciones agregadas

Al trabajar con un lote, es posible mostrar:

- **Distribución de riesgo**:
  - Gráfico de barras o pastel con categorías:
    - Bajo riesgo,
    - Riesgo medio,
    - Alto riesgo.
- **Ranking de docentes/asignaturas**:
  - Ordenados por probabilidad de bajo rendimiento.
- **Mapa de calor por indicador**:
  - Eje X: indicadores/preguntas.
  - Eje Y: docentes o asignaturas.
  - Celdas coloreadas según el nivel de riesgo.

---

## Recomendaciones de uso

- Antes de utilizar la pestaña **Predicciones**, asegúrate de que:
  - Hay un **modelo campeón** seleccionado y entrenado en la pestaña **Modelos**.
  - Los datos de entrada siguen el mismo esquema de preprocesamiento que el
    dataset de entrenamiento.

- Para casos límite (probabilidades cercanas al 50/50):
  - Complementar la decisión con información contextual adicional.
  - Utilizar la pestaña **Dashboard** para revisar tendencias históricas.

---

## Errores frecuentes

- **El sistema indica que no hay modelo disponible**  
  - Revisa la pestaña **Modelos** y entrena al menos un modelo para el
    dataset de interés.
- **Formato incorrecto en predicción por lote**  
  - Comprueba que las columnas coinciden con la plantilla usada en el
    entrenamiento.
  - Asegúrate de que las escalas de calificación son las mismas (por ejemplo,
    1–5).

---

## Relación con otras pestañas

- Se apoya en:
  - **Modelos**: elección y entrenamiento del modelo campeón.
  - **Datos**: estructura y preprocesamiento de los datasets.
- Alimenta:
  - **Dashboard**: KPIs y visualizaciones agregadas a nivel histórico,
    docente, asignatura y programa.
