# Pestaña «Dashboard»

## Objetivo

La pestaña **Dashboard** muestra una vista global del desempeño y riesgo
académico, basada en:

- Datasets históricos de evaluaciones.
- Predicciones generadas por el modelo campeón.
- Indicadores y KPIs clave para la toma de decisiones.

Permite a coordinadores, directores de programa y analistas:

- Identificar docentes/asignaturas con mayor riesgo de bajo rendimiento.
- Monitorizar la evolución temporal del desempeño.
- Comparar real vs predicho y detectar desviaciones.

---

## Organización de la pantalla

La pantalla se organiza en:

1. **Zona de filtros (arriba)**
   - Filtros típicos:
     - Semestre / periodo.
     - Facultad / programa.
     - Docente.
     - Asignatura.
   - Botón **«Aplicar filtros»** para actualizar las visualizaciones.

2. **Tarjetas de KPIs (debajo de filtros)**
   - Pequeñas cards con valores clave:
     - Predicciones totales.
     - Exactitud del modelo (en validación / test).
     - Último dataset registrado.
     - Número de evaluaciones registradas en el periodo.
     - Porcentaje de casos en riesgo alto.

3. **Zona de gráficos (en grid)**
   - Gráficos organizados en filas, máximo 2 por fila para mantener claridad:
     - Históricos por docente/asignatura.
     - Ranking de docentes.
     - Distribución de riesgo.
     - Comparación desempeño real vs predicho.
     - Promedio histórico vs semestres específicos.

---

## Filtros globales

### 1. Selección de ámbito

- **Semestre / periodo**:
  - Selecciona uno o varios semestres para analizar (por ejemplo, `2023-2`,
    `2024-1`).
- **Programa / facultad**:
  - Limita el análisis a un subconjunto de la institución (por ejemplo,
    “Ingeniería”, “Ciencias Económicas”).
- **Docente / asignatura**:
  - Permite enfocarse en un docente o asignatura concreta.

### 2. Aplicar y limpiar filtros

- **Aplicar filtros**:
  - Refresca todos los KPIs y gráficos con el contexto actual.
- **Limpiar filtros** (opcional):
  - Vuelve a la vista global de la institución.

---

## Tarjetas de KPIs

Ejemplos de KPIs que pueden aparecer:

- **Predicciones totales**:
  - Número total de predicciones generadas en el periodo filtrado.
- **Exactitud del modelo**:
  - Métrica de desempeño del modelo en el conjunto de evaluación.
- **Último dataset registrado**:
  - Identificador y fecha del último dataset cargado.
- **Número de evaluaciones registradas**:
  - Cantidad de encuestas o registros en el rango temporal seleccionado.
- **Porcentaje de riesgo alto**:
  - Proporción de casos clasificados como alto riesgo de bajo rendimiento.

Estas tarjetas se actualizan al modificar los filtros y sirven como resumen
rápido antes de entrar en el detalle de los gráficos.

---

## Gráficos principales

### 1. Histórico por docente y por asignatura

- **Gráfico de líneas o barras**:
  - Eje X: semestres o periodos.
  - Eje Y: indicador de desempeño (real o predicho, según configuración).
- Permite analizar:
  - Tendencias de cada docente o asignatura.
  - Mejoras o deterioros a lo largo del tiempo.
- En modo multi-serie, se pueden comparar varios docentes o asignaturas
  simultáneamente (con posibilidad de activar/desactivar series en la leyenda).

### 2. Ranking de docentes por desempeño proyectado

- **Gráfico de barras horizontales**:
  - Eje Y: docentes (ordenados de mejor a peor o viceversa).
  - Eje X: indicador de desempeño (predicho o combinado).
- Usado para:
  - Identificar rápidamente a los mejores docentes según las evaluaciones.
  - Localizar casos con desempeño significativamente por debajo de la media.

### 3. Riesgo en caso de bajo rendimiento

- **Gráfico de barras apiladas**:
  - Eje X: asignaturas o programas.
  - Eje Y: número de evaluaciones/casos.
  - Segmentos apilados:
    - Bajo riesgo,
    - Riesgo medio,
    - Alto riesgo.
- Permite observar “dónde se acumula el problema”:
  - Programas con un gran volumen de casos en riesgo alto.
  - Asignaturas con concentración de evaluaciones negativas.

---

## Comparación desempeño real vs predicho

Cuando se dispone de datos reales y predicciones para los mismos casos:

### 1. Gráfico de dispersión (scatter plot)

- Eje X: desempeño real.
- Eje Y: desempeño predicho.
- Puntos cercanos a la diagonal `y = x` indican buena calibración del modelo.

Complementos:

- Línea diagonal (ideal).
- Colores por rango de riesgo o por programa.

### 2. Gráfico de barras agrupadas

- Eje X: docentes o asignaturas.
- Eje Y: indicador de desempeño.
- Dos barras por grupo:
  - Real.
  - Predicho.
- Útil para identificar sistemáticas de sobreestimación o subestimación.

---

## Promedio histórico vs semestres específicos

### 1. Gráfico de líneas o barras

Se comparan dos series:

- **Promedio histórico general**:
  - Media de desempeño en un rango amplio de semestres.
- **Promedio del semestre seleccionado**:
  - Media de desempeño en el semestre filtrado.

Ejes típicos:

- Eje X: semestres.
- Eje Y: promedio de desempeño global o por unidad académica.

Esto permite contestar preguntas como:

- “¿El último semestre ha mejorado o empeorado respecto a la media histórica?”
- “¿Está este programa por encima o por debajo del promedio institucional?”

---

## Recomendaciones de uso

- Usa los filtros para **acotar el contexto** antes de interpretar gráficos:
  - No es lo mismo analizar un docente aislado que un programa completo.
- Combina las visualizaciones:
  - Por ejemplo, revisa el **ranking de docentes** y luego observa el
    **histórico de los casos en la parte baja del ranking**.
- Utiliza la comparación **real vs predicho** para:
  - Evaluar la confianza en el modelo.
  - Detectar áreas donde el modelo puede necesitar ajustes o reentrenamiento.

---

## Relación con otras pestañas

- La pestaña **Dashboard** se nutre de:
  - Los datasets cargados y preprocesados en **Datos**.
  - Los modelos entrenados y validados en **Modelos**.
  - Las predicciones generadas en **Predicciones** (especialmente en modo lote).

Es la vista que cierra el ciclo: de los datos brutos y el entrenamiento de
modelos, hasta la toma de decisiones basada en indicadores agregados.
