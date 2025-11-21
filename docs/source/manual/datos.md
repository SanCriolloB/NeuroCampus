# Pestaña «Datos» – Ingesta y análisis

## Objetivo

Registrar nuevos datasets de evaluaciones docentes, aplicar el preprocesamiento definido
en el backend y obtener un resumen estadístico y de sentimientos.

## Flujo de trabajo

1. **Ingreso de dataset**
   - Seleccionar archivo (`.csv`, `.xlsx`, `.xls`, `.parquet`).
   - Opciones:
     - *Sobrescribir si el dataset ya existe*.
     - *Aplicar preprocesamiento y actualizar resumen*.
     - *Ejecutar análisis de sentimientos con BETO*.

2. **Carga y validación**
   - Botón **«Validar sin guardar»**: envía el fichero al endpoint `/datos/validar` y
     muestra una vista previa de filas.
   - Botón **«Cargar y procesar»**: usa `/datos/upload`, actualiza el resumen y, si está
     marcado, lanza el job BETO.

3. **Resumen del dataset**
   - Métricas principales: número de filas, columnas, docentes y asignaturas.
   - Rango de fechas, periodos, descripción de columnas.

4. **Análisis de sentimientos con BETO**
   - Estado del job BETO.
   - Gráficas:
     - Distribución global (positivo / neutro / negativo).
     - Distribución por docente (top 10).
