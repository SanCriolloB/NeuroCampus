# Arquitectura del frontend

El frontend de NeuroCampus está desarrollado con **React + TypeScript** y se
construye con **Vite**. Está organizado en páginas, componentes reutilizables,
servicios de acceso a la API y un layout común.

---

## Estructura de directorios

Ubicación base del frontend:

```text
frontend/
  index.html
  package.json
  vite.config.ts
  src/
    App.tsx
    layout/
    pages/
    components/
    services/
    routes/
    styles/
```

### 1. Layout principal y navegación

- `src/layout/MainLayout.tsx`:
  - Define el layout de la aplicación:
    - Sidebar a la izquierda (`Sidebar.tsx`).
    - Barra superior (`Topbar.tsx`).
    - Área principal de contenido (`<Outlet />`).

- `src/layout/Sidebar.tsx`:
  - Contiene el menú de navegación:
    - Inicio,
    - Dashboard,
    - Prediction,
    - Models,
    - Jobs,
    - Datos.

- `src/routes/Router.tsx`:
  - Define las rutas de React Router:
    - `/` → `Home` / `App` (según configuración),
    - `/dashboard` → `Dashboard.tsx`,
    - `/models` → `Models.tsx`,
    - `/prediction` → `Prediction.tsx`,
    - `/jobs` → `Jobs.tsx`,
    - `/datos` → `DataUpload.tsx`,
    - `/datos/diagnostico` → `DatosDiagnostico.tsx`.

La aplicación se renderiza dentro de `MainLayout`, que aporta una experiencia
consistente en todas las pestañas.

---

## Páginas principales

- `src/pages/DataUpload.tsx`:
  - Implementa la pestaña **Datos**:
    - Carga y validación de datasets.
    - Resumen estructural del dataset.
    - Lanzamiento y visualización de BETO (análisis de sentimientos).
    - Vista previa de filas.

- `src/pages/Models.tsx`:
  - Implementa la pestaña **Modelos**:
    - Configuración y lanzamiento de entrenamientos.
    - Comparación de métricas entre BM, RBM, DBM, etc.
    - Visualización de curvas de pérdida/accuracy y matrices de confusión.

- `src/pages/Prediction.tsx`:
  - Implementa la pestaña **Predicciones**:
    - Ingreso manual de nuevas evaluaciones.
    - Predicción por lote.
    - Visualización de probabilidades, radar de indicadores, barras comparativas y proyecciones temporales.

- `src/pages/Dashboard.tsx`:
  - Implementa la pestaña **Dashboard**:
    - Filtros por periodo, programa, docente, asignatura.
    - KPIs globales.
    - Gráficos de histórico, ranking de docentes, riesgo y real vs predicho.

- `src/pages/AdminCleanup.tsx`:
  - Interfaz de administración para limpiar artefactos y datasets temporales.

- `src/pages/Jobs.tsx`:
  - Muestra y gestiona jobs en ejecución/históricos (por ejemplo, jobs de BETO
    o entrenamientos).

---

## Componentes reutilizables

Ubicación: `src/components/`.

Ejemplos:

- `UploadDropzone.tsx`:
  - Zona de arrastre/selección de archivos.
  - Usado en pestaña Datos y, potencialmente, en Predicciones por lote.

- `ResultsTable.tsx`:
  - Tabla genérica para mostrar resultados o muestras de datos.

- `MetricCard.tsx`:
  - Tarjetas de KPI que se utilizan principalmente en el Dashboard.

- `ValidationReport.tsx`:
  - Muestra informes de validación de datasets (errores, advertencias, etc.).

Estos componentes encapsulan lógica de presentación y evitan duplicación de
código entre páginas.

---

## Servicios de acceso a la API

Ubicación: `src/services/`.

- `apiClient.ts`:
  - Cliente base sobre `fetch` o `axios` para acceder a la API del backend.
  - Maneja la URL base y algunas opciones comunes.

- `api.ts`:
  - Utilidades generales sobre el cliente base (por ejemplo, helpers para
    instanciar el cliente según entorno).

- `datos.ts`:
  - Funciones para consumir la API de datos:
    - `esquema()`,
    - `validar()`,
    - `upload()`,
    - `resumen()`,
    - `sentimientos()`.

- `jobs.ts`:
  - Funciones para lanzar y consultar jobs:
    - incluyendo el preprocesamiento BETO (`/jobs/preproc/beto/run`, etc.).

- `modelos.ts`:
  - Funciones para interactuar con endpoints de entrenamiento y consulta de
    modelos.

- `prediccion.ts`:
  - Funciones para generar predicciones manuales o por lote desde el frontend.

Esta estructura mantiene una separación clara entre:

- **Lógica de presentación** (páginas y componentes) y
- **Lógica de comunicación con la API** (servicios).

---

## Estilos y gráficos

- Estilos globales en `src/styles/index.css`:
  - Define tipografía, colores base, disposición general.
- Gráficos:
  - Principalmente con **Recharts**:
    - Gráficos de barras,
    - líneas,
    - radar,
    - gráficos apilados, etc.
  - Usados en:
    - Pestaña Datos (sentimientos),
    - Modelos (curvas de entrenamiento, matrices de confusión),
    - Dashboard (históricos, rankings, distribuciones de riesgo).

---

Esta arquitectura del frontend permite:

- Crecer en número de páginas y componentes sin perder orden.
- Probar cada pieza de forma aislada (tests en `src/pages/__tests__` y
  `src/services/*.test.ts`).
- Reutilizar componentes visuales y servicios API en todas las pestañas.
