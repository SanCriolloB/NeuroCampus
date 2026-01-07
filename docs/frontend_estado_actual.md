# Estado Actual del Frontend

## Rutas y Componentes
A continuación, se documentan las rutas principales del frontend actual y el estado de cada una.

### Rutas principales
- **`/`**: Página de inicio (Dashboard o Home)
- **`/dashboard`**: Vista del dashboard
- **`/datos`**: Página de datos (actual, basada en el backend)
- **`/datos/v2`**: Nueva vista de datos, aún no implementada completamente
- **`/models`**: Página de modelos (actual)
- **`/prediction`**: Página de predicciones (actual)
- **`/jobs`**: Página de jobs (actual)

### Estado de las rutas
- **Rutas Legacy**:
  - `/datos`, `/models`, `/prediction`
- **Nuevas rutas (con prototipo de Data Flow)**:
  - `/datos/v2` (borrador basado en prototipo, aún por completar)
  - `/models/v2` (a implementar)
  - `/prediction/v2` (a implementar)
  
### Componentes
- **`App.tsx`**: Componente principal de la aplicación.
- **`Sidebar.tsx`**: Sidebar de navegación.
- **`Dashboard.tsx`**: Dashboard legacy, debe ser reemplazado por el prototipo.
- **`DataUpload.tsx`**: Componente actual de la página de datos.
