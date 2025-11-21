# Despliegue y operación

Esta guía ofrece una visión general de cómo desplegar NeuroCampus en un entorno
más cercano a producción. Los detalles concretos (orquestador, proveedor cloud,
etc.) pueden variar, pero los pasos conceptuales suelen ser similares.

---

## 1. Construir artefactos

### Backend

El backend es una aplicación FastAPI que se puede empaquetar como:

- Una imagen Docker, o
- Una aplicación Python servida con Uvicorn/Gunicorn en un servidor.

Ejemplo de arranque con Uvicorn (modo producción básico):

```bash
cd backend
uvicorn neurocampus.app.main:app --host 0.0.0.0 --port 8000
```

Para entornos más exigentes se recomienda usar:

- `gunicorn` con workers `uvicorn`:

  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker neurocampus.app.main:app     --workers 4 --bind 0.0.0.0:8000
  ```

### Frontend

El frontend (Vite + React) debe compilarse en modo producción:

```bash
cd frontend
npm install
npm run build
```

Esto genera archivos estáticos (generalmente en `frontend/dist/`) que pueden:

- Servirse desde un servidor web (Nginx, Apache, etc.).
- Servirse a través de un CDN.
- O integrarse en una imagen Docker junto con un servidor HTTP ligero.

---

## 2. Configuración de entorno

Variables de entorno recomendadas:

- **Backend**:
  - `NEURO_ENV` (`dev`, `staging`, `prod`).
  - `NEURO_DATA_DIR`: ruta persistente donde se almacenan datasets y artefactos.
  - Configuración de logging (si se parametriza por entorno).
  - Cadenas de conexión a bases de datos (si aplica).

- **Frontend**:
  - Variables que definen la URL base de la API, por ejemplo:
    - `VITE_API_BASE_URL=http://backend:8000`

En Vite, las variables de entorno de **build** deben empezar por `VITE_` para
ser accesibles desde el código del frontend.

---

## 3. Configuración de red y dominios

Una configuración típica puede ser:

- Backend (`FastAPI`):
  - Corriendo en `http://backend:8000` (interno).
- Frontend:
  - Servido desde `http://app:4173` o un servidor web (Nginx) en el puerto 80/443.
- Proxy reverso (Nginx/Traefik/etc.) delante, con rutas como:
  - `/api` → backend FastAPI.
  - `/` → frontend estático.

Ejemplo conceptual de Nginx:

```nginx
location /api/ {
    proxy_pass http://backend:8000/;
}

location / {
    root /usr/share/nginx/html;  # carpeta con el build de Vite
    try_files $uri /index.html;
}
```

> Ajustar las rutas (`/api`, `/datos`, etc.) según la convención utilizada en
> el proyecto (por ejemplo, si la API está montada en `/`).

---

## 4. Persistencia de datos y artefactos

Es importante que las rutas donde se almacenan:

- datasets procesados,
- artefactos de modelos,
- logs relevantes,

estén ubicadas en volúmenes persistentes (en Docker) o en carpetas que no se
pierdan al actualizar el despliegue.

Si se usan contenedores, se recomienda montar volúmenes como:

- `/app/data` para datasets y artefactos.
- `/app/logs` para logs de aplicación (si no se envían a un sistema central).

---

## 5. Monitorización y logging

El backend incluye una capa de observabilidad:

- Middleware de **Correlation-Id** para trazar peticiones.
- Configuración de logging estructurado.

Recomendaciones:

- Redirigir logs a `stdout`/`stderr` en contenedores para que el orquestador
  (Kubernetes, Docker Compose, etc.) los recoja.
- Integrar con herramientas de monitorización (Prometheus, ELK/EFK, etc.)
  según las necesidades del entorno.

---

## 6. Actualizaciones y migraciones

Cuando se actualiza el proyecto:

1. Construir nuevos artefactos (backend y frontend).
2. Aplicar migraciones de datos si son necesarias (por ejemplo, cambios en
   estructura de datasets o rutas de artefactos).
3. Desplegar versiones nuevas de forma controlada:
   - Blue/green deployment,
   - rolling updates,
   - o simplemente parar/levantar en entornos pequeños.

Es recomendable mantener:

- Copias de seguridad de `NEURO_DATA_DIR` antes de cambios mayores.
- Versionado de modelos y artefactos (por ejemplo, incluyendo el `dataset_id`
  y la fecha en los nombres).

---

## 7. Despliegue de documentación

Si se utiliza Sphinx y GitHub Pages:

1. Construir la documentación localmente:

   ```bash
   make docs-html
   ```

2. O bien, configurar un workflow de GitHub Actions que:

   - Instale dependencias de `backend/requirements-dev.txt`.
   - Ejecute `cd docs && make html`.
   - Publique `docs/build/html/` en la rama `gh-pages`.

De esta forma, la documentación se mantiene actualizada con cada cambio en la
rama principal del proyecto.

---

Esta guía ofrece una referencia general; los detalles exactos de despliegue
dependerán del entorno (servidor propio, PaaS, contenedores, Kubernetes, etc.).
En cualquier caso, la separación backend/frontend y el uso de variables de
entorno facilitan la adaptación a distintos escenarios.
