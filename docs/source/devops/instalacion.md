# Instalación y entorno de desarrollo

Esta guía explica cómo preparar un entorno de desarrollo para trabajar con
NeuroCampus (backend + frontend + datos).

---

## Requisitos previos

- **Git**
- **Python** >= 3.10 (recomendado 3.11)
- **Node.js** LTS (por ejemplo, 18.x o 20.x) + **npm**
- Herramientas de compilación básicas (dependiendo del sistema operativo)

Opcional, pero recomendado:

- Entorno virtual para Python (venv, conda, etc.).
- `make` instalado (en Windows se puede usar Git Bash, WSL o herramientas
  equivalentes).

---

## Clonar el repositorio

```bash
git clone https://github.com/SanCriolloB/NeuroCampus.git
cd NeuroCampus
```

A partir de aquí asumimos que estamos en la raíz del proyecto.

---

## Backend (Python / FastAPI)

### 1. Crear y activar entorno virtual

```bash
cd backend
python -m venv .venv
# En Linux/macOS
source .venv/bin/activate
# En Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Para desarrollo y documentación
pip install -r requirements-dev.txt
```

> Si solo existe `requirements.txt`, puedes concentrarte en ese archivo.
> `requirements-dev.txt` se recomienda para tests, linters y Sphinx.

### 3. Variables de entorno básicas

Dependiendo de la configuración, puede ser útil definir:

- `NEURO_DATA_DIR`: ruta base donde se guardan datasets y artefactos.
- `NEURO_ENV`: entorno actual (`dev`, `staging`, `prod`).

Ejemplo en Linux/macOS:

```bash
export NEURO_DATA_DIR="$(pwd)/../data"
export NEURO_ENV="dev"
```

En Windows (PowerShell):

```powershell
$env:NEURO_DATA_DIR = "$PWD\..\data"
$env:NEURO_ENV = "dev"
```

### 4. Ejecutar el backend en modo desarrollo

Desde `backend/`:

```bash
uvicorn neurocampus.app.main:app --reload --port 8000
```

O bien, usando el `Makefile` de la raíz si hay un target para backend:

```bash
cd ..   # volver a la raíz del repo
make be-dev
```

---

## Frontend (React + TypeScript + Vite)

### 1. Instalar dependencias

```bash
cd frontend
npm install
```

### 2. Ejecutar en modo desarrollo

```bash
npm run dev
```

Por defecto, Vite usará un puerto como `5173` (o similar). Para abrir la
aplicación:

- Backend: `http://localhost:8000/docs` (documentación interactiva de FastAPI).
- Frontend: `http://localhost:5173/` (UI de NeuroCampus).

Si el proyecto utiliza un `Makefile` con target para el frontend, también se
puede emplear:

```bash
cd ..   # raíz del repo
make fe-dev
```

---

## Estructura básica de carpetas

Resumen a alto nivel:

```text
NeuroCampus/
  backend/
    src/neurocampus/
      app/
      data/
      models/
      observability/
      validation/
      ...
  frontend/
    src/
      pages/
      components/
      services/
      routes/
      ...
  data/
    raw/
    processed/
    labeled/
  reports/
  docs/   (documentación Sphinx, si se usa)
```

Esta organización facilita trabajar de forma independiente sobre el backend, el
frontend y la documentación, manteniendo al mismo tiempo una estructura común
para datos y experimentos.

---

## Primeros pasos recomendados

1. Levantar backend y frontend en modo desarrollo.
2. Probar la pestaña **Datos** con un dataset pequeño de ejemplo.
3. Revisar la documentación de la API (`/docs`) para entender los endpoints
   disponibles.
4. Ejecutar los tests básicos (ver sección **Tests**).