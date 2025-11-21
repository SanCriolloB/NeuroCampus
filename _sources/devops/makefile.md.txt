# Targets del Makefile

El `Makefile` en la raíz del proyecto reúne comandos de uso frecuente para el
backend, el frontend, los tests, el procesamiento de datos y la documentación.
Esta sección describe los targets más importantes y su propósito.

> **Nota**: los nombres exactos de los targets pueden variar ligeramente según
> la versión del proyecto. Esta guía se centra en los más habituales y en su
> intención funcional.

---

## Variables principales

En la parte superior del `Makefile` se definen variables reutilizables:

```make
SRC_DIR       := backend/src
BACKEND_DIR   := backend
FRONTEND_DIR  := frontend
REPORTS_DIR   := reports
DATA_DIR      := data
EXAMPLES_DIR  := examples
OUT_DIR      ?= $(DATA_DIR)/prep_auto
```

Estas variables permiten escribir reglas más legibles, por ejemplo:

- `cd $(BACKEND_DIR)` en lugar de `cd backend`.
- Usar `$(DATA_DIR)` para referirse a la carpeta de datos.

---

## Desarrollo

Targets típicos para desarrollo (nombres aproximados):

- **`be-dev`**: arranca el backend en modo desarrollo.

  ```make
  be-dev:
	cd $(BACKEND_DIR) && uvicorn neurocampus.app.main:app --reload --port 8000
  ```

- **`fe-dev`**: arranca el frontend en modo desarrollo.

  ```make
  fe-dev:
	cd $(FRONTEND_DIR) && npm run dev
  ```

Estos comandos se ejecutan desde la raíz del proyecto:

```bash
make be-dev
make fe-dev
```

---

## Tests

### Backend

- **`be-test`**: ejecuta los tests de backend (normalmente con `pytest`).

  ```make
  be-test:
	cd $(BACKEND_DIR) && pytest
  ```

### Frontend

- **`fe-test`**: ejecuta los tests de frontend (Vitest).

  ```make
  fe-test:
	cd $(FRONTEND_DIR) && npm run test:run
  ```

Usar estos targets garantiza que se ejecutan los comandos de test de manera
consistente en todos los entornos.

---

## Procesamiento de datos y modelos

En muchos flujos de trabajo, el `Makefile` incluye targets para:

- **Preprocesamiento de datos**:

  ```make
  prep-all:
	cd $(BACKEND_DIR) && python -m neurocampus.app.jobs.cmd_preprocesar_batch
  ```

- **Entrenamiento de modelos** (RBM, DBM, etc.), siguiendo combinaciones como:

  ```make
  train-rbm:
	cd $(BACKEND_DIR) && python -m neurocampus.app.jobs.cmd_train_rbm_manual

  train-dbm:
	cd $(BACKEND_DIR) && python -m neurocampus.app.jobs.cmd_train_dbm_manual
  ```

Los nombres reales pueden cambiar, pero la idea es ofrecer un punto de entrada
sencillo para lanzar pipelines completos sin recordar comandos largos.

---

## Documentación (Sphinx)

Si se integra Sphinx en `docs/`, es habitual definir un target para generar la
documentación HTML:

```make
docs-html:
	cd docs && make html
```

De esta manera, desde la raíz del repositorio:

```bash
make docs-html
```

Genera la documentación estática en `docs/build/html/`, lista para ser servida
o publicada en GitHub Pages.

---

## Limpieza y utilidades

Otros targets que suelen aparecer:

- **`clean`**: elimina archivos temporales de Python (`__pycache__`, `.pytest_cache`, etc.).
- **`clean-data`**: limpia directorios de datos de ejemplo o temporales.
- **`lint`** o **`format`**: ejecutan herramientas como `black`, `isort`,
  `flake8`, `ruff`, etc. (si están configuradas).

Ejemplo genérico:

```make
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
```

---

## Recomendaciones de uso

- Usar `make` para las tareas repetitivas en lugar de recordar comandos largos.
- Documentar en comentarios de `Makefile` los targets menos evidentes.
- Mantener los targets que se utilizan en CI/CD alineados con los que se usan
  en desarrollo (por ejemplo, `be-test` y `fe-test`).