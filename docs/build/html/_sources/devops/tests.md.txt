# Tests y calidad del código

Esta guía explica cómo ejecutar los tests de backend y frontend en NeuroCampus,
así como algunas recomendaciones para mantener la calidad del código.

---

## Ejecución de tests desde el Makefile

En la raíz del repositorio, el `Makefile` define objetivos (targets) que
agregan los comandos más frecuentes. Para los tests principales:

- **Tests de backend**:

  ```bash
  make be-test
  ```

- **Tests de frontend**:

  ```bash
  make fe-test
  ```

En general, estos targets se encargan de:

- Moverse a la carpeta adecuada (`backend/` o `frontend/`).
- Ejecutar el comando de test correspondiente (por ejemplo, `pytest` para
  backend, `vitest` para frontend).

---

## Tests de backend

Los tests de backend suelen implementarse con **pytest** y se ubican en
directorios como:

```text
backend/
  tests/
    test_*.py
```

Para ejecutarlos directamente (desde `backend/` y con el entorno virtual
activo):

```bash
pytest
```

O con más detalle:

- Listar tests:

  ```bash
  pytest -q
  ```

- Ejecutar tests con mayor verbosidad:

  ```bash
  pytest -vv
  ```

### Cobertura (opcional)

Si el proyecto incluye configuración de cobertura, se puede usar:

```bash
pytest --cov=neurocampus
```

Esto permite ver qué partes del backend están bien cubiertas y dónde sería
conveniente añadir más tests.

---

## Tests de frontend

El frontend utiliza **Vitest** (integrado en el ecosistema Vite) junto con
Testing Library para pruebas de componentes y páginas.

Para ejecutar los tests desde `frontend/`:

```bash
npm run test:run
```

O con el target global desde la raíz del repo:

```bash
make fe-test
```

Los tests suelen ubicarse en:

```text
frontend/
  src/
    services/
      *.test.ts
    pages/
      __tests__/
        *.test.tsx
```

### Buenas prácticas

- Probar tanto la **lógica de servicios** (llamadas a la API) como la
  **experiencia de usuario** (interacción con componentes).
- Utilizar `screen.getByRole`, `getByText`, etc. para aserciones robustas.
- Evitar acoplar los tests a detalles internos de implementación que cambian
  con frecuencia.

---

## Tests de integración y end-to-end (opcional)

Dependiendo de la evolución del proyecto, se pueden incorporar:

- **Tests de integración** que levanten una instancia de FastAPI en modo test y
  validen flujos completos (ingesta de datos, entrenamiento, predicción).
- **Tests end-to-end (E2E)** con herramientas como Playwright o Cypress para
  validar flujos completos desde la UI.

Estos tests no son obligatorios en una primera fase, pero aportan valor
especialmente de cara a despliegues en producción.

---

## Recomendaciones generales

- Ejecutar `make be-test` y `make fe-test` antes de subir cambios importantes.
- Mantener los tests rápidos para facilitar la iteración en desarrollo.
- Añadir tests cuando se incorporen nuevas funcionalidades críticas (por
  ejemplo, cambios en la lógica de modelos, en la pestaña Datos o en el
  Dashboard).
