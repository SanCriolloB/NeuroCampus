# Runbook Backend — Modelos (P0/P1)

Este runbook describe cómo operar y verificar el backend de **Modelos** sin tocar frontend.

## Prerrequisitos

- Python instalado (idealmente el que usa el proyecto).
- Entorno virtual activo.
- Dependencias instaladas.

Variables de entorno relevantes:

- `NC_ARTIFACTS_DIR` (opcional): directorio físico donde se escriben artifacts.
  - Si no existe, se usa `<repo>/artifacts`.
  - El backend y los tests usan *referencias lógicas* tipo `artifacts/...` aunque el storage real esté en otro lugar.
- `NC_PROJECT_ROOT` (opcional): raíz del repo si ejecutas desde un cwd raro.

## Gates (calidad / reproducibilidad)

Desde la raíz del repo:

```bash
make lint
make be-test
make be-ci
