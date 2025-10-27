# Limpieza de artefactos y temporales (Días 1–4)

## Comandos
- `make clean-inventory` — inventario resumido, sin eliminación.
- `make clean-artifacts-dry-run` — simulación de borrado con `--keep-last` y `--retention-days`.

## Variables
- `NC_RETENTION_DAYS` (default 90)
- `NC_KEEP_LAST` (default 3)
- `NC_DRY_RUN` (true/false)

## Seguridad
- Artefactos bajo `artifacts/champions/*` están protegidos.
- En Día 1, el borrado real **está deshabilitado** desde el script.

## Próximos días
- Día 2–4: borrado real, CLI `--force`, endpoint `POST /admin/cleanup` bajo auth.
