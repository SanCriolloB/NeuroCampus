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

# Día 2 — Borrado real seguro

### Comandos
- `make clean-artifacts-dry-run` — simulación sin mover archivos.
- `make clean-artifacts` — **mueve a papelera** (requiere `--force` dentro del script, invocado por el target).

### Papelera
- Ruta: `.trash/YYYYMMDD/<ruta_relativa>`
- Retención: `NC_TRASH_RETENTION_DAYS` días (default 14).

### Logs
- CSV en `logs/cleanup.log` con columnas: `timestamp,action,path,size,age_days,reason`.

### Exclusiones
- `NC_EXCLUDE_GLOBS` (globs separados por coma). Por default protegen `artifacts/champions/**`.

