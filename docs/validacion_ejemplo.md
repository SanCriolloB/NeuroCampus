# Ejemplo de reporte de validación (Día 3)
Resumen + top de issues reales capturados en pruebas locales.
> Útil para Miembro B (UI) y para revisión rápida.

- summary.rows: 250
- summary.errors: 3
- summary.warnings: 7
- engine: pandas

| code              | severity | column       | row  | message                                      |
|-------------------|----------|--------------|------|----------------------------------------------|
| MISSING_COLUMN    | error    | pregunta_7   | —    | Columna requerida ausente: pregunta_7        |
| DOMAIN_VIOLATION  | error    | periodo      | —    | Valor fuera de dominio en periodo: 2024-13   |
| HIGH_NULL_RATIO   | warning  | Sugerencias: | —    | 45.6% nulos en Sugerencias:                  |

## CSV correcto (mínimo)
summary: rows=3, errors=0, warnings=1, engine=pandas
issues:
- HIGH_NULL_RATIO (warning) column=Sugerencias: message="Nulos 33%"