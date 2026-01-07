# _legacy_ui

Este directorio contiene el frontend anterior (legacy) de NeuroCampus.

Motivo:
- La reestructuración del frontend se está realizando “desde cero” para replicar 1:1 la UI de “Data Flow and Visualization”.
- Este código se conserva temporalmente solo como referencia.

Regla:
- No se debe importar nada desde `_legacy_ui` en el frontend nuevo.
- La capa `src/services/*` se mantiene como fuente de verdad para llamadas al backend.
