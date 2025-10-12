# Scripts utilitarios (backend)

Convención:
- Scripts ejecutables y versionados que apoyan QA/ops local y CI.
- Mantener idempotentes y con salidas claras.
- No poner credenciales ni rutas absolutas.

Ejemplos previstos (D5–D7):
- test_unificacion_quick.py  → smoke test de unificación/metodologías.
- predict_batch.py           → (D6) ejecutar predicciones por lote y volcar %.
- compare_models.py          → (D7) comparar métricas y seleccionar campeón.