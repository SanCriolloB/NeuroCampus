# Validación e Ingesta de Datasets — Diagnóstico (Día 5)

## Objetivo
Aislar y documentar las causas de error al validar/cargar datasets desde FE hacia BE.

## Checklist rápido
- [ ] Backend levanta sin errores (`uvicorn`).
- [ ] CORS permite `http://localhost:5173`.
- [ ] Endpoint `/datos/validar` responde 200 con body esperado.
- [ ] `validadores.py` expone `run_validations(df, *, dataset_id)`.
- [ ] `examples/*.csv` pasan validación mínima.

## Reproducibilidad
```bash
# Backend
make be-dev  # o: uvicorn neurocampus.app.main:app --reload

# Frontend
make fe-dev  # Vite en :5173

# Probar directamente el endpoint (sin FE):
curl -F "file=@examples/docentes.csv" -F "dataset_id=docentes" \
  http://localhost:8000/datos/validar
