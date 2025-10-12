# Metodologías de entrenamiento (PeriodoActual, Acumulado, Ventana)

## Resumen
- **periodo_actual**: entrena SOLO con el periodo actual.
- **acumulado**: entrena con todos los periodos <= periodo actual.
- **ventana**: entrena con los últimos N periodos (por defecto, N=4).

## Parámetros
- `metodologia`: "periodo_actual" | "acumulado" | "ventana" (default: "periodo_actual")
- `periodo_actual`: "YYYY-SEM" (p.ej., "2024-2"). Si se omite, se infiere el máximo presente.
- `ventana_n`: entero > 0 (solo para ventana; default 4)

## Ejemplos (curl)
### 1) Periodo actual inferido
curl -s -X POST http://127.0.0.1:8000/modelos/entrenar \
  -H "Content-Type: application/json" \
  -d '{"modelo":"rbm_general","epochs":5,"metodologia":"periodo_actual"}'

### 2) Acumulado hasta 2024-2
curl -s -X POST http://127.0.0.1:8000/modelos/entrenar \
  -H "Content-Type: application/json" \
  -d '{"modelo":"rbm_general","epochs":5,"metodologia":"acumulado","periodo_actual":"2024-2"}'

### 3) Ventana (últimos 6 periodos)
curl -s -X POST http://127.0.0.1:8000/modelos/entrenar \
  -H "Content-Type: application/json" \
  -d '{"modelo":"rbm_general","epochs":5,"metodologia":"ventana","ventana_n":6}'
