# Inferencia vía API (NeuroCampus)

> ⚠️ Nota (P2.2):
> El backend actualmente implementa `/predicciones/predict` como **resolve/validate** del predictor bundle (sin ejecutar inferencia real).
> La inferencia real está prevista para P2.4+.
> Para el contrato vigente y ejemplos actualizados ver `docs/predicciones.md`.

Este documento explica cómo invocar la API **vigente en P2.2** para:
- resolver `run_id` (directo o vía champion),
- validar que exista el bundle mínimo (`predictor.json`, `model.bin`, etc.),
- y obtener metadata del predictor (sin generar predicciones aún).

---

## 1) Requisitos

- Backend en marcha, por ejemplo:
  ```bash
  uvicorn neurocampus.app.main:app --reload --app-dir backend/src
  ```
- Artifacts disponibles en `NC_ARTIFACTS_DIR` (por defecto `<repo>/artifacts`).
- Para modo champion: debe existir `champion.json` en:
  - `artifacts/champions/<family>/<dataset_id>/champion.json`

---

## 2) Selección del predictor

Puedes invocar `/predicciones/predict` de dos formas:

### A) Por run_id (directo)
Usa el `run_id` retornado por el flujo de entrenamiento (P0/P1).

Request:
```json
{ "run_id": "<run_id>" }
```

### B) Por champion (recomendado para consumo)
Request:
```json
{ "use_champion": true, "dataset_id": "<dataset_id>", "family": "<family>" }
```

El backend leerá:
- `artifacts/champions/<family>/<dataset_id>/champion.json`
- y tomará `source_run_id` para resolver el bundle del run.

> Si `champion.json` existe pero NO trae `source_run_id`, debe responder **422**.

---

## 3) Endpoints vigentes (P2.2)

Base URL (local): `http://127.0.0.1:8000`  
Docs Swagger: `http://127.0.0.1:8000/docs`

### 3.1 GET /predicciones/health

Ejemplo:
```bash
curl -s "http://127.0.0.1:8000/predicciones/health"
```

### 3.2 POST /predicciones/predict (resolve/validate)

#### Ejemplo por run_id
```bash
curl -s -X POST "http://127.0.0.1:8000/predicciones/predict" \
  -H "Content-Type: application/json" \
  -d '{ "run_id": "<run_id>" }'
```

#### Ejemplo por champion
```bash
curl -s -X POST "http://127.0.0.1:8000/predicciones/predict" \
  -H "Content-Type: application/json" \
  -d '{ "use_champion": true, "dataset_id": "<dataset_id>", "family": "<family>" }'
```

---

## 4) Códigos de respuesta esperados

- **200**: bundle resuelto y válido.
- **404**: no existe champion o el run/bundle no existe (faltan archivos como `predictor.json`/`model.bin`).
- **422**: predictor “no listo” (placeholder) o champion inválido (ej. `champion.json` sin `source_run_id`), o request incompleto.
- **500**: solo para errores inesperados.

---

## 5) Errores comunes y solución

- **404 Not Found**
  - Champion inexistente o run/bundle incompleto.
  - Acción: promueve un run “completo” como champion (P0/P1) o verifica que el run tenga `predictor.json` y `model.bin`.

- **422 Unprocessable Entity**
  - Falta `run_id` cuando `use_champion` es falso.
  - `champion.json` sin `source_run_id`.
  - `model.bin` placeholder/no listo.

- **Problemas de quoting JSON en Windows/Git Bash**
  - Acción: usa archivos `payload.json` y `--data-binary @payload.json`, o heredoc con cuidado.

---

## 6) Ejemplo en Python (requests)

```python
import requests

url = "http://127.0.0.1:8000/predicciones/predict"
payload = {"use_champion": True, "dataset_id": "2025-1", "family": "sentiment_desempeno"}

r = requests.post(url, json=payload, timeout=30)
print(r.status_code, r.json())
```

---

## 7) Roadmap

- **P2.4+**: inferencia real (predicciones), persistencia de outputs y soporte de batch.
- En P2.2 este endpoint es solo **resolve/validate** del bundle.
