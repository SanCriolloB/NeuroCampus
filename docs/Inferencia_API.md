# Inferencia vía API (NeuroCampus)

Este documento explica cómo **invocar la API** de NeuroCampus para hacer inferencia con el modelo *campeón* entrenado (Student RBM con o sin texto). Incluye los **endpoints**, **payloads** de ejemplo, reglas de decisión y resolución de errores comunes.

> Requisitos: backend en marcha (`uvicorn neurocampus.app.main:app --reload --app-dir backend/src`) y un **modelo campeón** copiado en `artifacts/champions/.../current` (ver `Entrenamiento.md`).

---

## 1) Carga del modelo campeón

El backend busca el modelo activo en una ruta *champion*. Puedes configurar por **variable de entorno**:

```
CHAMPION_WITH_TEXT=artifacts/champions/with_text/current
```

Si no existe el campeón, el endpoint responderá **500** indicando que no hay modelo cargado.

---

## 2) Endpoints

Base URL (local): `http://127.0.0.1:8000`  
Documentación interactiva: `http://127.0.0.1:8000/docs`  
Esquema OpenAPI: `http://127.0.0.1:8000/openapi.json`

### 2.1 `POST /prediccion/online`

Predicción **sincrónica** para **un** registro.

- **Body (JSON)**
  - El esquema espera un objeto raíz con la clave **`input`**.
  - Campos dentro de `input`:
    - `calificaciones`: objeto con **10** calificaciones, claves `pregunta_1` … `pregunta_10` (float).
    - `comentario`: texto libre (string). Puede ser vacío si el modelo campeón no usa texto.

**Ejemplo (curl, Git Bash, heredoc):**
```bash
curl -s -X POST "http://127.0.0.1:8000/prediccion/online"   -H "Content-Type: application/json; charset=utf-8"   --data-binary @- <<'JSON'
{"input":{
  "calificaciones":{"pregunta_1":4.5,"pregunta_2":4.0,"pregunta_3":3.8,"pregunta_4":4.2,"pregunta_5":4.6,
                    "pregunta_6":4.3,"pregunta_7":4.1,"pregunta_8":4.4,"pregunta_9":4.0,"pregunta_10":4.5},
  "comentario":"La metodología fue clara y el profesor resolvió dudas con paciencia."
}}
JSON
```

**Respuesta (JSON, ejemplo):**
```json
{
  "proba": [0.12, 0.20, 0.68],        // [p_neg, p_neu, p_pos]
  "label": "pos",                      // etiqueta final con regla costo-sensible
  "decision_rule": "pos_if_ppos>=0.55" // (opcional) texto de la regla aplicada
}
```

> **Nota:** Si envías el cuerpo **sin** la clave `input`, FastAPI devolverá **422** con `"Field required: input"`.

**Regla de decisión costo-sensible (servidor):**
- Si `p_pos ≥ 0.55` ⇒ **pos**
- Si no, y `p_neg ≥ 0.35` **o** `(p_neg - p_neu) ≥ 0.05` ⇒ **neg**
- En caso contrario ⇒ **neu**

---

### 2.2 `POST /prediccion/batch`

Predicción **por lotes** leyendo un **archivo CSV** cargado como formulario `multipart/form-data`.

- **Parámetro de formulario**: `file` (CSV)
- **Formato CSV esperado** (encabezados):
  - `id` (opcional, para rastreo)
  - `comentario`
  - `pregunta_1` … `pregunta_10`

**Ejemplo (CSV mínimo):**
```csv
id,comentario,pregunta_1,pregunta_2,pregunta_3,pregunta_4,pregunta_5,pregunta_6,pregunta_7,pregunta_8,pregunta_9,pregunta_10
a1,"Clase clara y ordenada",4.5,4.0,3.8,4.2,4.6,4.3,4.1,4.4,4.0,4.5
a2,"Me gustaría más ejemplos",3.5,3.0,3.2,3.8,3.6,3.3,3.1,3.4,3.0,3.5
```

**Ejemplo (curl):**
```bash
curl -s -X POST "http://127.0.0.1:8000/prediccion/batch"   -H "Accept: application/json"   -F "file=@/ruta/a/tu/lote.csv" | jq .
```

**Respuesta (JSON, ejemplo de resumen):**
```json
{
  "received": 2,
  "predicted": 2,
  "failed": 0,
  "items": [
    {"id":"a1","label":"pos","proba":[0.10,0.22,0.68]},
    {"id":"a2","label":"neu","proba":[0.18,0.60,0.22]}
  ]
}
```
> La forma exacta puede variar según la implementación actual del *facade*; el router devuelve el **resumen** de `predict_batch`. Si necesitas las filas completas, adapta el router para retornar `items` además del resumen.

---

## 3) Errores y solución de problemas

- **422 Unprocessable Entity**
  - Causa típica: falta `{"input": ...}` en `/prediccion/online`.
  - Acción: envuelve tu JSON dentro de la clave **`input`**.
- **400 Bad Request (batch)**
  - CSV con columnas inválidas o formato inesperado.
  - Acción: verifica que existan `comentario` y `pregunta_1..pregunta_10`. Usa comillas si hay comas en el texto.
- **500 Internal Server Error**
  - Campeón no encontrado o error cargando pesos.
  - Acción: comprueba que `artifacts/champions/.../current` contenga `vectorizer.json`, `rbm.pt`, `head.pt` y que la variable `CHAMPION_WITH_TEXT` (o la ruta por defecto) sea correcta.
- **“There was an error parsing the body” (Starlette)**
  - JSON mal citado en shell Windows/Git Bash.
  - Acción: usa **heredoc** o archivo `payload.json` y `--data-binary @payload.json`.

> Revisa siempre los **logs** de Uvicorn en la consola; muestran trazas y la regla de decisión aplicada.

---

## 4) Buenas prácticas de cliente

- **Git Bash en Windows**: usa heredoc o archivos para el cuerpo JSON, evita backslashes y comillas problemáticas.
- **Time-outs**: para lotes grandes usa `batch` y divide tus archivos. Mantén el tamaño de texto razonable.
- **CORS**: si vas a consumir desde frontend, configura CORS en el backend (middleware) según tu dominio.
- **Versionado de modelos**: documenta el `JOB_ID` del campeón en `artifacts/champions/.../CHAMPION.json` para trazabilidad.

---

## 5) Ejemplo en Python (`requests`)

```python
import requests

url = "http://127.0.0.1:8000/prediccion/online"
payload = {
    "input": {
        "calificaciones": {
            "pregunta_1":4.5,"pregunta_2":4.0,"pregunta_3":3.8,"pregunta_4":4.2,"pregunta_5":4.6,
            "pregunta_6":4.3,"pregunta_7":4.1,"pregunta_8":4.4,"pregunta_9":4.0,"pregunta_10":4.5
        },
        "comentario": "La metodología fue clara y el profesor resolvió dudas con paciencia."
    }
}
r = requests.post(url, json=payload, timeout=30)
print(r.status_code, r.json())
```

---

## 6) Compatibilidad y cambios

- El endpoint `/prediccion/online` usa la forma **anidada** con `input` por diseño del esquema Pydantic.
- Puedes añadir un endpoint alterno `/prediccion/online_v2` que acepte el body **plano** y lo envuelva internamente para retrocompatibilidad.
- La lógica de decisión puede ajustarse en el *facade* o en el *strategy*; documenta cualquier cambio para el frontend.

---

## 7) Checklist antes de consumir en producción

- [ ] Modelo campeón actualizado en `artifacts/champions/.../current`.
- [ ] Variable `CHAMPION_WITH_TEXT` (o equivalente) apuntando al campeón.
- [ ] Backend **activo** y accesible; `/docs` abre correctamente.
- [ ] Prueba **online** con payload de ejemplo (respuesta `200` con `proba` y `label`).
- [ ] (Opcional) Prueba **batch** con un CSV pequeño.
- [ ] Logs verificados, sin excepciones persistentes.

---

¿Dudas o mejoras? Actualiza este documento junto con el *README* cuando cambien los endpoints o la regla de decisión.
