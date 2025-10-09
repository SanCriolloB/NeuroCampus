# Ejemplos y datos de prueba (Día 3)

Este directorio contiene datasets y guías rápidas para **probar la validación** de datos
(**/datos/validar**) y verificar la **normalización de encabezados** y la **coerción de tipos**.

## Archivos

- `dataset_ejemplo.csv` → dataset mínimo (headers “canónicos”).
- `Evaluacion.csv` → dataset realista (puede traer espacios, acentos o `:` en headers).
- `plantilla.xlsx` → ejemplo de planilla (estructura base).
- `requests.http` (opcional) → ejemplos de llamadas a la API local.

---

## Cómo probar rápido

### 1) Levantar el backend
```bash
# desde la raíz del repo
python -m uvicorn neurocampus.app.main:app --reload --app-dir backend/src
```

> Si el esquema no se encuentra automáticamente:
```bash
# Windows PowerShell
$env:NC_SCHEMA_PATH = "C:\ruta\al\repo\schemas\plantilla_dataset.schema.json"

# Git Bash
export NC_SCHEMA_PATH="/c/ruta/al/repo/schemas/plantilla_dataset.schema.json"
```

> Para usar Polars como engine (opcional):
```bash
# PowerShell
$env:NC_DF_ENGINE = "polars"
# Git Bash
export NC_DF_ENGINE="polars"
```

### 2) Validar datasets
```bash
# CSV mínimo (headers canónicos)
curl -X POST http://127.0.0.1:8000/datos/validar -F "file=@examples/dataset_ejemplo.csv"

# CSV “realista” (espacios/acentos/“:”; separadores ; o ,)
curl -X POST http://127.0.0.1:8000/datos/validar -F "file=@examples/Evaluacion.csv"

# XLSX (requiere openpyxl)
curl -X POST http://127.0.0.1:8000/datos/validar -F "file=@examples/plantilla.xlsx" -F "fmt=xlsx"
```

---

## Normalización de encabezados (lo que la API tolera)

La validación **acepta encabezados con variantes** y los mapea a nombres canónicos:

- **Minúsculas**, **sin acentos**, **sin puntuación** (`: ; , . () [] { } / \`)  
- **Espacios** ⇄ **`_`** (equivalentes)  
- Se ignoran columnas `Unnamed: N`
- Se mantienen los nombres **canónicos** en el reporte

**Ejemplos que se consideran iguales:**
```
"codigo_materia" ≡ "Código Materia" ≡ "codigo materia" ≡ "codigo_asignatura"
"cedula_profesor" ≡ "cédula docente" ≡ "cedula profesor" ≡ "docente_id"
"Sugerencias:" ≡ "Sugerencias" ≡ "sugerencias"
"pregunta_1" ≡ "pregunta 1"   ...   "pregunta_10" ≡ "pregunta 10"
```

> No hace falta editar los CSV para cambiar espacios por `_`: la API lo resuelve.

---

## Diccionario de columnas (canónicas)

| Columna            | Tipo esperado                 | Requerida | Notas / Sinónimos admitidos (no exhaustivo) |
|--------------------|-------------------------------|-----------|---------------------------------------------|
| `periodo`          | `string` (regex `AAAA-(1|2)`) | ✔︎         | `2024-2`                                    |
| `codigo_materia`   | `string`                      | ✔︎         | `codigo materia`, `codigo_asignatura`       |
| `grupo`            | `integer (1–9999)`            | ✔︎         | `grupo_id`                                  |
| `materia`          | `string`                      | ✔︎         | —                                           |
| `cedula_profesor`  | `string`                      | ✔︎         | `cedula profesor`, `cedula docente`, `docente_id` |
| `profesor`         | `string`                      | ✔︎         | —                                           |
| `pregunta_1`…`_10` | `number (0–50)`               | ✔︎         | Acepta `pregunta 1`…`pregunta 10`           |
| `Sugerencias:`     | `string | null`               | ✖︎         | `Sugerencias`, `sugerencias`                 |

> Los IDs se tipan como **string** (evita pérdida de ceros a la izquierda o overflow).
> Las preguntas `1..10` se validan con rango **0–50**.

---

## Coerción de tipos (antes de comparar)

Si el schema tipa una columna, se intentará convertir el dato:

- `string` → convierte números/enteros a texto (IDs).
- `integer` → `Int64` (pandas) / `Int64` (polars) con `to_numeric`.
- `number` → `float`.
- `boolean` → mapea `true/false/1/0`.
- `date/datetime` → `to_datetime(..., errors="coerce")`.
- **Nullables** (p. ej. `["string","null"]`) se respetan.

---

## Lectura de archivos (robusta)

- **CSV**: autodetección de **encoding** (`utf-8-sig`, `latin-1`) y **separador** (`,`, `;`, `\t`, `|`).  
- **XLSX**: con Polars se lee vía pandas y se convierte internamente.  
- **Parquet**: lectura nativa.

---

## Criterios de aceptación (QA)

- Un CSV con encabezados `pregunta 1`…`pregunta 10` debe validar como si fueran `pregunta_1`…`pregunta_10`.  
- `codigo_materia` y `cedula_profesor` numéricos en el CSV deben **coaccionarse a string** sin `BAD_TYPE`.  
- `Sugerencias:` puede venir vacío/nulo sin error.  
- Si `periodo` no cumple `^\d{4}-(1|2)$`, debe reportarse `DOMAIN_VIOLATION`.  
- Duplicados de fila completa aparecen como `DUPLICATE_ROW` (warning).  
- Columnas con ≥20% nulos se reportan como `HIGH_NULL_RATIO` (warning).

---

## Problemas comunes

- **No se encuentra el schema** → define `NC_SCHEMA_PATH` apuntando a `schemas/plantilla_dataset.schema.json`.  
- **XLSX falla** → instala `openpyxl` (`pip install openpyxl`).  
- **Separador `;`** en CSV → la API lo detecta; no es necesario editar el archivo.  
- **Engine** → por defecto pandas; para polars `NC_DF_ENGINE=polars`.

---
