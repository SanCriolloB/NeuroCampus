# Endpoints de administración

La API de administración agrupa endpoints pensados para tareas de **mantenimiento**
y **limpieza** de artefactos relacionados con datos y modelos. Estos endpoints
suelen estar protegidos o restringidos a usuarios con permisos elevados.

Todos los endpoints aquí descritos cuelgan del prefijo base:

- `/admin` (o `/admin/cleanup`, según la estructura concreta del backend)

---

## Resumen de endpoints principales

Los nombres exactos pueden variar, pero típicamente se exponen operaciones como:

| Método | Ruta                          | Descripción                                      |
| ------ | ----------------------------- | ----------------------------------------------- |
| GET    | `/admin/cleanup/preview`      | Muestra un resumen de lo que se puede limpiar   |
| POST   | `/admin/cleanup/datasets`     | Limpia datasets obsoletos o de prueba           |
| POST   | `/admin/cleanup/jobs`         | Limpia registros de jobs antiguos               |
| POST   | `/admin/cleanup/artifacts`    | Elimina artefactos de modelos no utilizados     |

> **Importante**: esta sección describe la intención de la API de administración.
> La implementación real puede agrupar o dividir endpoints de forma ligeramente
> diferente (por ejemplo, `/admin/cleanup` con payload que especifique qué
> limpiar).

---

## `GET /admin/cleanup/preview`

### Descripción

Devuelve un resumen de los elementos que podrían eliminarse mediante las
operaciones de limpieza:

- Datasets antiguos o marcados como de prueba.
- Registros de jobs que ya no son necesarios.
- Artefactos de modelos obsoletos o superseded.

Se utiliza en la pestaña de administración para que el usuario pueda ver qué
impacto tendrá la acción de limpieza antes de ejecutarla.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON típico:

  - `datasets`: lista resumida de datasets candidatos a limpieza.
  - `jobs`: lista resumida de jobs antiguos.
  - `artifacts`: lista de artefactos/módulos candidatos.
  - contadores globales por tipo.

---

## `POST /admin/cleanup/datasets`

### Descripción

Elimina datasets marcados como candidatos según alguna política de retención
(configuración, antigüedad, etiquetas, etc.).

### Entrada

- Cuerpo JSON opcional con criterios adicionales, por ejemplo:

  - `older_than`: fecha a partir de la cual los datasets se consideran antiguos.
  - `dry_run`: si es `true`, solo simula la limpieza sin ejecutarla.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON típico:

  - `deleted_count`: número de datasets eliminados.
  - `deleted_ids`: identificadores de los datasets afectados.

---

## `POST /admin/cleanup/jobs`

### Descripción

Elimina registros de jobs ya finalizados (por ejemplo, más antiguos que cierta
fecha o en estados terminales). Esto ayuda a mantener liviana la tabla de jobs
o las estructuras de almacenamiento que se usen.

### Entrada

- Cuerpo JSON opcional con filtros:
  - `status`: estados que se quieren limpiar (`completed`, `failed`, etc.).
  - `older_than`: fecha de corte.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON típico:

  - `deleted_count`: número de jobs borrados.

---

## `POST /admin/cleanup/artifacts`

### Descripción

Limpia artefactos de modelos ya no utilizados, por ejemplo:

- Modelos antiguos sustituidos por versiones más nuevas.
- Archivos de entrenamiento intermedio que ya no hacen falta.

### Entrada

- Cuerpo JSON opcional indicando criterios:
  - `keep_latest_per_dataset`: conservar solo el último modelo por dataset.
  - `dry_run`: simular limpieza.

### Respuesta

- Código `200 OK`.
- Cuerpo JSON típico:

  - `deleted_count`: número de artefactos eliminados.
  - `paths`: lista de rutas que se han eliminado (o se eliminarían en modo `dry_run`).

---

## Consideraciones de seguridad

Dado que estas operaciones pueden borrar información de forma irreversible, es
recomendable que:

- Estén protegidas por mecanismos de autenticación/autorización adecuados.
- Se registren en los logs con suficiente detalle (quién, cuándo, qué).
- Ofrezcan, en la interfaz de usuario, información clara antes de ejecutar
  acciones de limpieza masiva.
