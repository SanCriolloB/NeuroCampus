// frontend/src/services/datos.ts
// Cliente de API para el flujo de Datos:
//   - Obtener el esquema esperado.
//   - Validar archivos de entrada antes de persistirlos.
//   - Subir y registrar nuevos datasets.
//   - Consultar resúmenes estadísticos.
//   - Consultar análisis de sentimientos (BETO).
//
// Este fichero está pensado para ser el punto único de acceso desde la
// pestaña «Datos» del frontend. Los tipos están documentados para facilitar
// su uso y la generación de documentación automática.

import api from "./apiClient";

// ---------------------------------------------------------------------------
// Tipos para esquema y validación de datasets
// ---------------------------------------------------------------------------

/**
 * Descripción detallada de un campo del esquema de datos.
 * Se utiliza cuando el backend expone metadatos enriquecidos por columna.
 */
export type EsquemaField = {
  /** Nombre lógico de la columna (normalmente ya normalizado). */
  name: string;
  /** Tipo lógico (string, number, boolean, date, etc.). */
  dtype?: string | null;
  /** Indica si la columna es obligatoria para el esquema. */
  required?: boolean;
  /** Descripción corta y legible pensada para la UI. */
  desc?: string | null;
  /** Dominio de valores permitidos, si aplica. */
  domain?: unknown;
  /** Rango numérico permitido, si aplica. */
  range?: unknown;
  /** Longitud mínima (para cadenas), si aplica. */
  min_len?: number | null;
  /** Longitud máxima (para cadenas), si aplica. */
  max_len?: number | null;
};

/**
 * Respuesta flexible de GET /datos/esquema.
 *
 * El backend puede devolver:
 *  - Listas simples de columnas requeridas/opcionales.
 *  - Una colección de objetos detallados por campo (EsquemaField[]).
 */
export type EsquemaResp = {
  /** Versión del esquema expuesta por el backend. */
  version?: string;
  /** Lista simple de nombres de columnas requeridas. */
  required?: string[];
  /** Lista simple de nombres de columnas opcionales. */
  optional?: string[];
  /** Descripción detallada de cada campo, si el backend la proporciona. */
  fields?: EsquemaField[];
  /** Ejemplos o metadatos adicionales que pueda exponer el backend. */
  examples?: Record<string, unknown>;
};

/**
 * Elemento de detalle de un problema de validación.
 * Es opcional y solo se usa si el backend lo implementa.
 */
export type ValidarIssue = {
  /** Gravedad del problema detectado. */
  level: "info" | "warning" | "error";
  /** Código corto de error, útil para trazas o test automatizados. */
  code?: string;
  /** Mensaje legible para mostrar en la interfaz. */
  msg: string;
  /** Columna asociada al problema, si aplica. */
  col?: string | null;
};

/**
 * Respuesta de POST /datos/validar.
 *
 * Resume si el archivo respeta el esquema y puede incluir:
 *  - columnas faltantes o extra,
 *  - una muestra de filas interpretadas,
 *  - issues más estructurados para mostrar en la UI,
 *  - estadísticas básicas por columna.
 *
 * No todos los campos son obligatorios; el backend puede ir
 * enriqueciendo esta estructura sin romper al frontend.
 */
export type ValidarResp = {
  /** Indica si, en términos generales, el archivo supera la validación. */
  ok: boolean;
  /** Identificador lógico del dataset sobre el que se validó. */
  dataset_id?: string;
  /** Columnas requeridas que no se encontraron en el archivo. */
  missing?: string[];
  /** Columnas presentes en el archivo pero no contempladas en el esquema. */
  extra?: string[];
  /** Muestra de filas leídas, útil para renderizar una vista previa. */
  sample?: Array<Record<string, unknown>>;
  /** Mensaje global generada por el backend. */
  message?: string;
  /** Número estimado de filas interpretadas. */
  n_rows?: number;
  /** Número estimado de columnas interpretadas. */
  n_cols?: number;
  /** Problemas detallados asociados a columnas concretas. */
  issues?: ValidarIssue[];
  /** Estadísticas básicas por columna, si el backend las envía. */
  stats?: Record<string, unknown>;
};

/**
 * Respuesta de POST /datos/upload.
 *
 * Informa del resultado de la operación de ingesta y de cómo quedó
 * registrado el dataset en el sistema.
 */
export type UploadResp = {
  /** Indica si la operación de subida terminó sin errores. */
  ok: boolean;
  /** Identificador lógico final bajo el que se registró el dataset. */
  dataset_id?: string;
  /** Ruta física o nombre base con el que se guardó el archivo. */
  stored_as?: string;
  /** Mensaje de confirmación u observaciones adicionales. */
  message?: string;
};

// ---------------------------------------------------------------------------
// Tipos para resumen de dataset y análisis de sentimientos (BETO)
// ---------------------------------------------------------------------------

/**
 * Resumen de una columna del dataset tal como lo expone el backend.
 * Se utiliza en la pestaña Datos para construir la tabla descriptiva.
 */
export type ColumnaResumen = {
  /** Nombre normalizado de la columna (sin espacios ni acentos). */
  name: string;
  /** Tipo lógico detectado (string, number, boolean, etc.). */
  dtype: string;
  /** Número de filas no nulas en esta columna. */
  non_nulls: number;
  /** Muestra de valores distintos para ayudar a entender el contenido. */
  sample_values: string[];
};

/**
 * Resumen global del dataset.
 * Permite construir tarjetas KPI y la tabla de columnas en la UI.
 */
export type DatasetResumen = {
  /** Identificador lógico del dataset (por ejemplo «2024-2»). */
  dataset_id: string;
  /** Número total de filas. */
  n_rows: number;
  /** Número total de columnas. */
  n_cols: number;
  /** Lista de periodos detectados (si existe columna periodo). */
  periodos: string[];
  /** Fecha mínima (ISO) detectada en el dataset, si aplica. */
  fecha_min?: string | null;
  /** Fecha máxima (ISO) detectada en el dataset, si aplica. */
  fecha_max?: string | null;
  /** Número de docentes distintos, si el backend lo calcula. */
  n_docentes?: number | null;
  /** Número de asignaturas distintas, si el backend lo calcula. */
  n_asignaturas?: number | null;
  /** Resumen por columna para renderizar una tabla descriptiva. */
  columns: ColumnaResumen[];
};

/**
 * Etiquetas de sentimiento utilizadas por el modelo BETO.
 * Es recomendable mantener la misma convención que usa el backend.
 */
export type SentimentLabel = "neg" | "neu" | "pos";

/**
 * Conteo por etiqueta de sentimiento.
 */
export type SentimentBreakdown = {
  /** Etiqueta de sentimiento (neg/neu/pos). */
  label: SentimentLabel;
  /** Número de comentarios etiquetados con este sentimiento. */
  count: number;
  /** Proporción en [0,1] sobre el total de comentarios analizados. */
  proportion: number;
};

/**
 * Distribución de sentimientos agregada por grupo (docente, asignatura, etc.).
 */
export type SentimentByGroup = {
  /** Nombre del grupo (docente, asignatura u otro). */
  group: string;
  /** Lista de conteos por sentimiento para este grupo. */
  counts: SentimentBreakdown[];
};

/**
 * Respuesta completa de GET /datos/sentimientos.
 *
 * Se utiliza para alimentar las gráficas de:
 *  - distribución global de sentimientos, y
 *  - distribución por docente/asignatura.
 */
export type DatasetSentimientos = {
  /** Identificador lógico del dataset. */
  dataset_id: string;
  /** Número total de comentarios no vacíos analizados. */
  total_comentarios: number;
  /** Distribución global de sentimientos. */
  global_counts: SentimentBreakdown[];
  /** Distribución de sentimientos agregada por docente. */
  por_docente: SentimentByGroup[];
  /** Distribución de sentimientos agregada por asignatura. */
  por_asignatura: SentimentByGroup[];
};

// ---------------------------------------------------------------------------
// Funciones de cliente: esquema, validación y subida
// ---------------------------------------------------------------------------

/**
 * GET /datos/esquema
 *
 * Obtiene el esquema esperado por el backend para la ingesta de datos.
 * La pestaña Datos utiliza esta información para mostrar la plantilla
 * de columnas y ayudar al usuario a preparar sus archivos.
 */
export async function esquema() {
  const { data } = await api.get<EsquemaResp>("/datos/esquema");
  return data;
}

/**
 * POST /datos/validar (multipart)
 *
 * Envía un archivo al backend para comprobar si respeta el esquema
 * sin llegar a persistir todavía el dataset.
 *
 * Campos enviados:
 *  - file        : archivo CSV/XLSX/Parquet.
 *  - dataset_id  : identificador lógico del dataset (p.ej. «docentes»).
 *  - fmt?        : forzar lector ('csv' | 'xlsx' | 'parquet'), opcional.
 */
export async function validar(
  file: File,
  datasetId: string,
  opts?: { fmt?: "csv" | "xlsx" | "parquet" },
) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("dataset_id", datasetId);
  if (opts?.fmt) {
    fd.append("fmt", opts.fmt);
  }

  const { data } = await api.post<ValidarResp>("/datos/validar", fd);
  return data;
}

/**
 * POST /datos/upload (multipart)
 *
 * Persiste un dataset en el backend bajo un identificador lógico.
 * Esta función es la versión «básica» sin seguimiento de progreso.
 *
 * Campos enviados:
 *  - file         : archivo CSV/XLSX/Parquet.
 *  - periodo      : identificador lógico requerido por el backend.
 *  - dataset_id   : (compatibilidad hacia atrás con versiones previas).
 *  - overwrite    : "true" | "false" (string).
 */
export async function upload(file: File, dataset_id: string, overwrite: boolean) {
  const fd = new FormData();
  fd.append("file", file);
  // El backend actual espera el identificador lógico en el campo «periodo».
  fd.append("periodo", dataset_id);
  // Enviamos también «dataset_id» por compatibilidad con versiones previas.
  fd.append("dataset_id", dataset_id);
  fd.append("overwrite", String(overwrite));

  const { data } = await api.post<UploadResp>("/datos/upload", fd);
  return data;
}

/**
 * Helper de conveniencia para subir datasets desde componentes React.
 *
 * En lugar de pasar parámetros sueltos, acepta un objeto de opciones,
 * lo que simplifica su uso en formularios con múltiples campos.
 */
export async function uploadWithOptions(
  file: File,
  opts?: { dataset_id?: string; overwrite?: boolean },
) {
  const id = (opts?.dataset_id ?? "").trim() || "default";
  const ow = Boolean(opts?.overwrite);
  return upload(file, id, ow);
}

// ---------------------------------------------------------------------------
// Funciones de cliente: resumen de dataset y análisis de sentimientos
// ---------------------------------------------------------------------------

/**
 * GET /datos/resumen
 *
 * Devuelve un resumen estadístico del dataset indicado:
 *  - número de filas y columnas,
 *  - recuento de docentes/asignaturas,
 *  - periodos y rango de fechas,
 *  - resumen por columna.
 *
 * @param params.dataset Identificador lógico del dataset, tal y como
 *                       lo reconoce el backend (por ejemplo «2024-2»).
 */
export async function resumen(params: { dataset: string }) {
  const q = encodeURIComponent(params.dataset);
  const { data } = await api.get<DatasetResumen>(`/datos/resumen?dataset=${q}`);
  return data;
}

/**
 * GET /datos/sentimientos
 *
 * Devuelve la distribución de sentimientos (BETO) asociada al dataset.
 * Se utiliza para alimentar las gráficas de la pestaña Datos.
 *
 * @param params.dataset Identificador lógico del dataset sobre el que
 *                       ya se ejecutó el análisis de sentimientos.
 */
export async function sentimientos(params: { dataset: string }) {
  const q = encodeURIComponent(params.dataset);
  const { data } = await api.get<DatasetSentimientos>(
    `/datos/sentimientos?dataset=${q}`,
  );
  return data;
}

// ---------------------------------------------------------------------------
// Export agrupado para facilitar los imports en componentes React
// ---------------------------------------------------------------------------

const datosService = {
  esquema,
  validar,
  upload,
  uploadWithOptions,
  resumen,
  sentimientos,
};

export default datosService;

// ---------------------------------------------------------------------------
// Subida con seguimiento de progreso usando XMLHttpRequest
// ---------------------------------------------------------------------------

/**
 * Sube un archivo a /datos/upload utilizando XMLHttpRequest en lugar de
 * api.post para poder informar del progreso en tiempo real mediante
 * la función de callback `onProgress`.
 *
 * Mantiene compatibilidad con la API del backend y devuelve un UploadResp.
 */
export async function uploadWithProgress(
  file: File,
  dataset_id: string,
  overwrite: boolean,
  onProgress?: (percent: number) => void,
) {
  const url = "/datos/upload";
  const fd = new FormData();
  fd.append("file", file);
  // El backend actual espera el identificador lógico en «periodo».
  fd.append("periodo", dataset_id);
  // Campo de compatibilidad hacia atrás.
  fd.append("dataset_id", dataset_id);
  fd.append("overwrite", String(overwrite));

  const base =
    (import.meta as any).env?.VITE_API_BASE ??
    (import.meta as any).env?.VITE_API_URL ??
    "http://127.0.0.1:8000";

  const full = `${base}${url}`;

  // Envolvemos XHR en una Promise para poder usar await/try-catch
  // desde los componentes que llamen a esta función.
  const data = await new Promise<UploadResp>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", full, true);

    xhr.upload.onprogress = (evt: ProgressEvent<EventTarget>) => {
      if (evt.lengthComputable && onProgress) {
        const pct = Math.round((evt.loaded / evt.total) * 100);
        onProgress(pct);
      }
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== 4) return;

      const contentType = xhr.getResponseHeader("content-type") || "";

      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const json = contentType.includes("application/json")
            ? JSON.parse(xhr.responseText)
            : null;
          resolve((json || { ok: true }) as UploadResp);
        } catch {
          // Si la respuesta no es JSON, devolvemos un objeto mínimo.
          resolve({ ok: true } as UploadResp);
        }
      } else {
        let msg = `HTTP ${xhr.status} ${xhr.statusText}`;
        try {
          const body = contentType.includes("application/json")
            ? JSON.parse(xhr.responseText)
            : xhr.responseText;
          const detail =
            typeof body === "object" && (body as any)?.detail
              ? (body as any).detail
              : String(body || "");
          msg = `${msg} — ${detail}`;
        } catch {
          // Si falla el parseo del body, nos quedamos con el mensaje base.
        }
        reject(new Error(msg));
      }
    };

    // NOTA: si añades autenticación basada en tokens/cookies,
    // este es el lugar para configurar cabeceras personalizadas.
    // xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(fd);
  });

  return data;
}
