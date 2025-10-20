// frontend/src/services/datos.ts
// Cliente de API para el flujo de Datos: esquema, validar y subir dataset.
// Corrige: envío de dataset_id y overwrite en /datos/upload.

import api from "./apiClient";

/** Respuesta flexible de /datos/esquema (soporta distintas formas) */
export type EsquemaResp = {
  version?: string;
  /** Lista simple de columnas requeridas */
  required?: string[];
  /** Lista simple de columnas opcionales */
  optional?: string[];
  /** Alternativa: objetos detallados por columna (si el backend los expone) */
  fields?: Array<{
    name: string;
    dtype?: string | null;
    required?: boolean;
    desc?: string | null;
    domain?: unknown;
    range?: unknown;
    min_len?: number | null;
    max_len?: number | null;
  }>;
  /** Ejemplos o metadatos adicionales */
  examples?: Record<string, unknown>;
};

/** Respuesta de /datos/validar */
export type ValidarResp = {
  ok: boolean;
  missing?: string[];
  extra?: string[];
  sample?: Array<Record<string, unknown>>;
  message?: string;
};

/** Respuesta de /datos/upload */
export type UploadResp = {
  ok: boolean;
  dataset_id?: string;
  rows_ingested?: number;
  stored_as?: string;
  message?: string;
};

/** GET /datos/esquema */
export async function esquema() {
  const { data } = await api.get<EsquemaResp>("/datos/esquema");
  return data;
}

/** POST /datos/validar (multipart: file) */
export async function validar(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await api.post<ValidarResp>("/datos/validar", fd);
  return data;
}

/**
 * POST /datos/upload (multipart)
 * Envía:
 *  - file         : archivo CSV/XLSX/Parquet
 *  - dataset_id   : identificador lógico (p.ej. "2023-2")
 *  - overwrite    : "true" | "false" (string)
 *
 * Devuelve UploadResp con ok, rows_ingested, stored_as y message.
 */
export async function upload(
  file: File,
  dataset_id: string,
  overwrite: boolean
) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("dataset_id", dataset_id);
  fd.append("overwrite", String(overwrite));

  const { data } = await api.post<UploadResp>("/datos/upload", fd);
  return data;
}

/**
 * Alternativa conveniente por opciones (mantiene compatibilidad hacia atrás si tu código
 * llamaba upload(file) sin más). Ejemplo:
 *   uploadWithOptions(file, { dataset_id: "2023-2", overwrite: true })
 */
export async function uploadWithOptions(
  file: File,
  opts?: { dataset_id?: string; overwrite?: boolean }
) {
  const id = (opts?.dataset_id ?? "").trim() || "default";
  const ow = Boolean(opts?.overwrite);
  return upload(file, id, ow);
}

export default {
  esquema,
  validar,
  upload,
  uploadWithOptions,
};
