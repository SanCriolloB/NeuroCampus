// frontend/src/services/datos.ts
// Cliente de API para el flujo de Datos: esquema, validar y subir dataset.
// ✅ Correcciones clave:
//   - /datos/validar ahora requiere también `dataset_id` (y opcionalmente `fmt`).
//   - /datos/upload exige `periodo` en el body; enviamos también `dataset_id` por compatibilidad.

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

/** Respuesta de /datos/validar (Día 5) */
export type ValidarResp = {
  ok: boolean;
  dataset_id?: string;
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

/**
 * POST /datos/validar (multipart)
 * Envía:
 *  - file        : archivo CSV/XLSX/Parquet
 *  - dataset_id  : identificador lógico del dataset (p.ej. "docentes")
 *  - fmt?        : forzar lector ('csv' | 'xlsx' | 'parquet'), opcional
 */
export async function validar(
  file: File,
  datasetId: string,
  opts?: { fmt?: "csv" | "xlsx" | "parquet" }
) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("dataset_id", datasetId);
  if (opts?.fmt) fd.append("fmt", opts.fmt);

  const { data } = await api.post<ValidarResp>("/datos/validar", fd);
  return data;
}

/**
 * POST /datos/upload (multipart)
 * Envía:
 *  - file         : archivo CSV/XLSX/Parquet
 *  - periodo      : identificador lógico requerido por el backend
 *  - dataset_id   : (compatibilidad hacia atrás con versiones previas)
 *  - overwrite    : "true" | "false" (string)
 */
export async function upload(file: File, dataset_id: string, overwrite: boolean) {
  const fd = new FormData();
  fd.append("file", file);
  // Requerido por el backend actual:
  fd.append("periodo", dataset_id);
  // Compatibilidad con contratos previos:
  fd.append("dataset_id", dataset_id);
  fd.append("overwrite", String(overwrite));

  const { data } = await api.post<UploadResp>("/datos/upload", fd);
  return data;
}

/**
 * Alternativa conveniente por opciones (compatibilidad hacia atrás).
 * Ejemplo:
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

// --- Subida con progreso usando XHR (opcional) ---
// Mantiene compatibilidad con la API de nuestro backend y reporta %.
export async function uploadWithProgress(
  file: File,
  dataset_id: string,
  overwrite: boolean,
  onProgress?: (percent: number) => void
) {
  const url = "/datos/upload";
  const fd = new FormData();
  fd.append("file", file);
  fd.append("periodo", dataset_id);    // backend actual espera 'periodo'
  fd.append("dataset_id", dataset_id); // compat hacia atrás
  fd.append("overwrite", String(overwrite));

  const base =
    (import.meta as any).env?.VITE_API_BASE ??
    (import.meta as any).env?.VITE_API_URL ??
    "http://127.0.0.1:8000";

  const full = `${base}${url}`;

  // Envolvemos XHR en una Promise para usar await/try-catch desde el componente.
  const data = await new Promise<UploadResp>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", full, true);

    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable && onProgress) {
        const pct = Math.round((evt.loaded / evt.total) * 100);
        onProgress(pct);
      }
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== 4) return;
      const ct = xhr.getResponseHeader("content-type") || "";
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const json = ct.includes("application/json") ? JSON.parse(xhr.responseText) : null;
          resolve(json || { ok: true } as any);
        } catch (e) {
          resolve({ ok: true } as any); // fallback benigno si no es JSON
        }
      } else {
        let msg = `HTTP ${xhr.status} ${xhr.statusText}`;
        try {
          const body = ct.includes("application/json") ? JSON.parse(xhr.responseText) : xhr.responseText;
          const detail = (typeof body === "object" && body?.detail) ? body.detail : String(body || "");
          msg = `${msg} — ${detail}`;
        } catch {}
        reject(new Error(msg));
      }
    };

    // NOTA: Si requieres auth headers, añádelos aquí con xhr.setRequestHeader(...)
    xhr.send(fd);
  });

  return data;
}

