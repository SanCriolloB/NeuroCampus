// frontend/src/services/datos.ts
// Servicio del dominio /datos para el Día 2.
// Usa apiClient con interfaz axios-like (get/post).
// Contratos: GET /datos/esquema, POST /datos/upload (mock).

import { apiClient } from "./apiClient";

/** Columna declarada en la plantilla del dataset */
export type EsquemaCol = {
  name: string;          // ej: "pregunta_1"
  dtype: string;         // ej: "string" | "number" | "integer"
  required: boolean;     // true si es obligatoria en la plantilla
  description?: string;  // opcional
  domain?: string[];     // opcional si el backend lo expone
  range?: [number, number] | null; // opcional si aplica
  max_len?: number | null;         // opcional si aplica
};

/** Respuesta de GET /datos/esquema */
export type EsquemaResponse = {
  version: string;       // ej: "v0.1.0"
  columns: EsquemaCol[];
};

/** Respuesta de POST /datos/upload (mock Día 2) */
export type UploadResponse = {
  dataset_id: string;    // ej: "2024-2"
  rows_ingested: number; // en Día 2 suele ser 0 (mock)
  stored_as: string;     // ej: "localfs://neurocampus/datasets/2024-2.parquet"
  warnings: string[];    // avisos del backend si los hay
};

/**
 * Obtiene el esquema de la plantilla desde el backend.
 * GET /datos/esquema
 */
export async function getEsquema(): Promise<EsquemaResponse> {
  const { data } = await apiClient.get<EsquemaResponse>("/datos/esquema");
  // Fallbacks defensivos por si el backend devuelve null/undefined
  return {
    version: data?.version ?? "",
    columns: Array.isArray(data?.columns) ? data.columns : [],
  };
}

/**
 * Sube un archivo de dataset con metadatos.
 * POST /datos/upload (multipart/form-data)
 * IMPORTANTE: No fijar Content-Type; el boundary lo maneja el navegador.
 */
export async function uploadDatos(params: {
  file: File;
  periodo: string;       // ej: "2024-2"
  overwrite: boolean;
}): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", params.file);
  form.append("periodo", params.periodo);
  form.append("overwrite", String(params.overwrite));

  const { data } = await apiClient.post<UploadResponse>("/datos/upload", form);
  return {
    dataset_id: data?.dataset_id ?? "",
    rows_ingested: Number.isFinite(data?.rows_ingested) ? data.rows_ingested : 0,
    stored_as: data?.stored_as ?? "",
    warnings: Array.isArray(data?.warnings) ? data.warnings : [],
  };
}
