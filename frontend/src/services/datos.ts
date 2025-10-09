// frontend/src/services/datos.ts
// ---------------------------------------------------------------
// Servicio del dominio /datos (Día 2 → extendido para v0.3.0).
// - Estándar de cliente: apiClient axios-like (get/post).
// - Contratos cubiertos:
//     • GET  /datos/esquema          → Esquema de columnas de la plantilla.
//     • POST /datos/upload           → Carga (mock) del dataset.
//     • POST /datos/validar          → Validación sin persistir (nuevo Día 3).
// - Notas clave:
//     • No forzar "Content-Type" cuando se envía FormData (el navegador
//       agrega el boundary automáticamente).
//     • Fallbacks defensivos ante respuestas parciales o nulas del backend.
//     • Tipos enriquecidos para documentar domain/range/pattern y issues.
//
// Este archivo sirve como fuente de verdad para el Frontend y como apoyo
// para la documentación técnica del proyecto.
//
// ---------------------------------------------------------------

import { apiClient } from "./apiClient";

/** Tipo de dato esperado en JSON Schema.
 *  Puede ser un string (p.ej. "string") o una lista (p.ej. ["string","null"]).
 *  Esto permite expresar valores nulos sin perder la compatibilidad con el validador.
 */
export type DType = string | string[];

/** Dominio permitido para una columna:
 *  - allowed: lista explícita de valores válidos (para string/boolean/enum).
 *  - min/max: rango numérico permitido (para integer/number).
 */
export type DomainSpec = {
  allowed?: (string | number | boolean | null)[];
  min?: number;
  max?: number;
};

/** Columna declarada en la plantilla del dataset (respuesta de /datos/esquema). */
export type EsquemaCol = {
  /** Nombre normalizado de la columna (snake_case, sin acentos/“:”) */
  name: string;              // ej: "pregunta_1"

  /** Tipo esperado: string | number | integer | boolean | date | ... | ["string","null"] */
  dtype: DType;

  /** Si la columna es requerida en la plantilla */
  required: boolean;

  /** Descripción opcional para documentación/UI */
  description?: string;

  /** Dominio permitido (enum o rango) — opcional */
  domain?: DomainSpec;

  /** Rango compacto por compatibilidad (ej. [0,50]); preferir "domain.min/max" */
  range?: [number, number] | null;

  /** Longitud máxima de cadenas (si aplica) */
  max_len?: number | null;

  /** Patrón (regex) que debe cumplir el valor — JSON Schema: pattern (opcional) */
  pattern?: string | null;
};

/** Respuesta de GET /datos/esquema */
export type EsquemaResponse = {
  /** Versión del contrato de esquema (ej. "v0.3.0") */
  version: string;
  /** Columnas esperadas con metadatos y restricciones */
  columns: EsquemaCol[];
};

/** Respuesta de POST /datos/upload (mock Día 2) */
export type UploadResponse = {
  /** Identificador lógico del dataset cargado (p.ej. periodo) */
  dataset_id: string;                 // ej: "2024-2"

  /** Filas ingeridas; en mock puede ser 0 */
  rows_ingested: number;              // en Día 2 suele ser 0

  /** Ruta lógica de almacenamiento (s3://..., localfs://..., etc.) */
  stored_as: string;                  // ej: "localfs://neurocampus/datasets/2024-2.parquet"

  /** Advertencias generadas por el backend (si las hay) */
  warnings: string[];
};

/** Códigos de issues esperados en la validación de datasets (Día 3). */
export type IssueCode =
  | "MISSING_COLUMN"
  | "BAD_TYPE"
  | "DOMAIN_VIOLATION"
  | "RANGE_VIOLATION"
  | "DUPLICATE_ROW"
  | "HIGH_NULL_RATIO"
  | "PATTERN_MISMATCH"
  | string; // para permitir extensiones futuras sin romper el tipado

/** Severidad del issue. */
export type IssueSeverity = "error" | "warning";

/** Detalle de un issue de validación. */
export type ValidacionIssue = {
  code: IssueCode;                 // p.ej. "BAD_TYPE"
  severity: IssueSeverity;         // "error" | "warning"
  column?: string | null;          // nombre de columna, si aplica
  row?: number | null;             // número de fila (1-based o 0-based según backend), si aplica
  message: string;                 // descripción legible
};

/** Resumen de validación (totales). */
export type ValidacionSummary = {
  rows: number;                    // total de filas detectadas
  errors: number;                  // cantidad de issues con severity=error
  warnings: number;                // cantidad de issues con severity=warning
  engine: "pandas" | "polars" | string; // motor usado en el backend
};

/** Respuesta de POST /datos/validar */
export type ValidacionResponse = {
  summary: ValidacionSummary;
  issues: ValidacionIssue[];
};

/* ========================================================================
 *  GET /datos/esquema
 *  Obtiene el esquema de la plantilla desde el backend.
 * ====================================================================== */
export async function getEsquema(): Promise<EsquemaResponse> {
  // apiClient.get<T>() devuelve { data: T }
  const { data } = await apiClient.get<EsquemaResponse>("/datos/esquema");

  // Fallbacks defensivos por si el backend devuelve null/undefined
  return {
    version: data?.version ?? "",
    columns: Array.isArray(data?.columns) ? data.columns : [],
  };
}

/* ========================================================================
 *  POST /datos/upload
 *  Sube un archivo de dataset con metadatos (multipart/form-data).
 *
 *  IMPORTANTE:
 *    - No fijar "Content-Type"; al usar FormData el navegador agrega el boundary.
 *    - El backend (mock Día 2) puede responder rows_ingested=0.
 * ====================================================================== */
export async function uploadDatos(params: {
  file: File;               // archivo CSV/XLSX/Parquet (según soporte)
  periodo: string;          // ej: "2024-2"
  overwrite: boolean;       // si true, sobrescribe dataset existente
}): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", params.file);
  form.append("periodo", params.periodo);
  form.append("overwrite", String(params.overwrite));

  const { data } = await apiClient.post<UploadResponse>("/datos/upload", form);

  // Normalizamos por si el backend devuelve valores parciales
  return {
    dataset_id: data?.dataset_id ?? "",
    rows_ingested: Number.isFinite(data?.rows_ingested) ? data.rows_ingested : 0,
    stored_as: data?.stored_as ?? "",
    warnings: Array.isArray(data?.warnings) ? data.warnings : [],
  };
}

/* ========================================================================
 *  POST /datos/validar  (NUEVO — Día 3)
 *  Valida un archivo sin persistirlo: revisa esquema, tipos, dominios,
 *  patrones (regex), duplicados y calidad, aplicando normalización y
 *  coerción de tipos previas según el backend.
 *
 *  Parámetros:
 *    - file: CSV/XLSX/Parquet (según soporte).
 *    - fmt:  "csv" | "xlsx" | "parquet" (opcional, si el backend lo requiere).
 *
 *  Devuelve:
 *    - summary: totales (rows, errors, warnings, engine).
 *    - issues : lista detallada de incidencias (código, severidad, columna/fila, mensaje).
 *
 *  Nota:
 *    - La UI puede usar 'issues' para resaltar columnas/filas problemáticas,
 *      y 'summary' para mostrar totales en cards/badges.
 * ====================================================================== */
export async function validarDatos(file: File, fmt?: "csv" | "xlsx" | "parquet") {
  const form = new FormData();
  form.append("file", file);
  if (fmt) form.append("fmt", fmt);
  // O, si prefieres: intenta inferir por extensión antes de agregar fmt.

  // NO forzar Content-Type; el navegador añade el boundary:
  const { data } = await apiClient.post<ValidacionResponse>("/datos/validar", form);

  // Fallbacks defensivos (conservar tu enfoque):
  return {
    summary: data?.summary ?? { rows: 0, errors: 0, warnings: 0, engine: "" },
    issues: Array.isArray(data?.issues) ? data.issues : [],
  };
}

/* ========================================================================
 *  Helpers opcionales para la UI (no obligatorios)
 *  - Son utilidades comunes que la UI puede reutilizar para presentar tipos.
 * ====================================================================== */

/** Devuelve true si el dtype declarado admite null (["string","null"], etc.). */
export function dtypeAdmiteNull(dtype: DType): boolean {
  return Array.isArray(dtype) ? dtype.map(d => d.toLowerCase()).includes("null") : false;
}

/** Devuelve el dtype "principal" para mostrar en tabla (si es lista, toma el primero distinto de "null"). */
export function dtypePrincipal(dtype: DType): string {
  if (Array.isArray(dtype)) {
    const firstNonNull = dtype.find(d => d.toLowerCase() !== "null");
    return firstNonNull ?? (dtype[0] ?? "");
  }
  return dtype;
}

/** Devuelve una descripción corta de restricciones para UI (domain, range, pattern). */
export function describeRestricciones(col: EsquemaCol): string {
  const parts: string[] = [];
  if (col.domain?.allowed && col.domain.allowed.length > 0) {
    parts.push(`allowed: ${col.domain.allowed.join(", ")}`);
  }
  if (typeof col.domain?.min === "number" || typeof col.domain?.max === "number") {
    const min = col.domain?.min ?? "";
    const max = col.domain?.max ?? "";
    parts.push(`range: [${min}, ${max}]`);
  }
  if (Array.isArray(col.range) && col.range.length === 2) {
    parts.push(`range: [${col.range[0]}, ${col.range[1]}]`);
  }
  if (typeof col.max_len === "number") {
    parts.push(`max_len: ${col.max_len}`);
  }
  if (col.pattern) {
    parts.push(`pattern: ${col.pattern}`);
  }
  return parts.join(" · ");
}

// Tipos aceptados
export type ArchivoFormato = "csv" | "xlsx" | "parquet";

// ✅ Asegúrate de exportarla como *named export*
export function inferFormatFromFilename(name?: string): ArchivoFormato | undefined {
  if (!name) return undefined;
  const ext = name.split(".").pop()?.toLowerCase();
  return ext === "csv" || ext === "xlsx" || ext === "parquet" ? (ext as ArchivoFormato) : undefined;
}
