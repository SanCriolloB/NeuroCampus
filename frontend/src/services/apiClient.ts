/**
 * apiClient — wrapper basado en fetch con interfaz tipo axios.
 * -----------------------------------------------------------
 * - BASE configurable con VITE_API_BASE o VITE_API_URL.
 * - Lanza Error enriquecido en respuestas no-2xx (incluye status y cuerpo).
 * - Soporta GET/POST/PUT/PATCH/DELETE.
 * - Body:
 *    • FormData → NO seteamos "Content-Type" (el navegador añade boundary).
 *    • JSON     → "Content-Type: application/json".
 * - Devuelve siempre { data } para compatibilidad "axios-like".
 * - Añadido: X-Correlation-Id (si no lo pasas tú) para trazar peticiones FE↔BE.
 * - Añadido: mensajes claros para 413 (payload demasiado grande).
 */

const RAW_BASE =
  (import.meta as any).env?.VITE_API_BASE ??
  (import.meta as any).env?.VITE_API_URL ??
  "http://127.0.0.1:8000";

/** Normaliza BASE sin trailing slash. */
const BASE = String(RAW_BASE).replace(/\/+$/, "");

type Json = Record<string, unknown>;

/** Construye querystring simple a partir de un objeto plano. */
export function qs(params?: Record<string, string | number | boolean | null | undefined>) {
  if (!params) return "";
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null);
  if (!entries.length) return "";
  const search = entries
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  return search ? `?${search}` : "";
}

/** Extrae mensaje útil desde un cuerpo de error (texto o JSON). */
function extractErrorMessage(raw: string): { message: string; json?: any } {
  if (!raw) return { message: "sin cuerpo" };
  try {
    const obj = JSON.parse(raw);
    // formatos comunes: {detail: "..."} o {error: {code, message}} o {message: "..."}
    const detail =
      (typeof obj.detail === "string" && obj.detail) ||
      (obj.error && (obj.error.message || obj.error.code)) ||
      obj.message ||
      obj.title;
    return { message: typeof detail === "string" ? detail : JSON.stringify(obj), json: obj };
  } catch {
    return { message: raw };
  }
}

/** Obtiene/genera un X-Correlation-Id. */
function correlationIdFrom(init?: RequestInit): string | undefined {
  // Si ya viene en headers, respétalo
  const h = init?.headers as Record<string, string> | undefined;
  const existing =
    (h && (h["X-Correlation-Id"] || h["x-correlation-id"])) ||
    undefined;
  if (existing) return existing;
  // Si el runtime soporta crypto.randomUUID, úsalo
  try {
    // @ts-ignore
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      // @ts-ignore
      return crypto.randomUUID();
    }
  } catch {}
  // Fallback simple (no estrictamente UUIDv4, pero suficiente para trazas locales)
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/** Hace la request y gestiona errores/timeout. Devuelve { data }. */
async function request<T = unknown>(
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
  path: string,
  body?: FormData | Json,
  init?: RequestInit & { timeoutMs?: number }
): Promise<{ data: T }> {
  const isForm = typeof FormData !== "undefined" && body instanceof FormData;

  const controller = new AbortController();
  const timeoutMs = init?.timeoutMs ?? 30000;
  const timer = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : undefined;

  // Construir headers sin mutar los entrantes
  const baseHeaders: Record<string, string> = {};

  // Evitar preflight (OPTIONS) en GET/DELETE sin body:
  // Solo seteamos Content-Type cuando realmente enviamos JSON.
  const shouldSendJsonContentType =
    !isForm &&
    (method === "POST" || method === "PUT" || method === "PATCH" || (method === "DELETE" && body !== undefined));

  if (shouldSendJsonContentType) baseHeaders["Content-Type"] = "application/json";
  // Añadir X-Correlation-Id si no viene en headers
  // X-Correlation-Id:
  // - Si el caller ya lo pasó, lo respetamos.
  // - Para evitar preflight en GET/DELETE sin body, solo lo adjuntamos en:
  //   POST/PUT/PATCH o DELETE con body.
  const initHeaders = (init?.headers || {}) as Record<string, string>;
  const existingCid = initHeaders["X-Correlation-Id"] || initHeaders["x-correlation-id"];

  const shouldAttachCid =
    Boolean(existingCid) ||
    method === "POST" ||
    method === "PUT" ||
    method === "PATCH" ||
    (method === "DELETE" && body !== undefined);

  if (shouldAttachCid) {
    const cid = existingCid || correlationIdFrom(init);
    if (cid) baseHeaders["X-Correlation-Id"] = cid;
  }

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      signal: controller.signal,
      ...init,
      headers: {
        ...baseHeaders,
        ...(init?.headers || {}),
      },
      body:
        method === "GET" || method === "DELETE"
          ? // GET/DELETE típicamente sin body; si hay body JSON en DELETE, lo admitimos:
            (method === "DELETE" && body && !isForm ? JSON.stringify(body) : undefined)
          : isForm
          ? (body as FormData)
          : body !== undefined
          ? JSON.stringify(body)
          : undefined,
      // Keep-Alive puede ayudar a estabilizar peticiones seguidas en local
      keepalive: init?.keepalive ?? false,
    });
  } catch (e: any) {
    const err = new Error(`NETWORK ${e?.name || "Error"} — ${e?.message || "sin detalle"}`);
    // @ts-expect-error info útil
    err.cause = e;
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }

  // Manejo de error HTTP
  if (!res.ok) {
    const raw = await res.text().catch(() => "");
    const { message, json } = extractErrorMessage(raw);
    // Mensaje más claro para 413 (día 7: límite de subida)
    const is413 = res.status === 413;
    const pretty = is413
      ? `El archivo excede el límite permitido por el servidor (HTTP 413). ${message || ""}`.trim()
      : `HTTP ${res.status} ${res.statusText} — ${message}`;

    const err = new Error(pretty);
    // @ts-expect-error adjuntamos datos crudos por si el caller quiere inspeccionarlos
    err.response = { status: res.status, statusText: res.statusText, body: raw, json };
    throw err;
  }

  // Intentamos parsear JSON; si no hay body o no es JSON, devolvemos null
  const ct = res.headers.get("content-type") || "";
  if (!ct) return { data: null as any };
  const isJson = /\bapplication\/json\b/i.test(ct);
  const data: T | null = isJson ? await res.json() : (null as any);

  return { data: data as T };
}

export const apiClient = {
  /** GET JSON */
  get<T = unknown>(path: string, init?: RequestInit & { timeoutMs?: number }) {
    return request<T>("GET", path, undefined, init);
  },
  /** POST (FormData o JSON) */
  post<T = unknown>(path: string, body?: FormData | Json, init?: RequestInit & { timeoutMs?: number }) {
    return request<T>("POST", path, body, init);
  },
  /** PUT (JSON o FormData) — por si lo necesitamos luego */
  put<T = unknown>(path: string, body?: FormData | Json, init?: RequestInit & { timeoutMs?: number }) {
    return request<T>("PUT", path, body, init);
  },
  /** PATCH (JSON o FormData) — por si lo necesitamos luego */
  patch<T = unknown>(path: string, body?: FormData | Json, init?: RequestInit & { timeoutMs?: number }) {
    return request<T>("PATCH", path, body, init);
  },
  /** DELETE (opcionalmente con body JSON) */
  delete<T = unknown>(path: string, body?: Json, init?: RequestInit & { timeoutMs?: number }) {
    return request<T>("DELETE", path, body, init);
  },
};

/**
 * Compatibilidad: alias nombrado `api` y default export.
 * - Permite `import api from "./apiClient"` (default).
 * - Mantiene `import { apiClient } from "./apiClient"` (named).
 */
export const api = apiClient;
export default api;
