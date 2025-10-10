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
 *
 * Mejora clave (alineado Día 3):
 * - Parseo de errores: intenta JSON; si hay {detail} o {error:{code,message}} lo incorpora al mensaje.
 */

const BASE =
  (import.meta as any).env?.VITE_API_BASE ??
  (import.meta as any).env?.VITE_API_URL ??
  "http://127.0.0.1:8000";

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

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      signal: controller.signal,
      ...init,
      headers: {
        ...(init?.headers || {}),
        ...(isForm ? {} : { "Content-Type": "application/json" }),
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
    const err = new Error(`HTTP ${res.status} ${res.statusText} — ${message}`);
    // @ts-expect-error adjuntamos datos crudos por si el caller quiere inspeccionarlos
    err.response = { status: res.status, statusText: res.statusText, body: raw, json };
    throw err;
  }

  // Intentamos parsear JSON; si no hay body o no es JSON, devolvemos null
  const ct = res.headers.get("content-type") || "";
  const data: T | null = ct.includes("application/json") ? await res.json() : (null as any);

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
