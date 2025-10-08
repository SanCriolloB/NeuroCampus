/**
 * apiClient — wrapper basado en fetch con interfaz tipo axios.
 * - BASE configurable con VITE_API_BASE o VITE_API_URL.
 * - Lanza error en respuestas no-2xx.
 * - Soporta GET JSON y POST (FormData o JSON).
 * - Devuelve { data } para que el caller use res.data (compat axios-like).
 */

const BASE =
  // permite cualquiera de las dos variables, priorizando la usada en Día 2
  (import.meta as any).env?.VITE_API_BASE ??
  (import.meta as any).env?.VITE_API_URL ??
  "http://127.0.0.1:8000";

type Json = Record<string, unknown>;

async function request(
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
  path: string,
  body?: FormData | Json,
  init?: RequestInit
) {
  const isForm = typeof FormData !== "undefined" && body instanceof FormData;

  // Si es FormData, NO seteamos 'Content-Type' (el navegador agrega el boundary).
  const headers: HeadersInit = {
    ...(init?.headers || {}),
    ...(isForm ? {} : { "Content-Type": "application/json" }),
  };

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body
      ? isForm
        ? (body as FormData)
        : JSON.stringify(body as Json)
      : undefined,
    ...init,
  });

  // Manejo de error HTTP
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const err = new Error(
      `HTTP ${res.status} ${res.statusText} — ${text || "sin cuerpo"}`
    );
    // @ts-expect-error: adjuntamos info útil para debug
    err.status = res.status;
    // @ts-expect-error
    err.body = text;
    throw err;
  }

  // Intentamos parsear JSON; si no hay body, devolvemos null
  const data = res.headers.get("content-type")?.includes("application/json")
    ? await res.json()
    : null;

  // interfaz tipo axios
  return { data };
}

export const apiClient = {
  get<T = unknown>(path: string, init?: RequestInit) {
    return request("GET", path, undefined, init) as Promise<{ data: T }>;
  },
  post<T = unknown>(
    path: string,
    body?: FormData | Json,
    init?: RequestInit
  ) {
    // Si nos pasan 'multipart/form-data' en headers, lo ignoramos si body es FormData
    if (body instanceof FormData && init?.headers) {
      const h = new Headers(init.headers);
      if (h.has("Content-Type")) h.delete("Content-Type");
      init = { ...init, headers: h };
    }
    return request("POST", path, body, init) as Promise<{ data: T }>;
  },
};
