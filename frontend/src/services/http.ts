// frontend/src/services/http.ts
// Punto único recomendado para consumir el cliente HTTP.
// Re-exporta apiClient existente para estandarizar imports desde Features.

export { apiClient, api } from "./apiClient";
export { qs } from "./apiClient";

// Helper: base URL (solo si lo requieres en logs o para debug)
export function getApiBaseUrl(): string {
  const raw =
    (import.meta as any).env?.VITE_API_BASE ??
    (import.meta as any).env?.VITE_API_URL ??
    "http://127.0.0.1:8000";
  return String(raw).replace(/\/+$/, "");
}
