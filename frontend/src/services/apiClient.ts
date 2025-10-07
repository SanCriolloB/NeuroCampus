/**
 * apiClient — Cliente de fetch básico.
 * - BASE se puede configurar con VITE_API_URL (ej. http://127.0.0.1:8000)
 * - Maneja errores HTTP lanzando excepción (para doc y pruebas).
 */
const BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export async function api(path: string, init?: RequestInit){
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    // Lanzamos error para que el caller maneje estado de error en UI
    throw new Error(`HTTP ${res.status}`)
  }
  return res.json()
}