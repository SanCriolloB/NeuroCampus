/**
 * Cliente API mínimo.
 * 
 * - Usa VITE_API_BASE definido en .env (lado frontend debe iniciar con VITE_*)
 * - Provee un wrapper simple sobre fetch para futuras llamadas.
 */
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function pingDatos(): Promise<{ datos: string }> {
  const res = await fetch(`${API_BASE}/datos/ping`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function pingJobs(): Promise<{ jobs: string }> {
  const res = await fetch(`${API_BASE}/jobs/ping`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export { API_BASE };