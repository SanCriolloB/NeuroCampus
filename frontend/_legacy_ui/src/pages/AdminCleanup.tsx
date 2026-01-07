// frontend/src/pages/AdminCleanup.tsx
import React, { useEffect, useMemo, useState } from "react";
import { getInventory, postCleanup, getLogs, Candidate } from "../services/adminCleanup";

function human(bytes: number) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let s = bytes;
  let i = 0;
  while (s >= 1024 && i < units.length - 1) {
    s /= 1024;
    i++;
  }
  return `${s.toFixed(2)} ${units[i]}`;
}

export default function AdminCleanupPage() {
  // Parámetros
  const [retentionDays, setRetentionDays] = useState<number>(90);
  const [keepLast, setKeepLast] = useState<number>(3);
  const [excludeGlobs, setExcludeGlobs] = useState<string>("artifacts/champions/**");

  // Token admin (no hardcode; se guarda en localStorage)
  const [token, setToken] = useState<string>(() => localStorage.getItem("NC_ADMIN_TOKEN") || "");

  // Estado
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<null | {
    summary: any;
    candidates: Candidate[];
    dry_run: boolean;
    force: boolean;
    moved_bytes: number;
  }>(null);
  const [logs, setLogs] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Guardar token en localStorage
  useEffect(() => {
    localStorage.setItem("NC_ADMIN_TOKEN", token || "");
  }, [token]);

  const loadInventory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getInventory({
        retention_days: retentionDays,
        keep_last: keepLast,
        exclude_globs: excludeGlobs || undefined,
      });
      setResult({
        summary: data.summary,
        candidates: data.candidates,
        dry_run: data.dry_run,
        force: data.force,
        moved_bytes: data.moved_bytes,
      });
    } catch (e: any) {
      setError(e?.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  const runCleanup = async () => {
    if (!confirm("¿Mover a papelera los candidatos seleccionados por parámetros actuales?")) return;
    setLoading(true);
    setError(null);
    try {
      const data = await postCleanup({
        retention_days: retentionDays,
        keep_last: keepLast,
        exclude_globs: excludeGlobs || undefined,
        dry_run: false,
        force: true,
      });
      setResult({
        summary: data.summary,
        candidates: data.candidates,
        dry_run: data.dry_run,
        force: data.force,
        moved_bytes: data.moved_bytes,
      });
      alert(`Movidos a papelera: ${human(data.moved_bytes)}`);
    } catch (e: any) {
      setError(e?.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  const loadLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getLogs(200);
      setLogs(data.lines);
    } catch (e: any) {
      setError(e?.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  const candidatesTotalSize = useMemo(() => {
    if (!result?.candidates) return 0;
    return result.candidates.reduce((acc, c) => acc + c.size, 0);
  }, [result]);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Administración · Limpieza de artefactos</h1>

      <section className="grid gap-4 md:grid-cols-2">
        <div className="p-4 rounded-xl border">
          <h2 className="font-semibold mb-2">Token admin</h2>
          <input
            className="w-full border rounded p-2"
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Pegue su NC_ADMIN_TOKEN"
          />
          <p className="text-xs text-gray-500 mt-1">
            El token se guarda localmente en tu navegador; no se envia hasta presionar un botón.
          </p>
        </div>
        <div className="p-4 rounded-xl border">
          <h2 className="font-semibold mb-2">Parámetros</h2>
          <div className="grid grid-cols-2 gap-2">
            <label className="text-sm">Retención (días)</label>
            <input className="border rounded p-2" type="number" value={retentionDays}
                   onChange={(e) => setRetentionDays(parseInt(e.target.value || "0", 10))} />
            <label className="text-sm">Keep last</label>
            <input className="border rounded p-2" type="number" value={keepLast}
                   onChange={(e) => setKeepLast(parseInt(e.target.value || "0", 10))} />
            <label className="text-sm">Exclude globs</label>
            <input className="border rounded p-2 col-span-1 md:col-span-1"
                   value={excludeGlobs} onChange={(e) => setExcludeGlobs(e.target.value)} />
          </div>
          <div className="mt-3 flex gap-2">
            <button className="px-3 py-2 rounded bg-gray-100 border" disabled={loading} onClick={loadInventory}>
              {loading ? "Cargando…" : "Inventario (dry-run)"}
            </button>
            <button className="px-3 py-2 rounded bg-red-600 text-white" disabled={loading} onClick={runCleanup}>
              {loading ? "Ejecutando…" : "Mover a papelera (force)"}
            </button>
            <button className="px-3 py-2 rounded bg-gray-100 border" disabled={loading} onClick={loadLogs}>
              {loading ? "Cargando…" : "Ver logs"}
            </button>
          </div>
        </div>
      </section>

      {error && <div className="p-3 bg-red-50 border text-red-700 rounded">{error}</div>}

      {result && (
        <section className="p-4 rounded-xl border">
          <h2 className="font-semibold mb-2">Resumen</h2>
          <ul className="text-sm grid grid-cols-1 md:grid-cols-2 gap-1">
            <li>Total archivos: {result.summary.total_files}</li>
            <li>Total tamaño: {human(result.summary.total_size_bytes)}</li>
            <li>Candidatos: {result.summary.candidates_count}</li>
            <li>Tamaño candidatos: {human(result.summary.candidates_size_bytes)}</li>
            <li>Bytes movidos (última acción): {human(result.moved_bytes || 0)}</li>
          </ul>
        </section>
      )}

      {result?.candidates && (
        <section className="p-4 rounded-xl border">
          <h2 className="font-semibold mb-2">
            Candidatos ({result.candidates.length}) · Total {human(candidatesTotalSize)}
          </h2>
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-2 pr-4">Ruta</th>
                  <th className="py-2 pr-4">Tamaño</th>
                  <th className="py-2 pr-4">Edad (d)</th>
                  <th className="py-2 pr-4">Motivo</th>
                </tr>
              </thead>
              <tbody>
                {result.candidates.map((c) => (
                  <tr key={c.path} className="border-b">
                    <td className="py-1 pr-4 font-mono">{c.path}</td>
                    <td className="py-1 pr-4">{human(c.size)}</td>
                    <td className="py-1 pr-4">{c.age_days.toFixed(1)}</td>
                    <td className="py-1 pr-4">{c.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {logs && (
        <section className="p-4 rounded-xl border">
          <h2 className="font-semibold mb-2">Logs (últimas {logs.length} líneas)</h2>
          <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto">
{logs.join("\n")}
          </pre>
        </section>
      )}
    </div>
  );
}
