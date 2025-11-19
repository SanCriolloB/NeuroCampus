// frontend/src/pages/Jobs.tsx
import { useEffect, useState } from "react";
import api from "../services/apiClient";
import {
  BetoPreprocJob,
  JobStatus,
  launchBetoPreproc,
  getBetoJob,
  listBetoJobs,
} from "../services/jobs";

type JobsPingResp = { jobs: string } | { status: string } | Record<string, unknown>;

function statusLabel(status: JobStatus): string {
  switch (status) {
    case "created":
      return "Creado";
    case "running":
      return "En ejecución";
    case "done":
      return "Completado";
    case "failed":
      return "Fallido";
    default:
      return status;
  }
}

export default function Jobs() {
  const [pong, setPong] = useState<string>("…");
  const [dataset, setDataset] = useState<string>("evaluaciones_2025");
  const [currentJob, setCurrentJob] = useState<BetoPreprocJob | null>(null);
  const [recentJobs, setRecentJobs] = useState<BetoPreprocJob[]>([]);
  const [isLaunching, setIsLaunching] = useState(false);

  // --- Ping inicial para verificar /jobs/ping (como antes) ---
  useEffect(() => {
    api
      .get<JobsPingResp>("/jobs/ping")
      .then(({ data }) => {
        if (typeof (data as any).jobs === "string") {
          setPong(String((data as any).jobs));
        } else if (typeof (data as any).status === "string") {
          setPong(String((data as any).status));
        } else {
          setPong("ok");
        }
      })
      .catch(() => setPong("error"));
  }, []);

  // --- Cargar jobs recientes al entrar ---
  useEffect(() => {
    listBetoJobs()
      .then(setRecentJobs)
      .catch(() => setRecentJobs([]));
  }, []);

  // --- Polling del job actual mientras esté en running/created ---
  useEffect(() => {
    if (!currentJob) return;
    if (currentJob.status === "done" || currentJob.status === "failed") return;

    const interval = setInterval(() => {
      getBetoJob(currentJob.id)
        .then((job) => setCurrentJob(job))
        .catch(() => {
          // si no se puede leer, paramos el polling
          clearInterval(interval);
        });
    }, 3000);

    return () => clearInterval(interval);
  }, [currentJob]);

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!dataset.trim()) return;
    setIsLaunching(true);
    try {
      const job = await launchBetoPreproc({ dataset: dataset.trim() });
      setCurrentJob(job);
      // refrescar lista de jobs recientes
      const jobs = await listBetoJobs();
      setRecentJobs(jobs);
    } catch (err) {
      console.error(err);
      alert("Error lanzando preprocesamiento BETO. Revisa la consola/backend.");
    } finally {
      setIsLaunching(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Tarjeta de ping (igual que antes, solo que un poco contextualizada) */}
      <div className="card">
        <div className="badge">/jobs/ping → {pong}</div>
        <p className="mt-2 text-sm">
          Router de jobs operativo. Aquí gestionamos el preprocesamiento BETO y, más adelante, otros
          jobs de entrenamiento.
        </p>
      </div>

      {/* Panel para lanzar preprocesamiento BETO */}
      <div className="card">
        <h2 className="text-lg font-semibold mb-2">Preprocesamiento BETO (offline)</h2>
        <form onSubmit={handleLaunch} className="flex flex-wrap items-end gap-2">
          <div className="flex flex-col">
            <label className="text-sm font-medium mb-1" htmlFor="dataset">
              Dataset procesado (data/processed/{"{dataset}"}.parquet)
            </label>
            <input
              id="dataset"
              type="text"
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="border rounded px-2 py-1 text-sm"
              placeholder="ej: evaluaciones_2025"
            />
          </div>
          <button
            type="submit"
            className="btn-primary px-3 py-1 rounded text-sm"
            disabled={isLaunching}
          >
            {isLaunching ? "Lanzando…" : "Lanzar BETO"}
          </button>
        </form>

        {currentJob && (
          <div className="mt-4 border-t pt-3 text-sm">
            <div className="flex items-center justify-between mb-1">
              <span className="font-medium">
                Job actual: <code>{currentJob.id}</code>
              </span>
              <span className="badge">{statusLabel(currentJob.status)}</span>
            </div>
            <p>
              Dataset: <code>{currentJob.dataset}</code>
            </p>
            <p>
              Entrada: <code>{currentJob.src}</code>
            </p>
            <p>
              Salida: <code>{currentJob.dst}</code>
            </p>

            {currentJob.meta && (
              <div className="mt-2">
                <p>
                  Modelo BETO: <code>{currentJob.meta.model}</code>
                </p>
                <p>
                  Filas totales: <strong>{currentJob.meta.n_rows}</strong> · Aceptadas por teacher:{" "}
                  <strong>{currentJob.meta.accepted_count}</strong>
                </p>
                <p>
                  Cobertura de texto:{" "}
                  <strong>{(currentJob.meta.text_coverage * 100).toFixed(1)}%</strong>
                </p>
                <p>
                  Umbrales → threshold: {currentJob.meta.threshold} · margin: {currentJob.meta.margin}{" "}
                  · neu_min: {currentJob.meta.neu_min}
                </p>
              </div>
            )}

            {currentJob.error && (
              <p className="mt-2 text-red-600">Error: {currentJob.error}</p>
            )}
          </div>
        )}
      </div>

      {/* Tabla simple de jobs recientes */}
      <div className="card">
        <h3 className="text-md font-semibold mb-2">Jobs recientes (BETO)</h3>
        {recentJobs.length === 0 ? (
          <p className="text-sm text-gray-500">No hay jobs registrados todavía.</p>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1">Job</th>
                <th className="text-left py-1">Dataset</th>
                <th className="text-left py-1">Estado</th>
                <th className="text-left py-1">Creado</th>
                <th className="text-left py-1">Filas</th>
                <th className="text-left py-1">Aceptadas</th>
              </tr>
            </thead>
            <tbody>
              {recentJobs.map((job) => (
                <tr key={job.id} className="border-b last:border-0">
                  <td className="py-1">
                    <code>{job.id}</code>
                  </td>
                  <td className="py-1">{job.dataset}</td>
                  <td className="py-1">{statusLabel(job.status)}</td>
                  <td className="py-1 text-xs">{job.created_at}</td>
                  <td className="py-1">
                    {job.meta ? job.meta.n_rows : "—"}
                  </td>
                  <td className="py-1">
                    {job.meta ? job.meta.accepted_count : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
