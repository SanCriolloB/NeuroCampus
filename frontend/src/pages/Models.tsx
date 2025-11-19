// frontend/src/pages/Models.tsx
// UI de modelos:
// - Entrenamiento rápido vía /modelos/entrenar (RBM general) con curva de pérdida.
// - Lanzar búsqueda de hiperparámetros RBM como job de background.
// - Listar runs desde artifacts/runs y mostrar histórico de entrenamiento con Recharts.

import { useEffect, useMemo, useState } from "react";
import {
  entrenar,
  estado,
  EstadoResp,
  listRuns,
  getRunDetails,
  RunSummary,
  RunDetails,
} from "../services/modelos";
import {
  launchRbmSearch,
  getRbmSearchJob,
  RbmSearchJob,
} from "../services/jobs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function Models() {
  // ---------------------------
  // 1) Entrenamiento rápido RBM
  // ---------------------------

  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [history, setHistory] = useState<EstadoResp["history"]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  // Carga opcional del último job persistido (por si quieres reanudar un estado visible)
  useEffect(() => {
    const last = localStorage.getItem("nc:lastJobId");
    if (last && !jobId) {
      setJobId(last);
      setStatus("resumed");
    }
  }, [jobId]);

  async function onStart() {
    setLoading(true);
    setError("");
    try {
      const res = await entrenar({
        modelo: "rbm_general",
        epochs: 5,
        hparams: {
          n_hidden: 16,
          lr: 0.01,
          batch_size: 64,
          cd_k: 1,
          momentum: 0.5,
          weight_decay: 0.0,
          seed: 42,
        },
      });

      // Guardamos el job para que Dashboard lo lea
      setJobId(res.job_id);
      setStatus(res.status ?? "started");
      try {
        localStorage.setItem("nc:lastJobId", res.job_id);
      } catch {
        // ignorar si el storage falla (modo private, etc.)
      }
      // Limpiamos cualquier histórico previo de otra corrida
      setHistory([]);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  // Polling sencillo cada 1s si hay jobId (entrenamiento rápido)
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const t = setInterval(async () => {
      try {
        const st = await estado(jobId);
        if (cancelled) return;

        setStatus(st.status ?? "unknown");

        // --- Propagar history y, si vienen, errores a nivel raíz ---
        const nextHist: any[] = Array.isArray(st.history) ? [...st.history] : [];
        const rootErr =
          (st as any)?.error || (st as any)?.detail || (st as any)?.message;
        if (rootErr) {
          nextHist.push({
            error: (st as any).error,
            detail: (st as any).detail,
            message: (st as any).message,
          });
        }
        setHistory(nextHist);

        if (
          st.status === "completed" ||
          st.status === "failed" ||
          st.status === "unknown"
        ) {
          clearInterval(t);
        }
      } catch (e: any) {
        // Si el poll falla, no rompemos la UI; detenemos el intervalo si 404/unknown
        setError(e?.message ?? "Error al consultar estado del job");
        clearInterval(t);
      }
    }, 1000);

    return () => {
      clearInterval(t);
      cancelled = true;
    };
  }, [jobId]);

  // Normaliza puntos para el plot, prefiriendo recon_error si existe
  const points = useMemo(() => {
    const list = (history ?? []).map((h: any) => {
      const yVal =
        typeof h?.recon_error === "number"
          ? h.recon_error
          : typeof h?.loss === "number"
          ? h.loss
          : null;
      return yVal == null
        ? null
        : { x: Number(h?.epoch ?? 0), y: Number(yVal) };
    });
    return list.filter(
      (p): p is { x: number; y: number } =>
        !!p && isFinite(p.x) && isFinite(p.y),
    );
  }, [history]);

  // Busca el último mensaje de error en el history (si existe)
  const lastEntry: any =
    Array.isArray(history) && history.length > 0
      ? history[history.length - 1]
      : null;
  const lastErrorMsg: string | null =
    (lastEntry?.error as string) ||
    (lastEntry?.detail as string) ||
    (lastEntry?.message as string) ||
    null;

  // -----------------------------------------
  // 2) Búsqueda de hiperparámetros + runs RBM
  // -----------------------------------------

  const [rbmConfigPath, setRbmConfigPath] = useState<string>("");
  const [rbmJob, setRbmJob] = useState<RbmSearchJob | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunSummary | null>(null);
  const [runDetails, setRunDetails] = useState<RunDetails | null>(null);

  // Cargar runs al entrar
  useEffect(() => {
    listRuns("rbm")
      .then(setRuns)
      .catch(() => setRuns([]));
  }, []);

  // Polling del job de búsqueda RBM
  useEffect(() => {
    if (!rbmJob) return;
    if (rbmJob.status === "done" || rbmJob.status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const job = await getRbmSearchJob(rbmJob.id);
        setRbmJob(job);
        if (job.status === "done") {
          // refrescar lista de runs al terminar
          const newRuns = await listRuns("rbm");
          setRuns(newRuns);
        }
      } catch (err) {
        console.error("Error al refrescar job de RBM:", err);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [rbmJob?.id, rbmJob?.status]);

  const handleLaunchSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const job = await launchRbmSearch(rbmConfigPath || undefined);
      setRbmJob(job);
    } catch (err) {
      console.error(err);
      alert("Error lanzando búsqueda RBM. Revisa backend.");
    }
  };

  const handleSelectRun = async (run: RunSummary) => {
    setSelectedRun(run);
    try {
      const details = await getRunDetails(run.run_id);
      setRunDetails(details);
    } catch (err) {
      console.error(err);
      setRunDetails(null);
    }
  };

  // Transformar history de metrics.json a formato Recharts
  const chartData =
    runDetails && (runDetails.metrics as any).history
      ? ((runDetails.metrics as any).history.epoch as number[]).map(
          (epoch: number, idx: number) => ({
            epoch,
            loss: (runDetails.metrics as any).history.loss?.[idx],
            accuracy: (runDetails.metrics as any).history.accuracy?.[idx],
          }),
        )
      : [];

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold mb-2">
        Entrenamiento y auditoría de modelos (RBM)
      </h1>

      {/* 1) Entrenamiento rápido RBM clásico */}
      <div className="card space-y-3">
        <h2 className="text-lg font-semibold">
          Entrenamiento rápido (endpoint /modelos/entrenar)
        </h2>

        <div className="flex flex-wrap items-center gap-2">
          <button
            className="px-4 py-2 rounded bg-black text-white disabled:opacity-50"
            onClick={onStart}
            disabled={loading}
          >
            {loading ? "Lanzando..." : "Entrenar RBM (5 epochs)"}
          </button>

          {/* Botón para reusar el último job persistido (opcional) */}
          <button
            className="px-3 py-2 rounded border"
            onClick={() => {
              const last = localStorage.getItem("nc:lastJobId");
              if (last) {
                setJobId(last);
                setStatus("resumed");
                setError("");
              }
            }}
          >
            Usar último job guardado
          </button>
        </div>

        {error && (
          <div className="mt-1 text-sm text-red-700">
            <b>Error:</b> {error}
          </div>
        )}

        {jobId && (
          <div className="mt-2 text-sm">
            <div>
              <b>Job:</b> {jobId}
            </div>
            <div>
              <b>Status:</b> {status}
            </div>

            {/* Mostrar texto de error si el job falló */}
            {status === "failed" && lastErrorMsg && (
              <div
                className="mono"
                style={{ color: "#fca5a5", marginTop: 6 }}
              >
                {lastErrorMsg}
              </div>
            )}
          </div>
        )}

        {/* Gráfico simple sin libs externas: SVG line plot */}
        {points.length > 0 && (
          <div className="mt-4">
            <h3 className="font-medium mb-2">
              Curva de pérdida (recon_error / loss)
            </h3>
            <LinePlot data={points} width={600} height={240} />
          </div>
        )}
      </div>

      {/* 2) Búsqueda de hiperparámetros RBM */}
      <div className="card space-y-3">
        <h2 className="text-lg font-semibold">
          Búsqueda de hiperparámetros RBM
        </h2>
        <form
          onSubmit={handleLaunchSearch}
          className="flex flex-wrap items-end gap-2"
        >
          <div className="flex flex-col">
            <label
              className="text-sm font-medium mb-1"
              htmlFor="config-path"
            >
              Config (opcional, por defecto configs/rbm_search.yaml)
            </label>
            <input
              id="config-path"
              type="text"
              value={rbmConfigPath}
              onChange={(e) => setRbmConfigPath(e.target.value)}
              className="border rounded px-2 py-1 text-sm"
              placeholder="ej: configs/rbm_search.yaml"
            />
          </div>
          <button
            type="submit"
            className="btn-primary px-3 py-1 rounded text-sm"
          >
            Lanzar búsqueda RBM
          </button>
        </form>

        {rbmJob && (
          <div className="mt-2 text-sm">
            <p>
              Job actual: <code>{rbmJob.id}</code> · Estado:{" "}
              <strong>{rbmJob.status}</strong>
            </p>
            <p>
              Config: <code>{rbmJob.config_path}</code>
            </p>
            {rbmJob.error && (
              <p className="text-red-600 mt-1">Error: {rbmJob.error}</p>
            )}
          </div>
        )}
      </div>

      {/* 3) Tabla de runs desde artifacts/runs */}
      <div className="card">
        <h2 className="text-md font-semibold mb-2">Runs de RBM</h2>
        {runs.length === 0 ? (
          <p className="text-sm text-gray-500">No hay runs aún.</p>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1">Run</th>
                <th className="text-left py-1">Fecha</th>
                <th className="text-left py-1">Accuracy</th>
                <th className="text-left py-1">F1 macro</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  className="border-b last:border-0 cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSelectRun(run)}
                >
                  <td className="py-1">
                    <code>{run.run_id}</code>
                  </td>
                  <td className="py-1 text-xs">{run.created_at}</td>
                  <td className="py-1">
                    {run.metrics.accuracy ?? "—"}
                  </td>
                  <td className="py-1">
                    {run.metrics.f1_macro ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* 4) Gráfica de histórico del run seleccionado */}
      {selectedRun && runDetails && chartData.length > 0 && (
        <div className="card">
          <h2 className="text-md font-semibold mb-2">
            Histórico de entrenamiento —{" "}
            <code>{selectedRun.run_id}</code>
          </h2>
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="loss"
                  name="Loss"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  name="Accuracy"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// Pequeño componente de línea (SVG) para evitar dependencias en el entrenamiento rápido
function LinePlot({
  data,
  width,
  height,
}: {
  data: { x: number; y: number }[];
  width: number;
  height: number;
}) {
  const pad = 24;
  const xs = data.map((d) => d.x);
  const ys = data.map((d) => d.y);
  const xMin = Math.min(...xs),
    xMax = Math.max(...xs);
  const yMin = Math.min(...ys),
    yMax = Math.max(...ys);

  const safeDX = Math.max(1, xMax - xMin);
  const safeDY = Math.max(1e-9, yMax - yMin);

  const pts = data
    .map((d) => {
      const x = pad + ((d.x - xMin) / safeDX) * (width - pad * 2);
      const y =
        height - pad - ((d.y - yMin) / safeDY) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="border rounded">
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        points={pts}
      />
      {/* ejes mínimos */}
      <line
        x1={pad}
        y1={height - pad}
        x2={width - pad}
        y2={height - pad}
        stroke="currentColor"
      />
      <line
        x1={pad}
        y1={pad}
        x2={pad}
        y2={height - pad}
        stroke="currentColor"
      />
    </svg>
  );
}
