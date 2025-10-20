// frontend/src/pages/Models.tsx
// UI mínima: lanzar entrenamiento y graficar history[].loss (o recon_error) por epoch.
// - Persiste el último job en localStorage bajo la clave "nc:lastJobId" para que otros módulos (Dashboard)
//   puedan leerlo y mostrar KPIs.
// - Incluye manejo básico de errores y polling con limpieza adecuada.
// - NUEVO: muestra detalle de error cuando status=failed y propaga error/detail/message desde estado(jobId).

import { useEffect, useMemo, useState } from "react";
import { entrenar, estado, EstadoResp } from "../services/modelos";

export default function Models() {
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

  // Polling sencillo cada 1s si hay jobId
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
        const rootErr = (st as any)?.error || (st as any)?.detail || (st as any)?.message;
        if (rootErr) {
          nextHist.push({ error: (st as any).error, detail: (st as any).detail, message: (st as any).message });
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
    return list.filter((p): p is { x: number; y: number } => !!p && isFinite(p.x) && isFinite(p.y));
  }, [history]);

  // Busca el último mensaje de error en el history (si existe)
  const lastEntry: any = Array.isArray(history) && history.length > 0 ? history[history.length - 1] : null;
  const lastErrorMsg: string | null =
    (lastEntry?.error as string) ||
    (lastEntry?.detail as string) ||
    (lastEntry?.message as string) ||
    null;

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">Entrenamiento de Modelos (RBM)</h1>

      <div className="flex items-center gap-2">
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
        <div className="mt-3 text-sm text-red-700">
          <b>Error:</b> {error}
        </div>
      )}

      {jobId && (
        <div className="mt-4 text-sm">
          <div>
            <b>Job:</b> {jobId}
          </div>
          <div>
            <b>Status:</b> {status}
          </div>

          {/* NUEVO: Mostrar texto de error si el job falló */}
          {status === "failed" && lastErrorMsg && (
            <div className="mono" style={{ color: "#fca5a5", marginTop: 6 }}>
              {lastErrorMsg}
            </div>
          )}
        </div>
      )}

      {/* Gráfico simple sin libs externas: SVG line plot */}
      {points.length > 0 && (
        <div className="mt-6">
          <h2 className="font-medium mb-2">Curva de pérdida (recon_error / loss)</h2>
          <LinePlot data={points} width={600} height={240} />
        </div>
      )}
    </div>
  );
}

// Pequeño componente de línea (SVG) para evitar dependencias
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
      const y = height - pad - ((d.y - yMin) / safeDY) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="border rounded">
      <polyline fill="none" stroke="currentColor" strokeWidth="2" points={pts} />
      {/* ejes mínimos */}
      <line
        x1={pad}
        y1={height - pad}
        x2={width - pad}
        y2={height - pad}
        stroke="currentColor"
      />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="currentColor" />
    </svg>
  );
}
