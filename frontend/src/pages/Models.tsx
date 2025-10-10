// frontend/src/pages/Models.tsx
// UI mínima: lanzar entrenamiento y graficar history[].loss (epoch).
import { useEffect, useMemo, useState } from "react";
import { entrenar, estado, EstadoResp } from "../services/modelos";

export default function Models() {
  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [history, setHistory] = useState<EstadoResp["history"]>([]);
  const [loading, setLoading] = useState(false);

  async function onStart() {
    setLoading(true);
    try {
      const res = await entrenar({
        modelo: "rbm_general",
        epochs: 5,
        hparams: { n_hidden: 16, lr: 0.01, batch_size: 64, cd_k: 1, momentum: 0.5, weight_decay: 0.0, seed: 42 }
      });
      setJobId(res.job_id);
      setStatus(res.status);
    } finally {
      setLoading(false);
    }
  }

  // Polling sencillo cada 1s si hay jobId
  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(async () => {
      const st = await estado(jobId);
      setStatus(st.status);
      setHistory(st.history ?? []);
      if (st.status === "completed" || st.status === "failed" || st.status === "unknown") {
        clearInterval(t);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [jobId]);

  const points = useMemo(
    () => history.map((h) => ({ x: h.epoch, y: h.recon_error ?? h.loss })),
    [history]
  );

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">Entrenamiento de Modelos (RBM)</h1>

      <button
        className="px-4 py-2 rounded bg-black text-white disabled:opacity-50"
        onClick={onStart}
        disabled={loading}
      >
        {loading ? "Lanzando..." : "Entrenar RBM (5 epochs)"}
      </button>

      {jobId && (
        <div className="mt-4 text-sm">
          <div><b>Job:</b> {jobId}</div>
          <div><b>Status:</b> {status}</div>
        </div>
      )}

      {/* Gráfico simple sin libs externas: SVG line plot */}
      {points.length > 0 && (
        <div className="mt-6">
          <h2 className="font-medium mb-2">Curva de pérdida (recon_error)</h2>
          <LinePlot data={points} width={600} height={240} />
        </div>
      )}
    </div>
  );
}

// Pequeño componente de línea (SVG) para evitar dependencias
function LinePlot({ data, width, height }: { data: { x: number; y: number }[]; width: number; height: number }) {
  const pad = 24;
  const xs = data.map((d) => d.x);
  const ys = data.map((d) => d.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);

  const pts = data.map(d => {
    const x = pad + ((d.x - xMin) / Math.max(1, xMax - xMin)) * (width - pad * 2);
    const y = height - pad - ((d.y - yMin) / Math.max(1e-9, yMax - yMin)) * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width={width} height={height} className="border rounded">
      <polyline fill="none" stroke="currentColor" strokeWidth="2" points={pts} />
      {/* ejes mínimos */}
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="currentColor" />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="currentColor" />
    </svg>
  );
}
