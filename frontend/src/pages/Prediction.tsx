import { useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import ResultsTable from "../components/ResultsTable";
import * as pred from "../services/prediccion";

const initial = {
  p1: 5, p2: 5, p3: 5, p4: 5, p5: 5,
  p6: 5, p7: 5, p8: 5, p9: 5, p10: 5,
} as Record<string, number>;

export default function Prediction() {
  const [comentario, setComentario] = useState("");
  const [calif, setCalif] = useState<Record<string, number>>(initial);
  const [onlineRes, setOnlineRes] = useState<pred.PrediccionOnlineResponse | null>(null);
  const [batchRes, setBatchRes] = useState<pred.PrediccionBatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  function setK(k: string, v: number) {
    setCalif((prev) => ({ ...prev, [k]: v }));
  }

  async function onOnline() {
    setLoading(true);
    setError("");
    setBatchRes(null);
    try {
      const data = await pred.online({
        input: { comentario, calificaciones: calif },
      });
      setOnlineRes(data);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onBatch(file: File) {
    setLoading(true);
    setError("");
    setOnlineRes(null);
    try {
      const data = await pred.batch(file);
      setBatchRes(data);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Panel de entrada */}
      <div className="card" style={{ display: "grid", gap: 12 }}>
        <h3 style={{ margin: 0 }}>Predicción online</h3>
        <textarea
          value={comentario}
          onChange={(e) => setComentario(e.target.value)}
          rows={5}
          placeholder="Escribe un comentario…"
          style={{ width: "100%", padding: 8 }}
        />
        <div className="grid" style={{ gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
          {Object.keys(calif).map((k) => (
            <label key={k} className="mono" style={{ fontSize: 12 }}>
              {k.toUpperCase()}
              <input
                type="number"
                min={0}
                max={10}
                value={calif[k]}
                onChange={(e) => setK(k, Number(e.target.value))}
                style={{ width: "100%", padding: 6 }}
              />
            </label>
          ))}
        </div>
        <button onClick={onOnline} disabled={loading} className="badge" style={{ cursor: "pointer" }}>
          {loading ? "Calculando…" : "Predecir"}
        </button>
        {!!error && <div style={{ color: "#b91c1c" }}>{error}</div>}
      </div>

      {/* Panel de resultados */}
      <div className="card" style={{ display: "grid", gap: 12 }}>
        <h3 style={{ margin: 0 }}>Resultados</h3>
        {!onlineRes && !batchRes && <div className="badge">Sin resultados aún</div>}

        {onlineRes && (
          <div>
            <div><b>label_top:</b> {onlineRes.label_top}</div>
            {onlineRes.sentiment && (
              <div>
                <b>sentiment:</b> {onlineRes.sentiment} — <b>confidence:</b>{" "}
                {((onlineRes.confidence ?? 0) * 100).toFixed(1)}%
              </div>
            )}
            <div><b>latency:</b> {onlineRes.latency_ms} ms</div>
            <div><b>correlation_id:</b> <code>{onlineRes.correlation_id}</code></div>
            <pre
              className="mono"
              style={{ background: "#f8fafc", padding: 8, borderRadius: 8, overflow: "auto" }}
            >
{JSON.stringify(onlineRes.scores, null, 2)}
            </pre>
          </div>
        )}

        {batchRes && (
          <div style={{ display: "grid", gap: 12 }}>
            <div className="badge">batch_id: {batchRes.batch_id}</div>
            {Array.isArray(batchRes.sample) && batchRes.sample.length > 0 && (
              <ResultsTable
                columns={Object.keys(batchRes.sample[0]).slice(0, 8).map((k) => ({ key: k, header: k }))}
                rows={batchRes.sample}
              />
            )}
            <div>
              <a href={batchRes.artifact} target="_blank">Descargar artifact</a>
            </div>
            <div className="mono">correlation_id: {batchRes.correlation_id}</div>
          </div>
        )}

        <div style={{ borderTop: "1px solid #e5e7eb", paddingTop: 12 }}>
          <h4 style={{ margin: "8px 0" }}>Batch (CSV/XLSX/Parquet)</h4>
          <UploadDropzone onFileSelected={onBatch} accept=".csv,.xlsx,.xls,.parquet" />
        </div>
      </div>
    </div>
  );
}
