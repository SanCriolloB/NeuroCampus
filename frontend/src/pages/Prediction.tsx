// frontend/src/pages/Prediction.tsx
// Mostrar champion + gráficas en la página de Predicciones.
// - Tarjeta "Modelo actual (champion)" con métricas clave.
// - Gráfico de F1 por clase (si existe en metrics.f1_per_class).
// - Gráfico de probabilidades de la última predicción online.
// - Mantiene el flujo actual de predicción online y batch.
// (Requiere: npm install recharts)

import { useEffect, useMemo, useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import ResultsTable from "../components/ResultsTable";
import * as pred from "../services/prediccion";
import { getChampion, ChampionInfo } from "../services/modelos";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
} from "recharts";

const initial = {
  p1: 5, p2: 5, p3: 5, p4: 5, p5: 5,
  p6: 5, p7: 5, p8: 5, p9: 5, p10: 5,
} as Record<string, number>;

export default function Prediction() {
  // ---------------------------
  // Estado para predicción
  // ---------------------------
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

  // ---------------------------
  // Champion (modelo actual)
  // ---------------------------
  const [champion, setChampion] = useState<ChampionInfo | null>(null);
  const [championErr, setChampionErr] = useState<string>("");

  useEffect(() => {
    // Intentamos pedir el champion para "rbm" (ajusta si quieres hacerlo genérico)
    getChampion("rbm")
      .then(setChampion)
      .catch((e) => {
        setChampion(null);
        setChampionErr(e?.message ?? "No se pudo cargar el modelo campeón");
      });
  }, []);

  // Datos para el gráfico F1 por clase del champion
  const f1PerClassData = useMemo(() => {
    const metrics = champion?.metrics as any;
    if (metrics && metrics.f1_per_class && typeof metrics.f1_per_class === "object") {
      const entries = Object.entries(metrics.f1_per_class) as [string, number | string][];
      return entries.map(([label, value]) => ({
        label,
        f1: Number(value),
      }));
    }
    // Fallback: si no hay por clase, mostramos barras con métricas globales
    if (metrics) {
      const items: { label: string; value: number }[] = [];
      const pushIfNum = (label: string, v: any) => {
        if (typeof v === "number" && Number.isFinite(v)) items.push({ label, value: v });
      };
      pushIfNum("accuracy", metrics.accuracy);
      pushIfNum("f1_macro", metrics.f1_macro);
      pushIfNum("f1_weighted", metrics.f1_weighted);
      return items.map((d) => ({ label: d.label, f1: d.value }));
    }
    return [];
  }, [champion]);

  // Datos para gráfico de probabilidades de la última predicción online
  const probaData = useMemo(() => {
    if (!onlineRes) return [];
    // Intentamos leer la estructura típica: onlineRes.scores = {neg: 0.1, neu:0.2, pos:0.7} u objeto similar
    const scores: any = onlineRes.scores;
    if (scores && typeof scores === "object" && !Array.isArray(scores)) {
      return Object.entries(scores)
        .filter(([_, v]) => typeof v === "number" && Number.isFinite(v))
        .map(([label, v]) => ({ label, prob: Number(v) }));
    }
    // Fallback: si viene como array [ {label, score} ], lo acomodamos
    if (Array.isArray(scores)) {
      return scores
        .map((it: any) => {
          const label = it?.label ?? it?.class ?? it?.name;
          const prob = it?.score ?? it?.prob ?? it?.value;
          if (typeof label === "string" && typeof prob === "number") {
            return { label, prob: Number(prob) };
          }
          return null;
        })
        .filter(Boolean) as { label: string; prob: number }[];
    }
    return [];
  }, [onlineRes]);

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* ---------------- Champion / Modelo actual (spans both columns) ---------------- */}
      <div className="card" style={{ gridColumn: "1 / -1", display: "grid", gap: 12 }}>
        <h3 style={{ margin: 0 }}>Modelo actual (champion)</h3>
        {championErr && <div className="badge" style={{ background: "#fee2e2", color: "#991b1b" }}>
          {championErr}
        </div>}

        {!champion && !championErr && <div className="badge">Cargando champion…</div>}

        {champion && (
          <>
            <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 12 }}>
              <div className="card" style={{ padding: 8 }}>
                <div className="text-sm">Modelo</div>
                <div className="text-lg font-semibold">{champion.model_name}</div>
              </div>
              <div className="card" style={{ padding: 8 }}>
                <div className="text-sm">Accuracy</div>
                <div className="text-lg font-semibold">
                  {(champion.metrics as any)?.accuracy?.toFixed?.(3) ?? "—"}
                </div>
              </div>
              <div className="card" style={{ padding: 8 }}>
                <div className="text-sm">F1 macro</div>
                <div className="text-lg font-semibold">
                  {(champion.metrics as any)?.f1_macro?.toFixed?.(3) ?? "—"}
                </div>
              </div>
            </div>

            {/* Gráfico F1 por clase (o métricas globales como fallback) */}
            {f1PerClassData.length > 0 && (
              <div>
                <h4 style={{ margin: "8px 0" }}>F1 por clase</h4>
                <div style={{ width: "100%", height: 260 }}>
                  <ResponsiveContainer>
                    <BarChart data={f1PerClassData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="f1" name="F1" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* ---------------- Panel de entrada ---------------- */}
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

      {/* ---------------- Panel de resultados ---------------- */}
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

            {/* Gráfico de probabilidades de la predicción */}
            {probaData.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ margin: "8px 0" }}>Probabilidades por clase</h4>
                <div style={{ width: "100%", height: 220 }}>
                  <ResponsiveContainer>
                    <BarChart data={probaData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="prob" name="Probabilidad" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* JSON bruto de scores (útil para depurar) */}
            <pre
              className="mono"
              style={{ background: "#f8fafc", padding: 8, borderRadius: 8, overflow: "auto", marginTop: 12 }}
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
