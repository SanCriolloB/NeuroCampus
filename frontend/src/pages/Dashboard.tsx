// frontend/src/pages/Dashboard.tsx
import { useEffect, useState } from "react";
import MetricCard from "../components/MetricCard";
import api from "../services/apiClient";
// Si ya tienes EstadoResp exportado desde services/modelos, descomenta esta línea
// import { EstadoResp } from "../services/modelos";

/**
 * Si NO tienes el tipo EstadoResp exportado aún, usa este fallback minimal:
 * (Puedes borrar este bloque si ya importas EstadoResp real)
 */
type EstadoResp = {
  metrics?: {
    accuracy?: number;
    f1_macro?: number;
    f1?: number;
  };
  history?: any[];
};

/** Tipo de la respuesta de /health */
type HealthResp = { status: string };

export default function Dashboard() {
  const [health, setHealth] = useState<string>("unknown");
  const [estado, setEstado] = useState<EstadoResp | null>(null);

  useEffect(() => {
    // ✅ TIPAR la respuesta de /health para que data.status no sea "unknown"
    api
      .get<HealthResp>("/health")
      .then(({ data }) => setHealth(data?.status ?? "unknown"))
      .catch(() => setHealth("down"));

    const lastJobId = localStorage.getItem("nc:lastJobId");
    if (lastJobId) {
      // ✅ TIPAR la respuesta de /modelos/estado/{job_id} para que setEstado reciba EstadoResp
      api
        .get<EstadoResp>(`/modelos/estado/${lastJobId}`)
        .then(({ data }) => setEstado(data))
        .catch(() => {});
    }
  }, []);

  // Accesos seguros con optional chaining; sin "as any"
  const acc = estado?.metrics?.accuracy;
  const f1 = estado?.metrics?.f1_macro ?? estado?.metrics?.f1 ?? null;
  const historyLen = estado?.history?.length ?? 0;

  return (
    <div className="grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
      <MetricCard title="API" value={health} subtitle={health === "ok" ? "online" : "revisar"} />
      <MetricCard title="Accuracy (últ. job)" value={acc != null ? (acc * 100).toFixed(1) + "%" : "—"} />
      <MetricCard title="F1 (últ. job)" value={f1 != null ? (f1 * 100).toFixed(1) + "%" : "—"} />
      <MetricCard title="Jobs history" value={historyLen} subtitle="epochs registrados" />
    </div>
  );
}
