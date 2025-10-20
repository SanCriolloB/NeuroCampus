// frontend/src/pages/Jobs.tsx
import { useEffect, useState } from "react";
import api from "../services/apiClient";

// Tipo esperado del ping del backend
type JobsPingResp = { jobs: string } | { status: string } | Record<string, unknown>;

export default function Jobs() {
  const [pong, setPong] = useState<string>("…");

  useEffect(() => {
    // ✅ Tipamos la respuesta para que data no sea {}
    api
      .get<JobsPingResp>("/jobs/ping")
      .then(({ data }) => {
        // Soportar varias formas: {jobs:"pong"} o {status:"ok"} o algo desconocido
        const maybeJobs = (data as any)?.jobs;
        const maybeStatus = (data as any)?.status;
        if (typeof maybeJobs === "string") {
          setPong(maybeJobs);
        } else if (typeof maybeStatus === "string") {
          setPong(maybeStatus);
        } else {
          setPong("ok"); // valor por defecto si no viene ninguno
        }
      })
      .catch(() => setPong("error"));
  }, []);

  return (
    <div className="card">
      <div className="badge">/jobs/ping → {pong}</div>
      <ul>
        <li>training.started — job-xxx</li>
        <li>training.completed — job-xxx</li>
        <li>select_champion — job-yyy</li>
      </ul>
    </div>
  );
}
