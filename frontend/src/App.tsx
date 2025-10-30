/**
 * frontend/src/App.tsx
 * App.tsx — Entrada visual + Router.
 * - "/"                → Landing de demo (tu contenido original)
 * - "/models"          → Models
 * - "/prediction"      → Prediction
 * - "/jobs"            → Jobs
 * - "/datos"           → Página placeholder de Datos (si ya tienes una página real, impórtala y úsala aquí)
 * - "/datos/diagnostico" → Diagnóstico de validación de datasets (nueva)
 */

import "./styles/index.css";
import { Link, Routes, Route } from "react-router-dom";

// Páginas existentes
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import Prediction from "./pages/Prediction";
import Jobs from "./pages/Jobs";

// Nueva página (creada en pasos anteriores)
import DatosDiagnostico from "@/pages/DatosDiagnostico";

/** Landing de demo (tu contenido original movido a un componente) */
function HomeLanding() {
  return (
    <main className="grid" style={{ gap: 16, padding: 16 }}>
      <header className="grid" style={{ gap: 8 }}>
        <h1 style={{ margin: 0 }}>NeuroCampus — MVP UI</h1>
        <p className="opacity-75" style={{ margin: 0 }}>
          Frontend con Vite + React + TypeScript. Este landing es solo para la demo; la navegación real está en el router.
        </p>
        <div className="flex items-center gap-2">
          <span className="badge">demo</span>
          <Link to="/" className="badge" aria-label="Ir al Dashboard (ruta)">
            / (Dashboard)
          </Link>
          <Link to="/prediction" className="badge" aria-label="Ir a Prediction (ruta)">
            /prediction
          </Link>
          <Link to="/models" className="badge" aria-label="Ir a Models (ruta)">
            /models
          </Link>
          <Link to="/jobs" className="badge" aria-label="Ir a Jobs (ruta)">
            /jobs
          </Link>
          <Link to="/datos" className="badge" aria-label="Ir a DataUpload (ruta)">
            /datos
          </Link>
          <Link to="/datos/diagnostico" className="badge" aria-label="Ir a Diagnóstico (ruta)">
            /datos/diagnostico
          </Link>
        </div>
      </header>

      <section className="card">
        <div className="flex items-center justify-between">
          <h2 style={{ margin: 0 }}>Dashboard</h2>
          <Link to="/" className="badge" aria-label="Abrir Dashboard en su ruta">
            abrir ruta
          </Link>
        </div>
        <div style={{ marginTop: 12 }}>
          <Dashboard />
        </div>
      </section>

      {/* Tarjeta clickeable: navega a /datos */}
      <Link
        to="/datos"
        className="card block hover:shadow focus:outline-none focus:ring"
        aria-label="Ir a carga de datos (ruta /datos)"
      >
        <h2 style={{ marginTop: 0 }}>DataUpload</h2>
        <p className="opacity-75" style={{ marginBottom: 0 }}>
          Ir a la carga de datos (ruta dedicada). Aquí no se monta el formulario.
        </p>
      </Link>

      <section className="card">
        <div className="flex items-center justify-between">
          <h2 style={{ margin: 0 }}>Models</h2>
          <Link to="/models" className="badge" aria-label="Abrir Models en su ruta">
            abrir ruta
          </Link>
        </div>
        <div style={{ marginTop: 12 }}>
          <Models />
        </div>
      </section>

      <section className="card">
        <div className="flex items-center justify-between">
          <h2 style={{ margin: 0 }}>Prediction</h2>
          <Link to="/prediction" className="badge" aria-label="Abrir Prediction en su ruta">
            abrir ruta
          </Link>
        </div>
        <div style={{ marginTop: 12 }}>
          <Prediction />
        </div>
      </section>

      <section className="card">
        <div className="flex items-center justify-between">
          <h2 style={{ margin: 0 }}>Jobs</h2>
          <Link to="/jobs" className="badge" aria-label="Abrir Jobs en su ruta">
            abrir ruta
          </Link>
        </div>
        <div style={{ marginTop: 12 }}>
          <Jobs />
        </div>
      </section>
    </main>
  );
}

/** Placeholder de /datos (sustituye por tu página real si ya existe) */
function DatosUploadPlaceholder() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">Carga de Datos</h1>
      <p className="opacity-75">
        Página placeholder. Si ya tienes una página real de carga de datos, impórtala y úsala en la ruta <code>/datos</code>.
      </p>
      <p className="mt-2">
        Para diagnósticos rápidos, usa{" "}
        <Link className="underline" to="/datos/diagnostico">
          /datos/diagnostico
        </Link>
        .
      </p>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomeLanding />} />
      <Route path="/models" element={<Models />} />
      <Route path="/prediction" element={<Prediction />} />
      <Route path="/jobs" element={<Jobs />} />
      <Route path="/datos" element={<DatosUploadPlaceholder />} />
      <Route path="/datos/diagnostico" element={<DatosDiagnostico />} />
      {/* (opcional) 404 simple */}
      <Route path="*" element={<div className="p-6">No encontrado</div>} />
    </Routes>
  );
}
