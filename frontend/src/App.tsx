/**
 * App.tsx — Entrada visual del MVP del frontend (Día 1).
 * Landing con tarjetas; ahora DataUpload navega a /datos.
 */
import "./styles/index.css";
import { Link } from "react-router-dom";

// Páginas vacías del Día 1 (solo se mantienen como placeholders visuales)
// Ya NO montamos DataUpload aquí; se renderiza en la ruta /datos
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import Prediction from "./pages/Prediction";
import Jobs from "./pages/Jobs";

export default function App() {
  return (
    <main className="grid">
      <header className="grid">
        <h1>NeuroCampus — MVP UI (Día 1)</h1>
        <p>Frontend inicial con Vite + React + TypeScript.</p>
        <span className="badge">pantallas vacías</span>
      </header>

      <section className="card">
        <h2>Dashboard</h2>
        <Dashboard />
      </section>

      {/* Tarjeta clickeable: navega a /datos */}
      <Link to="/datos" className="card block hover:shadow focus:outline-none focus:ring">
        <h2>DataUpload</h2>
        <p className="opacity-75">Ir a la carga de datos (Día 2)</p>
        {/* Si quieres puedes dejar un mini placeholder visual aquí */}
      </Link>

      <section className="card">
        <h2>Models</h2>
        <Models />
      </section>

      <section className="card">
        <h2>Prediction</h2>
        <Prediction />
      </section>

      <section className="card">
        <h2>Jobs</h2>
        <Jobs />
      </section>
    </main>
  );
}
