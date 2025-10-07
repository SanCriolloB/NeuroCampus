/**
 * App.tsx — Entrada visual del MVP del frontend (Día 1).
 * Renderiza las 5 pantallas vacías para cumplir el “Listo cuando”.
 * Usa utilidades simples desde styles/index.css (grid, card, etc.).
 */
import './styles/index.css'

// Páginas vacías del Día 1 (asegúrate de tener estos archivos)
import Dashboard from './pages/Dashboard'
import DataUpload from './pages/DataUpload'
import Models from './pages/Models'
import Prediction from './pages/Prediction'
import Jobs from './pages/Jobs'

export default function App(){
  return (
    <main className="grid">
      <header className="grid">
        <h1>NeuroCampus — MVP UI (Día 1)</h1>
        {/* Puedes conservar tu texto introductorio aquí */}
        <p>Frontend inicial con Vite + React + TypeScript.</p>
        <span className="badge">pantallas vacías</span>
      </header>

      <section className="card"><h2>Dashboard</h2><Dashboard/></section>
      <section className="card"><h2>DataUpload</h2><DataUpload/></section>
      <section className="card"><h2>Models</h2><Models/></section>
      <section className="card"><h2>Prediction</h2><Prediction/></section>
      <section className="card"><h2>Jobs</h2><Jobs/></section>
    </main>
  )
}
