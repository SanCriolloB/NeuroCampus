import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="grid" style={{ gap: 16 }}>
      <header className="grid" style={{ gap: 6 }}>
        <h1 style={{ margin: 0 }}>NeuroCampus — Panel</h1>
        <p className="opacity-75" style={{ margin: 0 }}>
          Selecciona una sección para comenzar.
        </p>
      </header>

      <section className="grid" style={{ gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        <Link to="/dashboard" className="card block">
          <h2 style={{ marginTop: 0 }}>Dashboard</h2>
          <p className="opacity-75">KPIs de salud, métricas y actividad reciente.</p>
        </Link>

        <Link to="/models" className="card block">
          <h2 style={{ marginTop: 0 }}>Models</h2>
          <p className="opacity-75">Entrenar RBM y ver curva de pérdida.</p>
        </Link>

        <Link to="/prediction" className="card block">
          <h2 style={{ marginTop: 0 }}>Prediction</h2>
          <p className="opacity-75">Predicción online y por lote.</p>
        </Link>

        <Link to="/jobs" className="card block">
          <h2 style={{ marginTop: 0 }}>Jobs</h2>
          <p className="opacity-75">Ejecuciones/cola y verificación de ping.</p>
        </Link>

        <Link to="/datos" className="card block">
          <h2 style={{ marginTop: 0 }}>Datos</h2>
          <p className="opacity-75">Carga y validación de datasets.</p>
        </Link>
      </section>
    </div>
  );
}
