import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";

export default function MainLayout() {
  return (
    <div
      className="grid"
      style={{
        gridTemplateColumns: "260px 1fr",
        gridTemplateRows: "56px 1fr",
        minHeight: "100vh",
      }}
    >
      {/* Sidebar */}
      <aside className="sidebar" style={{ gridRow: "1 / span 2" }}>
        <Sidebar />
      </aside>

      {/* Topbar */}
      <header className="topbar">
        <Topbar />
      </header>

      {/* Contenido */}
      <main style={{ padding: 16 }}>
        <Outlet />
      </main>
    </div>
  );
}
