import { NavLink } from "react-router-dom";

const Item = ({ to, label }: { to: string; label: string }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      "navlink" + (isActive ? " active" : "")
    }
  >
    <span>{label}</span>
  </NavLink>
);

export default function Sidebar() {
  return (
    <div>
      <div style={{ padding: "10px 8px", fontWeight: 700 }}>NeuroCampus</div>
      <nav style={{ display: "grid", gap: 6 }}>
        <Item to="/" label="Inicio" />
        <Item to="/dashboard" label="Dashboard" />
        <Item to="/prediction" label="Prediction" />
        <Item to="/models" label="Models" />
        <Item to="/jobs" label="Jobs" />
        <Item to="/datos" label="Datos" />
      </nav>
    </div>
  );
}
