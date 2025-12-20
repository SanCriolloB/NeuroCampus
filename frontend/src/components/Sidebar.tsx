// frontend/src/components/Sidebar.tsx
import { NavLink } from "react-router-dom";
import { Home, Database, Network, TrendingUp } from "lucide-react";

type NavItem = {
  to: string;
  label: string;
  Icon: React.ComponentType<{ className?: string }>;
  end?: boolean;
};

const navItems: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", Icon: Home, end: true },
  { to: "/datos", label: "Data", Icon: Database },
  { to: "/models", label: "Models", Icon: Network },
  { to: "/prediction", label: "Predictions", Icon: TrendingUp },
];

export default function Sidebar() {
  return (
    <aside className="w-[200px] bg-[#1a1f2e] border-r border-gray-800 flex flex-col">
      <nav className="flex-1 py-4">
        {navItems.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `w-full px-4 py-3 flex items-center gap-3 transition-colors ${
                isActive
                  ? "bg-blue-600/20 text-blue-400 border-r-2 border-blue-500"
                  : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
              }`
            }
          >
            <Icon className="w-5 h-5" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
