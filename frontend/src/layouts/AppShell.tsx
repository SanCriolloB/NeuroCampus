// frontend/src/layouts/AppShell.tsx
import { Outlet } from "react-router-dom";
import { Bell, User } from "lucide-react";
import Sidebar from "@/components/Sidebar";

export default function AppShell() {
  return (
    <div className="flex h-screen bg-[#0f1419] text-white">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-[72px] border-b border-gray-800 flex items-center justify-between px-8">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded" />
            <h1 className="text-white">NeuroCampus</h1>
          </div>

          <div className="flex items-center gap-4">
            <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors" aria-label="Notificaciones">
              <Bell className="w-5 h-5 text-gray-400" />
            </button>
            <button className="p-2 hover:bg-gray-800 rounded-full transition-colors" aria-label="Usuario">
              <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-gray-400" />
              </div>
            </button>
          </div>
        </header>

        {/* Routed Content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
