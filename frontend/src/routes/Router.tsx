// frontend/src/routes/Router.tsx
import { createBrowserRouter, Navigate } from "react-router-dom";
import AppShell from "@/layouts/AppShell";

import DashboardPage from "@/pages/DashboardPage";
import DatosPage from "@/pages/DatosPage";
import ModelosPage from "@/pages/ModelosPage";
import PrediccionesPage from "@/pages/PrediccionesPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: "dashboard", element: <DashboardPage /> },
      { path: "datos", element: <DatosPage /> },
      { path: "models", element: <ModelosPage /> },
      { path: "prediction", element: <PrediccionesPage /> },
      { path: "*", element: <Navigate to="/dashboard" replace /> },
    ],
  },
]);
