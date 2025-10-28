/**
 * Punto de entrada de React.
 * - Monta Router con rutas:
 *   "/" (App), "/datos" (DataUpload), "/admin/cleanup" (AdminCleanup).
 * - Importa el CSS global.
 */
import "./styles/index.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

// Páginas
import App from "./App";                   // Landing
import DataUpload from "./pages/DataUpload"; // Pantalla de carga de datos
import AdminCleanupPage from "./pages/AdminCleanup"; // Día 4: UI administración limpieza

const router = createBrowserRouter([
  { path: "/", element: <App /> },
  { path: "/datos", element: <DataUpload /> },
  { path: "/admin/cleanup", element: <AdminCleanupPage /> }, // <-- nueva ruta
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
