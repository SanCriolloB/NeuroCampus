/**
 * Punto de entrada de React.
 * - Monta Router con rutas "/" (App) y "/datos" (DataUpload).
 * - Importa el CSS global.
 */
import "./styles/index.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

// Páginas
import App from "./App"; // tu landing actual
import DataUpload from "./pages/DataUpload"; // la pantalla del Día 2

const router = createBrowserRouter([
  { path: "/", element: <App /> },
  { path: "/datos", element: <DataUpload /> },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
