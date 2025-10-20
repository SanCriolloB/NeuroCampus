import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import Jobs from "./pages/Jobs";
// importa tus otras páginas si las usas en rutas públicas

import "./index.css";

const router = createBrowserRouter([
  { path: "/", element: <Dashboard /> },
  { path: "/prediction", element: <Prediction /> },
  { path: "/jobs", element: <Jobs /> },
  // agrega /models, /datos, etc., si quieres navegación directa
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
