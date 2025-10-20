import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import Models from "./pages/Models";
import Jobs from "./pages/Jobs";

// ✅ Corrige la ruta del import de estilos:
import "./styles/index.css";

const router = createBrowserRouter([
  { path: "/", element: <Dashboard /> },
  { path: "/prediction", element: <Prediction /> },
  { path: "/models", element: <Models /> },
  { path: "/jobs", element: <Jobs /> },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
