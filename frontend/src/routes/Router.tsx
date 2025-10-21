import { createBrowserRouter } from "react-router-dom";
import MainLayout from "../layout/MainLayout";

// Páginas
import Home from "../pages/Home";
import Dashboard from "../pages/Dashboard";
import Models from "../pages/Models";
import Prediction from "../pages/Prediction";
import Jobs from "../pages/Jobs";
// Si no tienes DataUpload, comenta esta línea y su ruta:
import DataUpload from "../pages/DataUpload";

export const router = createBrowserRouter([
  {
    element: <MainLayout />,
    children: [
      { path: "/", element: <Home /> },
      { path: "/dashboard", element: <Dashboard /> },
      { path: "/models", element: <Models /> },
      { path: "/prediction", element: <Prediction /> },
      { path: "/jobs", element: <Jobs /> },
      { path: "/datos", element: <DataUpload /> }, // comenta si no existe
    ],
  },
]);
