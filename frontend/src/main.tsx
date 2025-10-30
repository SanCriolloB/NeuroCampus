/**
 * frontend/src/main.tsx
 * Punto de entrada de React: monta <App /> envuelto en <BrowserRouter>.
 * Las rutas están definidas dentro de App.tsx (usa <Routes> y <Route>).
 */
import "./styles/index.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
