/**
 * Punto de entrada de React.
 * - Crea el root y monta <App />.
 * - Importa el CSS global.
 */
import "./styles/index.css";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";


ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);