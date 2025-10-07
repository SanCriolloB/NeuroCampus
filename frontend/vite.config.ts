import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Configuración mínima de Vite para React.
// - El plugin react habilita fast refresh y JSX.
// - El server en dev se expone en el puerto 5173.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 }
});