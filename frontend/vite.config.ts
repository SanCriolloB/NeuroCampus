import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Configuración mínima de Vite para React.
// - El plugin react habilita fast refresh y JSX.
// - El server en dev se expone en el puerto 5173.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 }
});

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",          // simula DOM en Node
    globals: true,                 // describe/it/expect globales
    setupFiles: "./src/setupTests.ts", // extensiones de expect
    css: true,                     // permite importar CSS en pruebas
  },
});