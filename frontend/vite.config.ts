// frontend/vite.config.ts
import { defineConfig } from "vitest/config"; // ✅ para tipado de `test` y compatibilidad con Vitest
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // ✅ Alias "@/..." → "src/..."
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    // opcional: permite acceder desde otras IPs si lo necesitas
    host: true,
    port: 5173,
  },
  preview: {
    port: 4173,
  },
  // Config de Vitest (reconocida gracias a vitest/config)
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
    css: true,
  },
});
