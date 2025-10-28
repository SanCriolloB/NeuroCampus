// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // opcional: permite acceder desde otras IPs si lo necesitas
    host: true,
    port: 5173,
  },
  preview: {
    port: 4173,
  },
  // Config de Vitest
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
    css: true,
  },
});
