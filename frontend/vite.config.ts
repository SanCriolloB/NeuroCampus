// frontend/vite.config.ts
import { defineConfig } from "vitest/config"; // ✅ soporta `test` + tipado Vitest
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Alias "@/..." → "src/..."
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: true,
    port: 5173,
  },
  preview: {
    port: 4173,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
    css: true,

    // ✅ Solo correr tests del frontend actual (no legacy UI)
    include: ["src/**/*.test.{ts,tsx}"],

    // ✅ Evita que Vitest intente ejecutar tests/archivos de _legacy_ui
    exclude: ["_legacy_ui/**", "node_modules/**", "dist/**"],
  },
});