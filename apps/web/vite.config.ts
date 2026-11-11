import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    host: "127.0.0.1",

    allowedHosts: ["slaw-hazing-porthole.ngrok-free.dev"],

    hmr: false,

    proxy: {
      "/health": {
        target: backend,
        changeOrigin: true,
      },

      "/reference-groups": {
        target: backend,
        changeOrigin: true,
      },

      "/analyze": {
        target: backend,
        changeOrigin: true,
      },

      "/analyze-upload": {
        target: backend,
        changeOrigin: true,
      },

      "/score-upload": {
        target: backend,
        changeOrigin: true,
      },

      "/artifact-file": {
        target: backend,
        changeOrigin: true,
      },
    },
  },
});
