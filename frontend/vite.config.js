import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // `npm run dev` talks to `python api.py` running next to it.
    proxy: { "/api": process.env.API_URL ?? "http://127.0.0.1:8000" },
  },
});
