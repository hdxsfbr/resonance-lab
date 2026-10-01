import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API target for both `vite` (dev) and `vite preview`; override with VITE_API_TARGET=http://host:port
const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:8000'
const proxy = { '/api': { target: apiTarget, changeOrigin: true } }

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { proxy },
  build: { outDir: 'dist', sourcemap: false },
})
