import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Listen on all network addresses (useful for Wi-Fi and tunnels)
    allowedHosts: true, // Allows Cloudflare Tunnel, ngrok, and all external hosts
    proxy: {
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/detect': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/video': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
})
