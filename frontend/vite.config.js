import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://localhost:8000',
      '/sessions': 'http://localhost:8000',
      '/session': 'http://localhost:8000',
      '/generate': 'http://localhost:8000',
      '/download': 'http://localhost:8000',
    },
  },
})