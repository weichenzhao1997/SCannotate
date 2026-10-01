import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/cluster':        'http://127.0.0.1:8000',
      '/shap':           'http://127.0.0.1:8000',
      '/annotate':       'http://127.0.0.1:8000',
      '/annotations':    'http://127.0.0.1:8000',
      '/load-dataset':   'http://127.0.0.1:8000',
      '/upload-dataset': 'http://127.0.0.1:8000',
      '/dataset-info':   'http://127.0.0.1:8000',
      '/export':         'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: '../backend/dist',
    emptyOutDir: true,
  },
})
