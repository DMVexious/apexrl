import { defineConfig } from 'vite';

// Cross-origin isolation enables SharedArrayBuffer => multi-threaded ONNX WASM (?threads=4).
// 'credentialless' still allows the Google Fonts stylesheet to load.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  server: { headers: isolationHeaders, fs: { allow: ['..'] } },
  preview: { headers: isolationHeaders },
  // Keep ORT out of the dep optimizer so its `new URL('*.wasm', import.meta.url)` resolves to node_modules
  // in dev; in production Vite emits the .wasm as a hashed asset automatically.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
});
