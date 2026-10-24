import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

const DEFAULT_HOST = 'wss://maincloud.spacetimedb.com';   // same default as src/main.tsx

// The SpacetimeDB host as both a WebSocket and an HTTP origin: the SDK POSTs
// /v1/identity/websocket-token over http(s) before it opens the ws(s) socket.
function spacetimeOrigins(host: string): string[] {
  const u = new URL(host);
  const secure = u.protocol === 'wss:' || u.protocol === 'https:';
  return [`${secure ? 'wss' : 'ws'}://${u.host}`, `${secure ? 'https' : 'http'}://${u.host}`];
}

// Content-Security-Policy as a <meta> tag in production builds (vite build / vite preview), so it
// holds on any static host. Row payloads in the public tables are written by anyone, so this is
// defense in depth behind React's escaping: no injected or third-party scripts, and the page can
// only talk to SpacetimeDB and fetch OSM tiles. Not applied to `vite` dev, whose React refresh
// preamble is an inline script. 'unsafe-eval' is required: the spacetimedb SDK compiles its BSATN
// (de)serializers with Function() from the module schema (field names, never row data).
function csp(): Plugin {
  let connect = '';
  return {
    name: 'kite-csp',
    apply: 'build',
    configResolved(config) {
      connect = spacetimeOrigins(config.env.VITE_SPACETIMEDB_HOST || DEFAULT_HOST).join(' ');
    },
    transformIndexHtml() {
      const policy = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-eval'",
        "style-src 'self'",
        "img-src 'self' data: https://tile.openstreetmap.org",
        "font-src 'self'",
        `connect-src 'self' ${connect}`,
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
      ].join('; ');
      return [{ tag: 'meta', attrs: { 'http-equiv': 'Content-Security-Policy', content: policy }, injectTo: 'head-prepend' }];
    },
  };
}

export default defineConfig({
  plugins: [react(), csp()],
});
