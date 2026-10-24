// A WebSocket that reconnects quietly; send() drops messages while disconnected.
export function reconnectingSocket(url: string, binary = false) {
  let ws: WebSocket | null = null;
  const connect = () => {
    const s = new WebSocket(url);
    if (binary) s.binaryType = "arraybuffer";
    s.onopen = () => { ws = s; };
    s.onclose = () => { ws = null; setTimeout(connect, 2000); };
    s.onerror = () => s.close();
  };
  connect();
  return {
    get open() { return ws !== null; },
    get buffered() { return ws?.bufferedAmount ?? 0; },
    send(data: string | ArrayBuffer | Blob) { ws?.send(data); },
    json(obj: object) { ws?.send(JSON.stringify(obj)); },
  };
}
