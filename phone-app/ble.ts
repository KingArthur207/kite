// The Bluetooth link to every Kite receiver in range (python -m phone.ble on the main computer and on the
// UNO Q board): find them, connect to each, write every fix to all of them, reconnect forever.
import { BleManager, type Device, type Subscription } from 'react-native-ble-plx';
import { chunk, writeSize } from './frame';

export const SERVICE_UUID = '4b495445-0000-4b69-7465-50686f6e6500';
export const FIX_CHAR_UUID = '4b495445-0001-4b69-7465-50686f6e6500';

export type LinkState = 'searching' | 'connected' | 'lost';

const SEND_MS = 500; // 2 Hz
const CONNECT_TIMEOUT_MS = 10_000;
const RETRY_MS = 1500;

const manager = new BleManager();

interface LinkHandlers {
  nextLine: () => string | null; // the newline-terminated fix to send now, or null if there is none yet
  onState: (state: LinkState) => void;
  onSent: (count: number) => void;
}

interface Peer {
  device: Device;
  size: number;
  writing: boolean;
  lostSub: Subscription;
}

// Starts the link; returns a function that stops it and disconnects from every receiver.
export function startLink({ nextLine, onState, onSent }: LinkHandlers): () => void {
  let alive = true;
  const peers = new Map<string, Peer>(); // connected receivers, by device id
  const connecting = new Set<string>();
  let state: LinkState = 'searching';
  let sent = 0;
  let retry: ReturnType<typeof setTimeout> | undefined;

  const setState = (next: LinkState) => {
    if (next === state) return;
    state = next;
    onState(next);
  };
  // Connected while at least one receiver is; lost once the last one drops.
  const refresh = () => {
    if (peers.size > 0) setState('connected');
    else if (state === 'connected') setState('lost');
  };

  // Scanning stays on while connected, so a second receiver (or one that came back) is picked up.
  const scan = () => {
    manager.startDeviceScan([SERVICE_UUID], null, (err, found) => {
      if (!alive) return;
      if (err) {
        console.warn('[kite] scan', err.message);
        manager.stopDeviceScan();
        clearTimeout(retry);
        retry = setTimeout(() => alive && scan(), RETRY_MS);
        return;
      }
      if (!found || peers.has(found.id) || connecting.has(found.id)) return;
      connecting.add(found.id);
      connect(found).finally(() => connecting.delete(found.id));
    });
  };

  const drop = (id: string) => {
    const p = peers.get(id);
    if (!p) return;
    peers.delete(id);
    p.lostSub.remove();
    p.device.cancelConnection().catch(() => undefined);
    if (alive) refresh();
  };

  const connect = async (found: Device) => {
    try {
      const d = await found.connect({ timeout: CONNECT_TIMEOUT_MS });
      await d.discoverAllServicesAndCharacteristics();
      if (!alive) {
        d.cancelConnection().catch(() => undefined);
        return;
      }
      peers.set(d.id, { device: d, size: writeSize(d.mtu), writing: false, lostSub: d.onDisconnected(() => drop(d.id)) });
      refresh();
    } catch (e) {
      console.warn('[kite] connect', e instanceof Error ? e.message : e);
      found.cancelConnection().catch(() => undefined);
    }
  };

  const sendTo = async (p: Peer, line: string): Promise<boolean> => {
    p.writing = true;
    try {
      // with-response writes keep chunks in order and surface a dead link as an error
      for (const part of chunk(line, p.size)) {
        await p.device.writeCharacteristicWithResponseForService(SERVICE_UUID, FIX_CHAR_UUID, btoa(part));
      }
      return true;
    } catch (e) {
      console.warn('[kite] write', e instanceof Error ? e.message : e);
      drop(p.device.id);
      return false;
    } finally {
      p.writing = false;
    }
  };

  const send = async () => {
    const line = peers.size > 0 ? nextLine() : null;
    if (!line) return;
    const writes = [...peers.values()].filter((p) => !p.writing).map((p) => sendTo(p, line));
    if (writes.length === 0) return;
    if ((await Promise.all(writes)).some(Boolean)) onSent(++sent);
  };

  onState('searching');
  // Scanning before Bluetooth is powered on fails, so wait for it (the simulator never gets there: it just searches).
  const powerSub = manager.onStateChange((s) => {
    if (s !== 'PoweredOn') return;
    powerSub.remove();
    if (alive) scan();
  }, true);
  const timer = setInterval(send, SEND_MS);

  return () => {
    alive = false;
    clearInterval(timer);
    clearTimeout(retry);
    powerSub.remove();
    manager.stopDeviceScan();
    for (const id of [...peers.keys()]) drop(id);
  };
}
