// The Bluetooth link to the Kite computer (python -m phone.ble): find it, connect, write fixes, reconnect forever.
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

// Starts the link; returns a function that stops it and disconnects.
export function startLink({ nextLine, onState, onSent }: LinkHandlers): () => void {
  let alive = true;
  let device: Device | null = null;
  let size = writeSize(null);
  let writing = false;
  let connecting = false;
  let sent = 0;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let lostSub: Subscription | undefined;

  const later = (fn: () => void) => {
    clearTimeout(retry);
    retry = setTimeout(() => alive && fn(), RETRY_MS);
  };

  const scan = () => {
    manager.startDeviceScan([SERVICE_UUID], null, (err, found) => {
      if (!alive) return;
      if (err) {
        console.warn('[kite] scan', err.message);
        manager.stopDeviceScan();
        later(scan);
        return;
      }
      if (!found || connecting) return;
      connecting = true;
      manager.stopDeviceScan();
      connect(found).finally(() => (connecting = false));
    });
  };

  // The link dropped (disconnect event or a failed write): forget the device and look for it again.
  const drop = () => {
    if (!device) return;
    const d = device;
    device = null;
    lostSub?.remove();
    d.cancelConnection().catch(() => undefined);
    if (!alive) return;
    onState('lost');
    later(scan);
  };

  const connect = async (found: Device) => {
    try {
      const d = await found.connect({ timeout: CONNECT_TIMEOUT_MS });
      await d.discoverAllServicesAndCharacteristics();
      if (!alive) {
        d.cancelConnection().catch(() => undefined);
        return;
      }
      device = d;
      size = writeSize(d.mtu);
      lostSub = d.onDisconnected(drop);
      onState('connected');
    } catch (e) {
      console.warn('[kite] connect', e instanceof Error ? e.message : e);
      found.cancelConnection().catch(() => undefined);
      if (alive) later(scan);
    }
  };

  const send = async () => {
    const d = device;
    const line = d && !writing ? nextLine() : null;
    if (!d || !line) return;
    writing = true;
    try {
      // with-response writes keep chunks in order and surface a dead link as an error
      for (const part of chunk(line, size)) {
        await d.writeCharacteristicWithResponseForService(SERVICE_UUID, FIX_CHAR_UUID, btoa(part));
      }
      onSent(++sent);
    } catch (e) {
      console.warn('[kite] write', e instanceof Error ? e.message : e);
      if (device === d) drop();
    } finally {
      writing = false;
    }
  };

  onState('searching');
  // Scanning before Bluetooth is powered on fails, so wait for it (the simulator never gets there: it just searches).
  const powerSub = manager.onStateChange((state) => {
    if (state !== 'PoweredOn') return;
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
    lostSub?.remove();
    device?.cancelConnection().catch(() => undefined);
    device = null;
  };
}
