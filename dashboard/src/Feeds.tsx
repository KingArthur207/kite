import { useEffect, useState } from 'react';

// Live MJPEG camera panes for the analytics view. A plain <img> plays multipart/x-mixed-replace in every browser.
// Sources: ?webcam=<url>&drone=<url>, else what the Feeds popover saved in localStorage.
//   webcam: the UNO Q board, http://<board-ip>:8765/video.mjpg?token=<token>
//   drone:  python -m kite.sources.mjpeg --source <tello | recordings/<take> | file.mp4> --port 8090

export type FeedKey = 'webcam' | 'drone';
const STORE = (k: FeedKey) => `kite.feed.${k}`;
const RETRY_MS = 3000;

// Only http(s) URLs become an img src.
export function safeFeed(raw: string | null | undefined): string {
  if (!raw) return '';
  try {
    const u = new URL(raw.trim());
    return u.protocol === 'http:' || u.protocol === 'https:' ? u.toString() : '';
  } catch {
    return '';
  }
}

export function useFeeds() {
  const params = new URLSearchParams(location.search);
  const read = (k: FeedKey) => safeFeed(params.get(k)) || safeFeed(localStorage.getItem(STORE(k)));
  const [feeds, setFeeds] = useState<Record<FeedKey, string>>(() => ({ webcam: read('webcam'), drone: read('drone') }));
  const save = (next: Record<FeedKey, string>) => {
    const clean = { webcam: safeFeed(next.webcam), drone: safeFeed(next.drone) };
    for (const k of ['webcam', 'drone'] as const) {
      if (clean[k]) localStorage.setItem(STORE(k), clean[k]);
      else localStorage.removeItem(STORE(k));
    }
    setFeeds(clean);
  };
  return [feeds, save] as const;
}

type PaneProps = { label: string; param: FeedKey; src: string };

export function VideoPane({ label, param, src }: PaneProps) {
  const [state, setState] = useState<'loading' | 'live' | 'error'>('loading');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => { setState('loading'); setAttempt(0); }, [src]);
  useEffect(() => {
    if (state !== 'error' || !src) return;
    const id = setTimeout(() => { setState('loading'); setAttempt(n => n + 1); }, RETRY_MS);
    return () => clearTimeout(id);
  }, [state, src]);

  // A fresh URL per retry so the browser opens a new connection instead of reusing the failed one.
  let url = src;
  if (src && attempt > 0) {
    const u = new URL(src);
    u.searchParams.set('_r', String(attempt));
    url = u.toString();
  }
  const live = !!src && state === 'live';
  return (
    <figure className={`video-pane ${live ? 'live' : 'offline'}`}>
      {src && state !== 'error' && (
        <img key={url} src={url} alt={`${label} live video`} onLoad={() => setState('live')} onError={() => setState('error')} />
      )}
      {!live && (
        <div className="video-empty">
          {!src ? <p>No feed: set <code>?{param}=</code></p>
            : state === 'error' ? <p>Can't reach the feed. Retrying…<br /><span className="muted small">{src}</span></p>
              : <p className="muted">Connecting…</p>}
        </div>
      )}
      <figcaption>
        <span className="video-label">{label}</span>
        <span className={`video-state ${live ? 'live' : ''}`}><i aria-hidden="true" />{live ? 'Live' : 'Offline'}</span>
      </figcaption>
    </figure>
  );
}

type SettingsProps = { feeds: Record<FeedKey, string>; onSave: (f: Record<FeedKey, string>) => void };

export function FeedSettings({ feeds, onSave }: SettingsProps) {
  const [draft, setDraft] = useState(feeds);
  return (
    <details className="feed-settings" onToggle={e => { if ((e.target as HTMLDetailsElement).open) setDraft(feeds); }}>
      <summary>Feeds</summary>
      <form onSubmit={e => { e.preventDefault(); onSave(draft); (e.currentTarget.parentElement as HTMLDetailsElement).open = false; }}>
        <label>Walker camera <input value={draft.webcam} onChange={e => setDraft({ ...draft, webcam: e.target.value })}
          placeholder="http://<board-ip>:8765/video.mjpg?token=…" autoComplete="off" spellCheck={false} /></label>
        <label>Drone camera <input value={draft.drone} onChange={e => setDraft({ ...draft, drone: e.target.value })}
          placeholder="http://localhost:8090/video.mjpg" autoComplete="off" spellCheck={false} /></label>
        <button type="submit">Save</button>
      </form>
    </details>
  );
}
