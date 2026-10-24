"""A Tello-like H.264 video sender on local UDP, for reproducing "packets arrive but no frame decodes".

    python scripts/tello_video_sim.py                          # 960x720 30 fps, IDR every 1 s, to 127.0.0.1:11111
    python scripts/tello_video_sim.py --keyint 150 --loss 0.02 # IDR every 5 s, 2 % datagrams dropped
    python scripts/tello_video_sim.py --replay recordings/video_doctor_<stamp>   # a real capture, original timing

What it imitates (the Tello's known stream; see docs/mac-video-root-cause.md):
  * raw H.264 Annex B, no container, no timestamps, 960x720, ~30 fps, Main profile, no B-frames
  * each access unit (SPS+PPS+IDR, or one P frame) is cut into 1460-byte datagrams; the last one is shorter
  * the datagrams of one frame leave back to back (a burst, no pacing inside a frame), frames 1/30 s apart
  * it is already streaming when the reader opens (the reader joins mid-GOP, like after `streamon`)
Faults: --loss (independent datagram drops), --burst-loss (drop the tail of a frame's burst beyond N datagrams, i.e.
what a too-small receive buffer does), --params separate (SPS and PPS in their own small datagrams).

Content is detailed and moving (testsrc2 + temporal noise) so the IDR frames are tens of KB like the drone's.
The encoded stream is cached in the scratch dir given by --cache (default: $TMPDIR/kite_tello_sim).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DGRAM = 1460


def encode(path: Path, seconds: float = 20.0, fps: int = 30, keyint: int = 30, bitrate: str = "3M",
           profile: str = "main", noise: int = 12, intra_refresh: bool = False, strip_sei: bool = False) -> Path:
    """Encode a Tello-like raw H.264 file with ffmpeg/libx264 (SPS+PPS repeated before every IDR)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("needs the ffmpeg CLI")
    x264 = "repeat-headers=1:bframes=0" + (":intra-refresh=1" if intra_refresh else "")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc2=size=960x720:rate={fps},noise=alls={noise}:allf=t",
           "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
           "-profile:v", profile, "-level", "4.0", "-pix_fmt", "yuv420p", "-g", str(keyint), "-keyint_min", str(keyint),
           "-sc_threshold", "0", "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
           "-x264-params", x264, "-bsf:v", "h264_metadata=aud=remove" + (",filter_units=remove_types=6" if strip_sei else ""),
           "-f", "h264", str(path)]
    subprocess.run(cmd, check=True)
    return path


def encode_nonidr(path: Path, seconds: float = 20.0, fps: int = 30, keyint: int = 30, bitrate_kbps: int = 3000) -> Path:
    """Keyframes that are NOT IDR: one IDR at the start, then a non-IDR I frame every `keyint` frames and no
    recovery-point SEI (x264 CLI qpfile type 'i'). The Sender puts SPS/PPS before each one (inject_params)."""
    if shutil.which("x264") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("needs the ffmpeg and x264 CLIs (brew install x264)")
    n = int(seconds * fps)
    qp = path.with_suffix(".qp")
    qp.write_text("".join(f"{i} {'I' if i == 0 else 'i'} -1\n" for i in range(0, n, keyint)))
    y4m = path.with_suffix(".y4m")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=960x720:rate={fps},noise=alls=12:allf=t", "-t", str(seconds), "-pix_fmt", "yuv420p",
                    str(y4m)], check=True)
    subprocess.run(["x264", "--quiet", "--qpfile", str(qp), "--keyint", "100000", "--min-keyint", "100000", "--open-gop",
                    "--bframes", "0", "--profile", "main", "--preset", "veryfast", "--bitrate", str(bitrate_kbps),
                    "--vbv-maxrate", str(bitrate_kbps), "--vbv-bufsize", str(bitrate_kbps), "--no-scenecut",
                    "-o", str(path), str(y4m)], check=True, capture_output=True)
    y4m.unlink()
    # drop SEI (x264's version string): no recovery-point hint at all
    path.write_bytes(b"".join(nal for _, nals in access_units(path.read_bytes()) for t, nal in nals if t != 6))
    return path


def cached_stream(cache: Path, **kw) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(repr(sorted(kw.items())).encode()).hexdigest()[:10]
    path = cache / f"tello_sim_{key}.h264"
    if not path.exists() or path.stat().st_size == 0:
        if kw.pop("nonidr", False):
            encode_nonidr(path, keyint=kw.get("keyint", 30))
        else:
            encode(path, **kw)
    return path


def nal_starts(data: bytes) -> list[int]:
    """Offsets of every NAL unit's start code (3- or 4-byte), pointing at the first 0x00."""
    out, i = [], 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0:
            return out
        out.append(i - 1 if i > 0 and data[i - 1] == 0 else i)
        i += 3


def access_units(data: bytes) -> list[tuple[bytes, list[tuple[int, bytes]]]]:
    """Split Annex B into access units: [(au_bytes, [(nal_type, nal_bytes), ...])]. A new AU starts at an SPS/PPS/AUD/
    SEI after a slice, or at a slice whose first_mb_in_slice is 0 (byte after the header has its top bit set)."""
    starts = nal_starts(data) + [len(data)]
    aus, cur, seen_slice = [], [], False
    for a, b in zip(starts, starts[1:]):
        nal = data[a:b]
        hdr = nal.index(b"\x00\x00\x01") + 3
        typ = nal[hdr] & 0x1F
        first_mb_zero = typ in (1, 5) and len(nal) > hdr + 1 and nal[hdr + 1] & 0x80
        if cur and seen_slice and (typ in (6, 7, 8, 9) or first_mb_zero):
            aus.append(cur)
            cur, seen_slice = [], False
        cur.append((typ, nal))
        seen_slice = seen_slice or typ in (1, 5)
    if cur:
        aus.append(cur)
    return [(b"".join(n for _, n in au), au) for au in aus]


def _ue(bits: str, i: int) -> tuple[int, int]:
    z = 0
    while bits[i + z] == "0":
        z += 1
    return int(bits[i + z:i + 2 * z + 1], 2) - 1, i + 2 * z + 1


def slice_type(nal: bytes) -> int | None:
    """slice_type % 5 of a slice NAL (0 P, 1 B, 2 I), None for other NAL types."""
    hdr = nal.index(b"\x00\x00\x01") + 3
    if nal[hdr] & 0x1F not in (1, 5):
        return None
    body = nal[hdr + 1:hdr + 12].replace(b"\x00\x00\x03", b"\x00\x00")
    bits = "".join(f"{b:08b}" for b in body)
    try:
        _, i = _ue(bits, 0)
        st, _ = _ue(bits, i)
    except (IndexError, ValueError):  # truncated at a datagram edge
        return None
    return st % 5


def inject_params(aus, every: int = 0):
    """Put the stream's first SPS+PPS in front of every access unit that starts with an I slice and has none (a
    stream whose keyframes are non-IDR I frames, each preceded by its parameter sets). every=N: also in front of
    every Nth access unit (parameter sets far more often than keyframes)."""
    sps = next(n for _, nals in aus for t, n in nals if t == 7)
    pps = next(n for _, nals in aus for t, n in nals if t == 8)
    out = []
    for k, (raw, nals) in enumerate(aus):
        types = [t for t, _ in nals]
        first_slice = next((n for t, n in nals if t in (1, 5)), None)
        periodic = every and k % every == 0
        if 7 not in types and first_slice is not None and (periodic or slice_type(first_slice) == 2):
            nals = [(7, sps), (8, pps)] + nals
            raw = sps + pps + raw
        out.append((raw, nals))
    return out


def datagrams(au, params: str = "inline") -> list[bytes]:
    """One access unit -> its datagrams. params='separate': SPS/PPS each in their own datagram first."""
    raw, nals = au
    out = []
    if params == "separate":
        rest = b""
        for typ, nal in nals:
            if typ in (7, 8):
                out.append(nal)
            else:
                rest += nal
        raw = rest
    out += [raw[i:i + DGRAM] for i in range(0, len(raw), DGRAM)]
    return out


class Sender:
    """Sends a Tello-like stream in a background thread until stop(). Loops the file forever."""

    def __init__(self, path: Path | None = None, host: str = "127.0.0.1", port: int = 11111, fps: float = 30.0,
                 loss: float = 0.0, burst_loss: int = 0, params: str = "inline", seed: int = 1,
                 replay: Path | None = None, start_offset_s: float = 0.0, burst_gap_us: float = 0.0,
                 outage: tuple[float, float] | None = None, big_loss: float = 0.0, inject: bool = False,
                 params_every: int = 0):
        self.addr = (host, port)
        self.fps, self.loss, self.burst_loss, self.params = fps, loss, burst_loss, params
        self.rng = random.Random(seed)
        self.burst_gap_us = burst_gap_us
        self.big_loss = big_loss  # extra loss only inside frames of > 10 datagrams (the keyframes)
        self.outage = outage  # (period s, length s): every period, drop everything for length (Wi-Fi off-channel)
        self._t0 = time.monotonic()
        self.sent = self.dropped = 0
        self.frames: list[list[bytes]] = []
        self.replay_rows: list[tuple[float, bytes]] | None = None
        if replay is not None:
            self.replay_rows = load_capture(replay)
        else:
            data = path.read_bytes()
            aus = access_units(data)
            if inject or params_every:
                aus = inject_params(aus, params_every)
            self.frames = [datagrams(au, params) for au in aus]
            # start mid-stream, like a drone that has been streaming for a while
            k = int(start_offset_s * fps) % max(len(self.frames), 1)
            self.frames = self.frames[k:] + self.frames[:k]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> Sender:
        self._thread.start()
        return self

    def _send(self, d: bytes) -> None:
        if self.outage and (time.monotonic() - self._t0) % self.outage[0] < self.outage[1]:
            self.dropped += 1
            return
        if self.loss and self.rng.random() < self.loss:
            self.dropped += 1
            return
        for _ in range(50):
            try:
                self.sock.sendto(d, self.addr)
                self.sent += 1
                return
            except BlockingIOError:
                time.sleep(0.0005)
            except OSError as e:  # ENOBUFS on a long burst over loopback: retry briefly
                if e.errno != 55:
                    raise
                time.sleep(0.0005)

    def _run(self) -> None:
        if self.replay_rows is not None:
            return self._run_replay()
        period, t_next, i = 1.0 / self.fps, time.monotonic(), 0
        while not self._stop.is_set():
            for j, d in enumerate(self.frames[i % len(self.frames)]):
                if self.burst_loss and j >= self.burst_loss:
                    self.dropped += 1
                    continue
                if self.big_loss and len(self.frames[i % len(self.frames)]) > 10 and self.rng.random() < self.big_loss:
                    self.dropped += 1
                    continue
                self._send(d)
                if self.burst_gap_us:
                    t_end = time.perf_counter() + self.burst_gap_us / 1e6
                    while time.perf_counter() < t_end:
                        pass
            i += 1
            t_next += period
            delay = t_next - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                t_next = time.monotonic()

    def _run_replay(self) -> None:
        rows = self.replay_rows
        while not self._stop.is_set():
            t0 = time.monotonic()
            for t, d in rows:
                delay = t0 + t - time.monotonic()
                if delay > 0.002:
                    if self._stop.wait(delay):
                        return
                elif delay > 0:
                    while time.monotonic() < t0 + t:
                        pass
                self._send(d)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.sock.close()


def load_capture(folder: Path) -> list[tuple[float, bytes]]:
    """A kite.video_doctor capture -> [(t, datagram)] with the recorded arrival times."""
    data = (folder / "video.h264").read_bytes()
    rows, off = [], 0
    with open(folder / "packets.csv") as fh:
        for r in csv.DictReader(fh):
            n = int(r["size"])
            rows.append((float(r["t"]), data[off:off + n]))
            off += n
    return rows


def default_cache() -> Path:
    return Path(os.environ.get("KITE_SIM_CACHE", Path(tempfile.gettempdir()) / "kite_tello_sim"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=11111)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0: until Ctrl+C)")
    p.add_argument("--keyint", type=int, default=30, help="frames between IDRs (Tello: see the report)")
    p.add_argument("--bitrate", default="3M")
    p.add_argument("--profile", default="main")
    p.add_argument("--intra-refresh", action="store_true", help="no IDR at all after the first: periodic intra refresh")
    p.add_argument("--strip-sei", action="store_true", help="remove SEI NAL units (no recovery-point SEI)")
    p.add_argument("--outage", help="PERIOD_MS:LEN_MS - drop every datagram for LEN ms once every PERIOD ms")
    p.add_argument("--loss", type=float, default=0.0)
    p.add_argument("--big-loss", type=float, default=0.0, help="extra loss inside keyframe-sized bursts only")
    p.add_argument("--burst-loss", type=int, default=0, help="keep only the first N datagrams of every frame")
    p.add_argument("--params", choices=("inline", "separate"), default="inline")
    p.add_argument("--start-offset", type=float, default=0.37, help="start this many seconds into the GOP pattern")
    p.add_argument("--replay", type=Path, help="a kite.video_doctor folder: replay its datagrams with their timing")
    p.add_argument("--cache", type=Path, default=default_cache())
    p.add_argument("--file", type=Path, help="send this raw .h264 file instead of encoding one")
    p.add_argument("--inject-params", action="store_true", help="SPS+PPS before every I frame (non-IDR keyframes)")
    p.add_argument("--params-every", type=int, default=0, help="also send SPS+PPS before every Nth frame")
    p.add_argument("--nonidr", action="store_true", help="keyframes are non-IDR I frames with SPS+PPS before each")
    a = p.parse_args(argv)
    path = None
    if a.file:
        path = a.file
    elif a.replay is None:
        path = cached_stream(a.cache, keyint=a.keyint, bitrate=a.bitrate, profile=a.profile, intra_refresh=a.intra_refresh,
                             strip_sei=a.strip_sei, **({"nonidr": True} if a.nonidr else {}))
    outage = tuple(float(x) / 1000 for x in a.outage.split(":")) if a.outage else None
    s = Sender(path, a.host, a.port, loss=a.loss, burst_loss=a.burst_loss, params=a.params, replay=a.replay,
               start_offset_s=a.start_offset, outage=outage,
               big_loss=a.big_loss, inject=a.inject_params or a.nonidr, params_every=a.params_every).start()
    print(f"streaming {'replay ' + str(a.replay) if a.replay else path} -> {a.host}:{a.port}", flush=True)
    try:
        t0 = time.monotonic()
        while not a.seconds or time.monotonic() - t0 < a.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    s.stop()
    print(f"sent {s.sent} datagrams, dropped {s.dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
