"""Run every Tello video reader against a Tello-like UDP stream and tabulate what each one delivers.

    python scripts/mac_video_repro.py                       # the default matrix (a few minutes)
    python scripts/mac_video_repro.py --readers pyav_kite,opencv_kite --conditions clean,loss2
    python scripts/mac_video_repro.py --replay recordings/video_doctor_<stamp>   # a real capture instead of the sim
    python scripts/mac_video_repro.py --reader opencv_kite --port 11190 --seconds 12   # one reader (child mode)

Every reader runs in its own process (a fresh FFmpeg, like the real app) while scripts/tello_video_sim.py streams in
another; the reader process prints one RESULT json line. FFmpeg's log lines are scanned for the error signatures seen
on the real drone. See docs/mac-video-root-cause.md for the results on Arthur's Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "scripts" / "tello_video_sim.py"
KITE_OPENCV_OPTS = "fflags;nobuffer|flags;low_delay|framedrop;1|probesize;10000000|analyzeduration;12000000"

SIGNATURES = {
    "non-existing PPS": r"non-existing PPS",
    "sps_id out of range": r"sps_id \d+ out of range",
    "decode_slice_header error": r"decode_slice_header error",
    "missing picture in AU": r"missing picture in access unit",
    "error while decoding MB": r"error while decoding MB",
    "Stream timeout": r"Stream timeout triggered",
    "circular buffer overrun": r"[Cc]ircular buffer overrun",
    "can't capture by name": r"can't be used to capture by name",
    "could not find codec parameters": r"Could not find codec parameters",
}

# reader name -> what it is
READERS = {
    "opencv_kite": "kite VideoStream as drone_test uses it: kite's OPENCV_FFMPEG_CAPTURE_OPTIONS, open 5 s / read 3 s",
    "opencv_kite_15s": "kite VideoStream, kite's options, open/read 15 s (TelloVideoSource backend=opencv)",
    "opencv_noopts": "kite VideoStream, NO capture options (FFmpeg defaults), open 5 s / read 3 s",
    "opencv_fast": "kite VideoStream, nobuffer/low_delay, probesize 32 KB... see FAST_OPTS, open 5 s",
    "opencv_fast_buf": "opencv_fast + udp://...?buffer_size=4M&fifo_size=..&overrun_nonfatal=1",
    "opencv_probe_15s": "kite VideoStream, PROBE_OPTS (formatprobesize 10 MB, analyzeduration 1 s), open/read 15 s",
    "pyav_kite": "kite PyAVStream (av.open format=h264, timeout 15/15, nobuffer/low_delay)",
    "pyav_buf": "PyAVStream on udp://...?buffer_size=4M&overrun_nonfatal=1",
    "pyav_noprobe": "av.open with probesize/analyzeduration small (no nobuffer)",
    "tello_source_pyav": "kite TelloVideoSource(backend=pyav), what python -m kite.sources.open tello reads",
    "tello_source_opencv": "kite TelloVideoSource(backend=opencv)",
    "djitellopy": "djitellopy 2.5 BackgroundFrameRead (av.open(url, timeout=(FRAME_GRAB_TIMEOUT, None)))",
    "parser": "raw UDP socket (8 MB SO_RCVBUF) + av.CodecContext('h264').parse()/decode()",
    "parser_showall": "parser + decoder flags2=+showall, flags=+output_corrupt (frames before a keyframe)",
    "parser_smallbuf":"raw UDP socket, default SO_RCVBUF, reader thread busy 30 ms per frame",
}
FAST_OPTS = "fflags;nobuffer|flags;low_delay|probesize;32768|analyzeduration;0|fpsprobesize;0"
# format detection may read 10 MB (default 1 MiB) and stream-info probing stops at 1 s of stream / 2 MB
PROBE_OPTS = "fflags;nobuffer|flags;low_delay|formatprobesize;10000000|probesize;2000000|analyzeduration;1000000"
BUF_QUERY ="?buffer_size=4194304&fifo_size=500000&overrun_nonfatal=1"

CONDITIONS = {
    # name: (sim args, description)
    "clean": ([], "IDR every 1 s, no loss, SPS/PPS inline"),
    "keyint5s": (["--keyint", "150"], "IDR every 5 s"),
    "keyint8s": (["--keyint", "240", "--bitrate", "1M"], "IDR every 8 s, 1 Mbit/s"),
    "rare_idr_params1s": (["--keyint", "240", "--bitrate", "1M", "--params-every", "30"],
                          "IDR every 8 s but SPS+PPS every 1 s"),
    "loss1": (["--loss", "0.01"], "1 % random datagram loss"),
    "loss5": (["--loss", "0.05"], "5 % random datagram loss"),
    "loss10": (["--loss", "0.10"], "10 % random datagram loss"),
    "loss20": (["--loss", "0.20"], "20 % random datagram loss"),
    "loss35": (["--loss", "0.35"], "35 % random datagram loss"),
    "tail8": (["--burst-loss", "8"], "only the first 8 datagrams of each frame arrive (burst tail dropped)"),
    "tail3": (["--burst-loss", "3"], "only the first 3 datagrams of each frame arrive"),
    "bigloss50": (["--big-loss", "0.5"], "50 % loss inside keyframe bursts only, P frames intact"),
    "bigloss50_sep": (["--big-loss", "0.5", "--params", "separate"], "bigloss50 with SPS/PPS in their own datagrams"),
    "nonidr": (["--nonidr", "--start-offset", "1.5"], "keyframes are non-IDR I frames (SPS+PPS before each, no SEI)"),
    "params_sep": (["--params", "separate"], "SPS and PPS in their own datagrams"),
    "lowrate": (["--bitrate", "1M"], "1 Mbit/s (IDR ~25 KB, the size in the drone's log)"),
    "intra_refresh": (["--intra-refresh"], "periodic intra refresh, a single IDR at the start of the file"),
    "intra_refresh_nosei": (["--intra-refresh", "--strip-sei"], "intra refresh without recovery-point SEI"),
    "outage": (["--outage", "1000:150"], "150 ms with every datagram lost, once a second (Wi-Fi off-channel)"),
    "outage_heavy": (["--outage", "500:200"], "200 ms lost every 500 ms"),
}


# ------------------------------------------------------------------ child: one reader
def _log_to_stderr() -> None:
    try:
        import av

        av.logging.set_level(av.logging.VERBOSE)  # surface FFmpeg's own messages (PyAV mutes them by default)
    except Exception:
        pass


def _poll_seq(read_seq, seconds: float, t_start: float) -> dict:
    first, n, last = None, 0, -1
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        seq = read_seq()
        if seq is not None and seq != last and seq >= 0:
            if first is None:
                first = time.monotonic() - t_start
            n += 1
            last = seq
        time.sleep(0.002)
    return {"first_frame_s": None if first is None else round(first, 2), "frames": n}


def run_reader(name: str, port: int, seconds: float) -> dict:
    url = f"udp://@0.0.0.0:{port}"
    sys.path.insert(0, str(ROOT))
    t0 = time.monotonic()
    res: dict = {"reader": name}
    if name.startswith("opencv"):
        if name == "opencv_noopts":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""
        elif name.startswith("opencv_fast"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FAST_OPTS
        elif name.startswith("opencv_probe"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = PROBE_OPTS
        import djitellopy  # noqa: F401  (drone_test imports it first: both FFmpeg builds loaded, like the real run)
        from kite.sources.video_stream import VideoStream

        ms = 15000 if name.endswith("15s") else 5000
        rd = 15000 if name.endswith("15s") else 3000
        u = url + (BUF_QUERY if name.endswith("_buf") else "")
        try:
            s = VideoStream(u, fps=60, open_timeout_ms=ms, read_timeout_ms=rd).start()
        except RuntimeError as e:
            res.update(open_s=round(time.monotonic() - t0, 2), error=str(e), frames=0, first_frame_s=None)
            return res
        res["open_s"] = round(time.monotonic() - t0, 2)
        res.update(_poll_seq(lambda: s.read_seq(copy=False)[0], seconds, t0))
        s.stop()
    elif name in ("pyav_kite", "pyav_buf"):
        _log_to_stderr()
        from kite.sources.pyav_stream import PyAVStream

        try:
            s = PyAVStream(url + (BUF_QUERY if name == "pyav_buf" else ""), fps=60).start()
        except RuntimeError as e:
            res.update(open_s=round(time.monotonic() - t0, 2), error=str(e), frames=0, first_frame_s=None)
            return res
        res["open_s"] = round(time.monotonic() - t0, 2)
        res.update(_poll_seq(lambda: s.read_seq(copy=False)[0], seconds, t0))
        s.stop()
    elif name == "pyav_noprobe":
        _log_to_stderr()
        import av

        c = av.open(url, format="h264", timeout=(15, 15), options={"probesize": "32768", "analyzeduration": "0",
                                                                   "fpsprobesize": "0", "flags": "low_delay"})
        res["open_s"] = round(time.monotonic() - t0, 2)
        box = {"seq": -1}

        def loop():
            try:
                for pkt in c.demux(video=0):
                    try:
                        for _f in pkt.decode():
                            box["seq"] += 1
                    except av.error.FFmpegError:
                        pass
            except Exception:
                pass

        threading.Thread(target=loop, daemon=True).start()
        res.update(_poll_seq(lambda: box["seq"], seconds, t0))
    elif name.startswith("tello_source"):
        if name.endswith("pyav"):
            _log_to_stderr()
        import djitellopy  # noqa: F401
        from kite.sources.opencv_sources import TelloVideoSource

        src = TelloVideoSource(url, backend=name.rsplit("_", 1)[1]).start()

        def rs():
            f = src.read()
            return None if f is None else f.seq

        res["open_s"] = None
        res.update(_poll_seq(rs, seconds, t0))
        res["reconnects"] = src.reconnects
        src.stop()
    elif name == "djitellopy":
        _log_to_stderr()
        from djitellopy.tello import BackgroundFrameRead

        try:
            b = BackgroundFrameRead(None, url)
        except Exception as e:
            res.update(open_s=round(time.monotonic() - t0, 2), error=repr(e), frames=0, first_frame_s=None)
            return res
        res["open_s"] = round(time.monotonic() - t0, 2)
        b.start()
        res.update(_poll_seq(lambda: id(b.frame) if b.frame.shape[0] == 720 else None, seconds, t0))
        b.stop()
    elif name.startswith("parser"):
        import av

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if name != "parser_smallbuf":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.5)
        res["rcvbuf"] = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        codec = av.CodecContext.create("h264", "r")
        if name == "parser_showall":  # output frames before any keyframe (gray where references are missing)
            codec.options = {"flags2": "+showall", "flags": "+output_corrupt"}
        first, n, errs, got = None, 0, 0, 0
        busy = name == "parser_smallbuf"
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            got += 1
            try:
                for pkt in codec.parse(data):
                    try:
                        frames = codec.decode(pkt)
                    except av.error.FFmpegError:
                        errs += 1
                        continue
                    for _f in frames:
                        n += 1
                        if first is None:
                            first = time.monotonic() - t0
                        if busy:
                            time.sleep(0.03)  # a slow consumer: the socket buffer must absorb the bursts
            except av.error.FFmpegError:
                errs += 1
        sock.close()
        res.update(open_s=0.0, first_frame_s=None if first is None else round(first, 2), frames=n,
                   decode_errors=errs, datagrams=got)
    else:
        raise SystemExit(f"unknown reader {name}")
    return res


# ------------------------------------------------------------------ offline: a real capture, joined at several times
def analyze_capture(folder: Path, joins=(0.0, 1.0, 2.5, 4.0, 6.0, 9.0)) -> list[dict]:
    """Feed a kite.video_doctor capture, datagram by datagram, to the H.264 parser+decoder as a live reader that
    starts listening `join` seconds into the capture would see it. Default decoder vs showall/output_corrupt.
    Reports the capture time of the first decoded frame, frames, and the NAL types seen (SPS/PPS/IDR/I timing)."""
    import av

    sys.path.insert(0, str(ROOT / "scripts"))
    import tello_video_sim as sim

    rows = sim.load_capture(folder)
    out = []
    for showall in (False, True):
        for join in joins:
            codec = av.CodecContext.create("h264", "r")
            if showall:
                codec.options = {"flags2": "+showall", "flags": "+output_corrupt"}
            first, n, errs = None, 0, 0
            for t, d in rows:
                if t < join:
                    continue
                try:
                    for pkt in codec.parse(d):
                        try:
                            for _f in codec.decode(pkt):
                                n += 1
                                if first is None:
                                    first = t
                        except av.error.FFmpegError:
                            errs += 1
                except av.error.FFmpegError:
                    errs += 1
            out.append({"decoder": "showall" if showall else "default", "join_s": join,
                        "first_frame_after_join_s": None if first is None else round(first - join, 2),
                        "frames": n, "errors": errs})
    # NAL timeline: when SPS, PPS, IDR and non-IDR I slices arrive
    tl, seen = [], 0
    for t, d in rows:
        for off in sim.nal_starts(d):
            hdr = d.find(b"\x00\x00\x01", off) + 3
            if hdr >= len(d):
                continue
            typ = d[hdr] & 0x1F
            if typ in (5, 7, 8) or (typ == 1 and sim.slice_type(d[off:]) == 2):
                name = {5: "IDR", 7: "SPS", 8: "PPS", 1: "I(non-IDR)"}[typ]
                if not any(n == name and t - t0 < 0.1 for t0, n in tl[-4:]):  # one entry per frame, not per slice
                    tl.append((round(t, 3), name))
        seen += 1
    out.append({"nal_timeline_first_60": tl[:60], "datagrams": seen})
    return out


# ------------------------------------------------------------------ parent: the matrix
def scan(stderr: str) -> list[str]:
    return [k for k, rx in SIGNATURES.items() if re.search(rx, stderr)]


def one(reader: str, cond: str, port: int, seconds: float, replay: Path | None, sim_extra: list[str],
        lead_s: float = 1.0) -> dict:
    env = dict(os.environ)
    sim_args = list(CONDITIONS[cond][0]) if cond in CONDITIONS else []
    cmd = [sys.executable, str(SIM), "--port", str(port)] + (["--replay", str(replay)] if replay else sim_args) + sim_extra
    sim = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        sim.stdout.readline()  # "streaming ..." once the stream is encoded and running
        time.sleep(lead_s)  # the drone is already streaming when the reader opens
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reader", reader, "--port", str(port),
                            "--seconds", str(seconds)], capture_output=True, text=True, timeout=seconds + 60, env=env,
                           cwd=str(ROOT))
    finally:
        sim.terminate()
        sim.wait()
    out = {"reader": reader, "condition": cond, "returncode": p.returncode}
    m = re.search(r"^RESULT (.*)$", p.stdout, re.M)
    if m:
        out.update(json.loads(m.group(1)))
    else:
        out["error"] = f"no result (rc {p.returncode}): {p.stderr.strip().splitlines()[-1:] if p.stderr.strip() else ''}"
    out["signatures"] = scan(p.stderr)
    out["stderr_tail"] = [ln for ln in p.stderr.splitlines() if "objc[" not in ln][-6:]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reader", help="child mode: run this one reader")
    ap.add_argument("--port", type=int, default=11190)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--readers", default=",".join(READERS))
    ap.add_argument("--conditions", default="clean,keyint5s,loss1,loss5,lowrate")
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--analyze", type=Path, help="a kite.video_doctor folder: offline decode, joining at several times")
    ap.add_argument("--sim-extra", default="", help="extra args for tello_video_sim.py, space separated")
    ap.add_argument("--json", type=Path, help="also write the rows here")
    a = ap.parse_args(argv)
    if a.reader:
        r = run_reader(a.reader, a.port, a.seconds)
        print("RESULT " + json.dumps(r), flush=True)
        return 0
    if a.analyze:
        for row in analyze_capture(a.analyze):
            print(json.dumps(row))
        return 0
    conds = ["replay"] if a.replay else a.conditions.split(",")
    rows = []
    for cond in conds:
        for i, reader in enumerate(a.readers.split(",")):
            r = one(reader, cond, a.port + i % 7, a.seconds, a.replay, a.sim_extra.split())
            rows.append(r)
            ff = r.get("first_frame_s")
            print(f"{cond:13s} {reader:20s} open {str(r.get('open_s')):>6s} s  first frame "
                  f"{'-' if ff is None else f'{ff:.2f} s':>8s}  frames {r.get('frames', 0):4d}  "
                  f"{r.get('error', '')[:60]}  {','.join(r['signatures'])}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
