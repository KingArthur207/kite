"""Record the Tello's raw video packets and diagnose why frames don't decode. No motion is ever sent.

    python -m kite.video_doctor                 # drone on, laptop on its Wi-Fi, nothing else using the Tello ports
    python -m kite.video_doctor --seconds 20    # longer capture
    python -m kite.video_doctor --video-bitrate 2 --video-resolution low   # measure a lighter stream

Ends with one link line (GOOD / FAIR / POOR: packet rate, gaps, loss estimated from the H.264 stream), on any OS.
On Windows it also lists every Wi-Fi adapter (netsh) and warns when two of them sit on 2.4 GHz.

Talks to the drone over plain UDP sockets (no djitellopy, no FFmpeg network code): `command`, `streamon` (re-sent
until video arrives), `streamoff` at the end. Every video datagram is stored with its arrival time, so the exact
stream can be replayed offline to every decoder. Output folder (default recordings/video_doctor_<stamp>/):

    video.h264     the datagrams' payloads, concatenated (raw H.264 Annex B as the drone sent it)
    packets.csv    one row per datagram: t (s since the first one), size
    summary.json   rates, sizes, gaps, H.264 structure (SPS/PPS/IDR timing) and offline decode results
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import add_video_link_args

NAL_NAMES = {1: "slice", 5: "IDR", 6: "SEI", 7: "SPS", 8: "PPS", 9: "AUD"}
HEALTHY_KB_S = 100  # a Tello streaming normally sends well over this; the starved Mac link gave ~32 KB/s


def _cmd(args: list[str], timeout: float = 40.0) -> str:
    import subprocess

    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def parse_netsh_interfaces(text: str) -> list[dict]:
    """`netsh wlan show interfaces` -> one {key: value} per Wi-Fi adapter (English Windows key names)."""
    out: list[dict] = []
    for line in text.splitlines():
        key, sep, value = line.partition(" : ")
        key = key.strip()
        if not sep or not key:
            continue
        if key == "Name":
            out.append({})
        if out:
            out[-1][key] = value.strip()
    return out


def windows_link_snapshot() -> dict:
    """Windows: every Wi-Fi adapter's SSID, channel, band, radio type, signal and rates (netsh, no scan, ~0.1 s)."""
    itfs = parse_netsh_interfaces(_cmd(["netsh", "wlan", "show", "interfaces"], timeout=10))
    keep = ("Name", "Description", "State", "SSID", "Radio type", "Band", "Channel", "Receive rate (Mbps)",
            "Transmit rate (Mbps)", "Signal")
    return {"wifi_interfaces": [{k: i[k] for k in keep if k in i} for i in itfs]} if itfs else {}


def shared_24ghz(link: dict) -> list[str]:
    """Names of the connected Windows adapters on 2.4 GHz (channel 1-14): two of them share the band."""
    names = []
    for i in link.get("wifi_interfaces", []):
        ch = i.get("Channel", "")
        if i.get("State", "").lower() == "connected" and ch.isdigit() and int(ch) <= 14:
            names.append(f"{i.get('Name')} ({i.get('SSID')}, channel {ch})")
    return names


def link_snapshot() -> dict:
    """macOS: the Wi-Fi link to the drone (channel, PHY mode, rate, MCS, signal/noise), Bluetooth devices connected
    (they share the 2.4 GHz radio) and AWDL state. Windows: every Wi-Fi adapter (windows_link_snapshot). Empty
    elsewhere. macOS is slow (~20 s): system_profiler scans for networks, which takes the radio off-channel, so it
    never runs while the video is being recorded."""
    if sys.platform == "win32":
        return windows_link_snapshot()
    if sys.platform != "darwin":
        return {}
    snap: dict = {}
    try:
        d = json.loads(_cmd(["system_profiler", "SPAirPortDataType", "-json"]) or "{}")["SPAirPortDataType"][0]
        for itf in d.get("spairport_airport_interfaces", []):
            cur = itf.get("spairport_current_network_information")
            if cur:
                snap["wifi"] = {k.replace("spairport_network_", "").replace("spairport_", ""): v
                                for k, v in cur.items() if k != "_name"}
                break
    except (ValueError, KeyError, IndexError):
        pass
    try:
        b = json.loads(_cmd(["system_profiler", "SPBluetoothDataType", "-json"]) or "{}")["SPBluetoothDataType"][0]
        snap["bluetooth_connected"] = [list(x.keys())[0] for x in b.get("device_connected", [])]
        snap["bluetooth_state"] = b.get("controller_properties", {}).get("controller_state")
    except (ValueError, KeyError, IndexError):
        pass
    awdl = _cmd(["ifconfig", "awdl0"])
    snap["awdl"] = "active" if "status: active" in awdl else ("inactive" if awdl else "none")
    return snap


class _Receiver:
    """Drains one UDP socket as fast as possible into a list of (monotonic t, bytes)."""

    def __init__(self, port: int, rcvbuf: int = 0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if rcvbuf:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self.sock.bind(("0.0.0.0", port))  # nosec B104: the Tello sends video from its own Wi-Fi network
        self.sock.settimeout(0.2)
        self.rcvbuf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.packets: list[tuple[float, bytes]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            self.packets.append((time.monotonic(), data))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.sock.close()


def nal_units(data: bytes) -> list[tuple[int, int]]:
    """(offset, nal_type) for every Annex B start code in data."""
    out, i, n = [], 0, len(data)
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= n:
            return out
        out.append((i, data[i + 3] & 0x1F))
        i += 3


def analyse_packets(packets: list[tuple[float, bytes]]) -> dict:
    if not packets:
        return {"packets": 0}
    t0 = packets[0][0]
    ts = [t - t0 for t, _ in packets]
    sizes = [len(p) for _, p in packets]
    dur = max(ts[-1], 1e-6)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    stream = b"".join(p for _, p in packets)
    # where each byte offset came from, to time the NAL units
    starts, off = [], 0
    for t, p in zip(ts, (p for _, p in packets)):
        starts.append((off, t))
        off += len(p)

    def t_at(offset: int) -> float:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid][0] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return starts[lo][1]

    nals = nal_units(stream)
    counts: dict[str, int] = {}
    for _, typ in nals:
        name = NAL_NAMES.get(typ, f"type{typ}")
        counts[name] = counts.get(name, 0) + 1
    idr_t = [round(t_at(o), 3) for o, typ in nals if typ == 5]
    sps_t = [round(t_at(o), 3) for o, typ in nals if typ == 7]
    idr_gaps = [round(b - a, 3) for a, b in zip(idr_t, idr_t[1:])]
    # does an SPS+PPS come right before every IDR?
    seq = [typ for _, typ in nals]
    idr_with_params = sum(1 for k, typ in enumerate(seq) if typ == 5 and 7 in seq[max(0, k - 4):k] and 8 in seq[max(0, k - 4):k])
    size_hist: dict[int, int] = {}
    for s in sizes:
        size_hist[s] = size_hist.get(s, 0) + 1
    common = sorted(size_hist.items(), key=lambda kv: -kv[1])[:5]
    return {
        "packets": len(packets), "seconds": round(dur, 3), "packets_per_s": round(len(packets) / dur, 1),
        "kbytes_per_s": round(sum(sizes) / dur / 1024, 1), "bytes": sum(sizes),
        "size_min": min(sizes), "size_max": max(sizes), "most_common_sizes": common,
        "max_gap_s": round(max(gaps), 3) if gaps else 0.0,
        "gaps_over_0.2s": sum(1 for g in gaps if g > 0.2),
        "starts_with_start_code": stream[:4] == b"\x00\x00\x00\x01" or stream[:3] == b"\x00\x00\x01",
        "nal_counts": counts, "first_sps_s": sps_t[0] if sps_t else None, "first_idr_s": idr_t[0] if idr_t else None,
        "idr_times_s": idr_t[:20], "idr_interval_s": idr_gaps[:20],
        "idr_with_sps_pps_before": f"{idr_with_params}/{len(idr_t)}",
    }


def decode_with_parser(path: Path) -> dict:
    """PyAV, no demuxer: feed the raw bytes through the H.264 parser like a live socket would."""
    try:
        import av
    except ImportError as e:
        return {"error": f"PyAV missing: {e}"}
    codec = av.CodecContext.create("h264", "r")
    data = path.read_bytes()
    frames = errors = 0
    first = None
    size = None
    for i in range(0, len(data), 1460):
        try:
            for pkt in codec.parse(data[i:i + 1460]):
                try:
                    for f in codec.decode(pkt):
                        frames += 1
                        size = (f.width, f.height)
                        if first is None:
                            first = i
                except av.error.FFmpegError:
                    errors += 1
        except av.error.FFmpegError:
            errors += 1
    return {"frames": frames, "decode_errors": errors, "size": size,
            "first_frame_at_byte": first, "first_frame_at_fraction": round(first / max(len(data), 1), 3) if first is not None else None}


def decode_with_raw_reader(path: Path) -> dict:
    """kite's own reader (kite.sources.tello_raw, the macOS default) fed the capture's bytes in 1460-byte chunks."""
    try:
        from .sources.tello_raw import RawTelloStream
    except ImportError as e:
        return {"error": f"raw reader unavailable: {e}"}
    data = path.read_bytes()
    s = RawTelloStream(feed=[data[i:i + 1460] for i in range(0, len(data), 1460)]).start()
    t_end = time.monotonic() + 60
    while not s.finished and time.monotonic() < t_end:
        time.sleep(0.02)
    s.stop()
    st = s.stats()
    return {k: st[k] for k in ("frames", "frames_decoded", "corrupt_frames", "decode_errors", "skipped_bytes", "resyncs", "size",
                               "last_error")}


def decode_with_demuxer(path: Path) -> dict:
    """PyAV's h264 demuxer on the file (what av.open(udp://, format='h264') does, minus the network)."""
    try:
        import av
    except ImportError as e:
        return {"error": f"PyAV missing: {e}"}
    frames = errors = 0
    try:
        with av.open(str(path), format="h264") as c:
            for pkt in c.demux(video=0):
                try:
                    frames += len(pkt.decode())
                except av.error.FFmpegError:
                    errors += 1
    except av.error.FFmpegError as e:
        return {"frames": frames, "open_error": str(e)}
    return {"frames": frames, "decode_errors": errors}


def decode_with_opencv(path: Path) -> dict:
    try:
        import cv2
    except ImportError as e:
        return {"error": f"OpenCV missing: {e}"}
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"frames": 0, "open_error": "VideoCapture did not open the file"}
    frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    cap.release()
    return {"frames": frames}


def verdict(summary: dict) -> list[str]:
    a, out = summary.get("stream", {}), []
    link = summary.get("link_after") or summary.get("link_before") or {}
    if a.get("packets") and a.get("kbytes_per_s", 0) < HEALTHY_KB_S:
        out.append(f"Only {a['kbytes_per_s']} KB/s of video arrived ({a.get('packets_per_s')} packets/s); a Tello streaming "
                   f"normally sends well over {HEALTHY_KB_S} KB/s. The radio link is starving the stream: most frames "
                   "never arrive, so what decodes is smeared. This is the link, not the decoder.")
        wifi = link.get("wifi", {})
        if wifi:
            out.append(f"Mac's link to the drone: {wifi.get('phymode')} on channel {wifi.get('channel')}, rate "
                       f"{wifi.get('rate')} Mbps, MCS {wifi.get('mcs')}, signal/noise {wifi.get('signal_noise')}.")
        if link.get("bluetooth_connected"):
            out.append(f"Bluetooth devices connected: {', '.join(link['bluetooth_connected'])}. They share the 2.4 GHz "
                       "radio with the Tello's Wi-Fi: disconnect them (or turn Bluetooth off) for the flight.")
        if link.get("awdl") == "active":
            out.append("AWDL (AirDrop/Handoff) is active: sudo ifconfig awdl0 down.")
    shared = shared_24ghz(link)
    if len(shared) > 1:
        out.append(f"Two Wi-Fi adapters are on 2.4 GHz at once: {'; '.join(shared)}. They share the band with the "
                   "Tello's video: put the phone hotspot (and the adapter on it) on 5 GHz.")
    if not a.get("packets"):
        return ["NO VIDEO PACKETS arrived: the drone never streamed to this laptop (streamon replies: "
                f"{summary.get('streamon_replies')}). Power-cycle the drone and rejoin its Wi-Fi."]
    if a.get("nal_counts", {}).get("IDR", 0) == 0:
        out.append("Packets arrived but contained NO keyframe (IDR): nothing can decode. Capture longer (--seconds 30).")
    p = summary.get("decode", {}).get("pyav_parser", {})
    if p.get("frames"):
        out.append(f"The captured stream DECODES offline ({p['frames']} frames at {p.get('size')}): the stream is fine, "
                   "so the live readers lose or mishandle it (receive buffer, FFmpeg udp:// probing, timeouts).")
    else:
        out.append("The captured stream does NOT decode offline even from the file: packets were lost on the way in "
                   "or the bitstream is unusual. Keep video.h264 for analysis.")
    if a.get("gaps_over_0.2s"):
        out.append(f"{a['gaps_over_0.2s']} gaps over 0.2 s between packets (max {a['max_gap_s']} s): weak Wi-Fi link.")
    return out


def link_line(summary: dict) -> tuple[str, str]:
    """kite.sources.tello_raw.link_verdict on the capture: packet rate and gaps as received, loss estimated from
    how many frames the raw reader had to conceal when decoding the capture offline."""
    try:
        from .sources.tello_raw import link_verdict
    except ImportError as e:
        return "?", f"link ?: raw reader unavailable ({e})"
    a, r = summary.get("stream", {}), summary.get("decode", {}).get("raw_reader", {})
    return link_verdict({"datagrams": a.get("packets", 0), "bytes": a.get("bytes", 0),
                         "receiving_s": a.get("seconds", 0.0), "max_gap_s": a.get("max_gap_s", 0.0),
                         "gaps_over_0.2s": a.get("gaps_over_0.2s", 0),
                         "frames_decoded": r.get("frames_decoded", r.get("frames")),
                         "corrupt_frames": r.get("corrupt_frames")})


def run(args) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"recordings/video_doctor_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    drone = (args.host, args.cmd_port)
    try:
        cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd.bind(("0.0.0.0", args.local_cmd_port))  # nosec B104: replies come from the Tello's network
        cmd.settimeout(5.0)
        video = _Receiver(args.video_port, rcvbuf=args.rcvbuf)
    except OSError as e:
        print(f"cannot bind the Tello ports ({e}): close every other Tello program "
              f"(macOS/Linux: lsof -nP -iUDP:{args.local_cmd_port} -iUDP:{args.video_port}; "
              f"Windows: netstat -ano -p udp | findstr \":{args.local_cmd_port} :{args.video_port}\")")
        return 2

    def send(text: str, wait: bool = True) -> str | None:
        cmd.sendto(text.encode(), drone)
        if not wait:
            return None
        try:
            data = cmd.recv(2048)
        except (socket.timeout, ConnectionResetError):  # Windows reports an ICMP "unreachable" as a reset
            return None
        try:
            return data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return "binary:" + data[:16].hex()

    if sys.platform == "darwin":
        print("reading the Wi-Fi link (about 20 s) ...")
    summary: dict = {"stamp": stamp, "host": args.host, "platform": sys.platform, "rcvbuf_bytes": video.rcvbuf,
                     "link_before": link_snapshot()}
    try:
        summary["command_reply"] = send("command")
        print(f"command -> {summary['command_reply']}")
        if summary["command_reply"] is None:
            print("no reply to 'command': is this laptop on the Tello's Wi-Fi?")
            return 1
        summary["battery"] = send("battery?")
        print(f"battery -> {summary['battery']}%")
        from .drone.tello import apply_video_link

        summary["video_link"] = apply_video_link(send, args.video_bitrate, args.video_resolution, args.video_fps,
                                                 say=print)
        replies = []
        for _ in range(args.streamon_tries):
            replies.append(send("streamon"))
            t_end = time.monotonic() + 3.0
            while time.monotonic() < t_end and not video.packets:
                time.sleep(0.05)
            if video.packets:
                break
        summary["streamon_replies"] = replies
        print(f"streamon -> {replies}; recording {args.seconds:g} s ...")
        time.sleep(args.seconds)
        summary["streamoff_reply"] = send("streamoff")
        if sys.platform == "darwin":
            print("reading the Wi-Fi link (about 20 s) ...")
        if sys.platform in ("darwin", "win32"):
            summary["link_after"] = link_snapshot()
    finally:
        video.close()
        cmd.close()

    packets = video.packets
    raw = out / "video.h264"
    raw.write_bytes(b"".join(p for _, p in packets))
    with open(out / "packets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "size"])
        t0 = packets[0][0] if packets else 0.0
        for t, p in packets:
            w.writerow([f"{t - t0:.6f}", len(p)])
    summary["stream"] = analyse_packets(packets)
    if packets:
        print("decoding the capture offline ...")
        summary["decode"] = {"pyav_parser": decode_with_parser(raw), "raw_reader": decode_with_raw_reader(raw),
                             "pyav_demuxer": decode_with_demuxer(raw),
                             "opencv_file": decode_with_opencv(raw)}
    summary["verdict"] = verdict(summary)
    summary["link_grade"], summary["link"] = link_line(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("link_before", "link_after", "stream", "decode") if k in summary},
                     indent=1, default=str))
    print("\n".join(["", "VERDICT:"] + [f"- {v}" for v in summary["verdict"]]))
    print(f"\n{summary['link']}")
    print(f"\nsaved -> {out}  (send this folder to whoever debugs the video)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.video_doctor", description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=15.0, help="how long to record the video packets (default 15)")
    p.add_argument("--out", help="output folder (default recordings/video_doctor_<stamp>)")
    p.add_argument("--host", default="192.168.10.1")
    p.add_argument("--cmd-port", type=int, default=8889, help="the drone's command port")
    p.add_argument("--local-cmd-port", type=int, default=8889, help="this laptop's command port")
    p.add_argument("--video-port", type=int, default=11111)
    p.add_argument("--rcvbuf", type=int, default=8 * 1024 * 1024, help="UDP receive buffer to ask for (bytes)")
    p.add_argument("--streamon-tries", type=int, default=4)
    add_video_link_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
