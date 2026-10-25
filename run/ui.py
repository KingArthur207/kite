"""The app window (Tkinter, dark): camera with the person boxes on the left, Mapbox on the right, and a
console underneath that shows every command sent and every reply received and takes the next command.

Tk runs on the main thread (macOS only draws windows there); the pilot's control thread never touches Tk. The UI
polls the pilot every ~33 ms: new console lines, the newest camera frame, the status line, and the phone's
newest position from SpacetimeDB.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk

from map_pane import MapPane
from nav_signal import NavSignal, nav_phrase
from whereabouts import Whereabouts, format_fix

# palette: dark, quiet, one warm accent
BG = "#1a1918"  # window
PANEL = "#222120"  # panels
BORDER = "#2e2c2a"
TEXT = "#e8e6e1"
MUTED = "#8a8782"
FAINT = "#5c5a56"
ACCENT = "#d97757"  # sent commands, the locked person
OK = "#7fb28a"  # replies from the drone
WARN = "#d6b25e"
ERROR = "#e5675e"
KIND_COLOURS = {"sent": ACCENT, "recv": OK, "info": MUTED, "warn": WARN, "error": ERROR}
ACCENT_BGR = (87, 119, 217)
OTHER_BGR = (130, 128, 125)

# pilot states in which the drone may be in the air
AIRBORNE = ("taking off", "climbing", "following", "searching", "holding", "landing")

STATE_COLOURS = {"following": OK, "climbing": WARN, "searching": WARN, "holding": WARN, "taking off": WARN, "landing": WARN, "emergency": ERROR,
                 "offline": ERROR, "live": OK}


def _fonts(root):
    families = set(tkfont.families(root))

    def pick(*names, default):
        return next((n for n in names if n in families), default)

    ui = pick("SF Pro Text", "SF Pro", "Segoe UI", "Inter", "Helvetica Neue", default="Helvetica")
    mono = pick("SF Mono", "Menlo", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", default="Courier")
    return ui, mono


def draw_people(img, seen, follow) -> None:
    """Every person YOLO sees (thin, grey) and the one being followed (thick, accent), drawn in place on img."""
    track = follow.snapshot() if follow is not None else None
    for d in (seen.last if seen is not None and time.time() - seen.last_t < 1.0 else []):
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), OTHER_BGR, 1, cv2.LINE_AA)
    if track is not None and track.confirmed and time.time() - track.t < 1.0:
        x1, y1, x2, y2 = (int(v) for v in track.box)
        locked = follow.active
        col = ACCENT_BGR if locked else (94, 178, 214)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        label = "following" if locked else "person in front"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        y = max(y1 - 8, th + 8)
        cv2.rectangle(img, (x1, y - th - 8), (x1 + tw + 12, y + 4), col, -1)
        cv2.putText(img, label, (x1 + 6, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (24, 24, 26), 1, cv2.LINE_AA)


def _dpi_aware() -> None:
    """Windows: sharp text with display scaling (must run before the Tk window exists)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


class App:
    def __init__(self, pilot_factory, title: str = "kite", *, opening: str | None = None, hint: str | None = None,
                 close_note: str | None = None, shutdown_note: str | None = None):
        _dpi_aware()
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1440, int(sw * 0.92)), min(900, int(sh * 0.88))
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
        self.root.minsize(min(900, w), min(600, h))
        self.ui_font, self.mono_font = _fonts(self.root)
        self.lines: queue.Queue = queue.Queue()
        self.history: list[str] = []
        self.hist_i = 0
        self._photo = None
        self._shown_seq = None
        self._closing = False
        self.whereabouts = None
        self.nav = NavSignal()
        self._route_mark = None
        self._gps_sig = None
        self._gps_logged = 0.0
        self._gps_paint = None
        self._gps_paint_at = 0.0
        self.opening = opening
        self.hint_text = hint or "destination: place  ·  find me …  ·  takeoff · land · e"
        self.close_note = close_note
        self.shutdown_note = shutdown_note
        self._build()
        self.pilot = pilot_factory(self.log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda e: self.entry.focus_set())

    # ------------------------------------------------------------------ layout
    def _panel(self, parent, title):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER, bd=0)
        head = tk.Frame(frame, bg=PANEL)
        head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(head, text=title, bg=PANEL, fg=MUTED, font=(self.ui_font, 11)).pack(side="left")
        return frame, head

    def _build(self):
        console = self._build_console()
        console.pack(side="bottom", fill="x", padx=14, pady=(7, 14))
        top = tk.Frame(self.root, bg=BG)
        top.pack(side="top", fill="both", expand=True, padx=14, pady=(14, 7))
        top.columnconfigure(0, weight=1, uniform="pane")
        top.columnconfigure(1, weight=1, uniform="pane")
        top.rowconfigure(0, weight=1)

        cam_panel, cam_head = self._panel(top, "camera")
        cam_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        map_panel, _ = self._panel(top, "map")
        map_panel.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.map_pane = MapPane(map_panel)
        self.map_pane.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.status = tk.Label(cam_head, text="", bg=PANEL, fg=FAINT, font=(self.mono_font, 10))
        self.status.pack(side="right")
        self.state_label = tk.Label(cam_head, text="starting", bg=PANEL, fg=MUTED, font=(self.ui_font, 11, "bold"))
        self.state_label.pack(side="right", padx=(0, 12))
        self.cam = tk.Label(cam_panel, bg="#121110", fg=FAINT, text="waiting for video", font=(self.ui_font, 12))
        self.cam.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _build_console(self):
        console, _ = self._panel(self.root, "console")
        row = tk.Frame(console, bg=PANEL)
        row.pack(side="bottom", fill="x", padx=14, pady=(6, 12))  # the input row first: it never gets squeezed
        lines = 11 if self.root.winfo_screenheight() >= 900 else 7
        self.text = tk.Text(console, height=lines, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0,
                            highlightthickness=0, wrap="word", font=(self.mono_font, 12), padx=4, pady=2)
        self.text.pack(side="top", fill="both", expand=True, padx=14)
        for kind, col in KIND_COLOURS.items():
            self.text.tag_configure(kind, foreground=col)
        self.text.tag_configure("time", foreground=FAINT)
        self.text.configure(state="disabled")
        tk.Label(row, text="›", bg=PANEL, fg=ACCENT, font=(self.mono_font, 15)).pack(side="left", padx=(2, 8))
        self.entry = tk.Entry(row, bg="#2a2826", fg=TEXT, insertbackground=ACCENT, relief="flat", bd=0,
                              highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
                              font=(self.mono_font, 13))
        self.entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Up>", lambda e: self._recall(-1))
        self.entry.bind("<Down>", lambda e: self._recall(+1))
        tk.Label(row, text=self.hint_text, bg=PANEL, fg=FAINT,
                 font=(self.ui_font, 11)).pack(side="right", padx=(12, 2))
        self.entry.focus_set()
        return console

    # ------------------------------------------------------------------ console
    def log(self, text: str, kind: str = "info") -> None:
        """Any thread: queue a console line (the UI thread writes it)."""
        self.lines.put((time.strftime("%H:%M:%S"), text, kind))

    def _flush_lines(self):
        wrote = False
        while True:
            try:
                stamp, text, kind = self.lines.get_nowait()
            except queue.Empty:
                break
            if not wrote:
                self.text.configure(state="normal")
                wrote = True
            prefix = {"sent": "› ", "recv": "← "}.get(kind, "  ")
            self.text.insert("end", f"{stamp}  ", "time")
            self.text.insert("end", f"{prefix}{text}\n", kind)
            print(f"{stamp} {prefix}{text}", flush=True)  # also in the terminal that started the app
        if wrote:
            self.text.see("end")
            self.text.configure(state="disabled")

    def _submit(self, event=None):
        text = self.entry.get().strip()
        self.entry.delete(0, "end")
        if not text:
            return
        self.history.append(text)
        self.hist_i = len(self.history)
        cmd = text.lower()
        if self._try_nav(text):
            return
        self.pilot.send(text)
        if cmd in ("e", "emergency"):
            self.root.after(300, self._after_emergency)
        elif cmd == "quit":
            self._on_close()

    def _try_nav(self, text: str) -> bool:
        """destination: and find me go to the walking route. A number answers which place. Drone words do not."""
        choice = text.strip()
        if self.nav.waiting_for_choice() and choice.isascii() and choice.isdigit() and int(choice) > 0:
            self.log(text, "sent")
            self.nav.submit(choice, self.log)
            return True
        phrase = nav_phrase(text)
        if phrase is None:
            return False
        self.log(text, "sent")
        if not phrase:
            self.log("Where would you like to go?", "error")
            return True
        self.nav.submit(phrase, self.log)
        return True

    def _after_emergency(self):
        if self.pilot.state == "emergency":  # sent: motors are off, quit
            self._quit(land=False)
        # not sent (Wi-Fi down): stay open so e can be typed again; the pilot keeps the drone under control

    def _recall(self, step):
        if not self.history:
            return "break"
        self.hist_i = max(0, min(len(self.history), self.hist_i + step))
        self.entry.delete(0, "end")
        if self.hist_i < len(self.history):
            self.entry.insert(0, self.history[self.hist_i])
        return "break"

    # ------------------------------------------------------------------ refresh
    def _refresh(self):
        if self._closing:
            return
        try:
            self._refresh_once()
        except Exception as e:  # one bad refresh must never freeze the window
            print(f"ui refresh: {e!r}", file=sys.stderr, flush=True)
        finally:
            if not self._closing:
                self.root.after(33, self._refresh)

    def _refresh_once(self):
        self._flush_lines()
        p = self.pilot
        s = p.snap
        self.state_label.configure(text=s.state, fg=STATE_COLOURS.get(s.state, MUTED))
        bits = []
        if s.battery is not None:
            bits.append(f"bat {s.battery:.0f}%")
        if s.height_m is not None:
            bits.append(f"h {s.height_m:.2f} m")
        bits.append(f"video {s.video_fps:.0f} fps")
        if s.det_fps:
            bits.append(f"{getattr(s, 'meter', None) or 'yolo'} {s.det_fps:.0f} fps")
        if s.yaw:
            bits.append(f"steer {s.yaw}")
        if s.link_error:
            bits.append("WI-FI DOWN")
        self.status.configure(text="   ".join(bits), fg=ERROR if s.link_error else FAINT)
        self._show_gps()
        self._show_route()
        f = p.last_frame
        if f is not None and f.seq != self._shown_seq:  # redraw only when a new frame arrived
            self._shown_seq = f.seq
            self._show_frame(f.image)

    def _show_frame(self, image):
        w, h = self.cam.winfo_width(), self.cam.winfo_height()
        if w < 20 or h < 20:
            return
        img = image.copy()  # frames are shared and read-only: draw on a copy
        draw_people(img, self.pilot.seen, self.pilot.follow)
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        rgb = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.cam.configure(image=self._photo, text="")

    def _show_gps(self):
        """The phone's newest SpacetimeDB position. Map and console stay quiet until it changes."""
        if self.whereabouts is None:
            return
        fix = self.whereabouts.latest()
        if not fix:
            return
        now = time.monotonic()
        paint = (round(fix["lat"], 5), round(fix["lon"], 5), fix.get("accuracy_m"), int(fix.get("age_s") or 0))
        if paint != self._gps_paint and now - self._gps_paint_at >= 1.0:
            self._gps_paint = paint
            self._gps_paint_at = now
            self.map_pane.set_fix(fix["lat"], fix["lon"], fix.get("accuracy_m"), fix.get("age_s"))
            self.nav.note_fix(fix["lat"], fix["lon"])
        sig = (round(fix["lat"], 5), round(fix["lon"], 5), bool(fix.get("accepted")))
        accepted_now = bool(fix.get("accepted")) and not (self._gps_sig and self._gps_sig[2])
        if sig != self._gps_sig and (self._gps_sig is None or accepted_now or now - self._gps_logged >= 5.0):
            self._gps_sig = sig
            self._gps_logged = now
            self.log(format_fix(fix), "recv" if fix.get("accepted") else "info")

    def _show_route(self):
        """Draw a new walking path when a destination or a find-me search resolves."""
        snap = self.nav.snapshot()
        route = snap["route"]
        mark = None if not route else (len(route), route[0], route[-1], snap["say"])
        if mark == self._route_mark:
            return
        self._route_mark = mark
        self.map_pane.set_route(route)

    # ------------------------------------------------------------------ lifecycle
    def run(self) -> int:
        self.log(self.opening or "starting: connecting to the drone and loading the person detector ...", "info")
        # Ctrl+C in the terminal: land, then close (Tk would otherwise kill the process mid-air on macOS)
        signal.signal(signal.SIGINT, lambda *a: self.root.after(0, self._on_close))
        self._poll_signals()
        self.pilot.start()
        try:
            self.whereabouts = Whereabouts(self.log)
            self.whereabouts.start()
            self.log("gps: waiting for the phone", "info")
            self.nav.start()
        except Exception as exc:
            self.whereabouts = None
            self.log(f"gps: listener did not start ({exc})", "error")
        self.root.after(33, self._refresh)
        self.root.mainloop()
        return 0

    def _poll_signals(self):
        """Tk's mainloop does not run Python signal handlers by itself: wake it up every 200 ms."""
        if not self._closing:
            self.root.after(200, self._poll_signals)

    def _on_close(self):
        if self._closing:
            return
        self.log(self.close_note or "closing: landing first if it is flying ...", "info")
        self._quit(land=True)

    def _quit(self, land: bool):
        """Land first (the window keeps refreshing meanwhile), then stop the pilot off the UI thread and close."""
        if self._closing:
            return
        self._closing = True
        if land and self.pilot.state in AIRBORNE and not self.pilot.closed.is_set():
            self.pilot.request_land()
            self._wait_down(time.monotonic() + 30.0)
        else:
            self._finish()

    def _wait_down(self, deadline: float):
        try:
            self._refresh_once()  # state label, status and video stay live while it lands
        except Exception:
            pass
        if self.pilot.state in AIRBORNE and time.monotonic() < deadline and not self.pilot.closed.is_set():
            self.root.after(100, lambda: self._wait_down(deadline))
        else:
            self._finish()

    def _finish(self):
        """pilot.shutdown() can block (djitellopy retries an unreachable drone): run it off the UI thread."""
        self.log(self.shutdown_note or "closing the drone link ...", "info")
        done = threading.Event()

        if self.whereabouts is not None:
            self.whereabouts.stop()
        self.nav.stop()

        def work():
            try:
                self.pilot.shutdown(land=False)
            finally:
                done.set()

        threading.Thread(target=work, name="shutdown", daemon=True).start()
        self._wait_shutdown(done, time.monotonic() + 40.0)

    def _wait_shutdown(self, done: threading.Event, deadline: float):
        try:
            self._flush_lines()
        except Exception:
            pass
        if not done.is_set() and time.monotonic() < deadline:
            self.root.after(100, lambda: self._wait_shutdown(done, deadline))
            return
        self._flush_lines_to_stdout()
        self.root.destroy()

    def _flush_lines_to_stdout(self):
        while True:
            try:
                stamp, text, kind = self.lines.get_nowait()
            except queue.Empty:
                return
            print(f"{stamp}   {text}", file=sys.stdout, flush=True)
