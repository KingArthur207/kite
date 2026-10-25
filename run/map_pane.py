"""Mapbox streets pane for the kite window.

ui.py owns the window chrome. This widget draws the tiles, the walking route and the GPS fix.
The four math helpers do not touch Tk. set_fix and set_route are safe from any thread: tile
bytes are fetched on a few workers, and Tk (including PhotoImage) runs only on the thread
that built the widget.

Tiles are Mapbox streets-v12 (© Mapbox, © OpenStreetMap). The token is MAPBOX_TOKEN from the
environment or the repo .env. It is never written into source.
"""

from __future__ import annotations

import io
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
import urllib.request
from collections import OrderedDict
from tkinter import font as tkfont

from PIL import Image, ImageDraw, ImageTk

# Slippy map: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
# Web Mercator uses a sphere of radius 6378137 m, so zoom 0 at the equator is ~156543 m/px.
_TILE = 256
_MIN_ZOOM = 13
_MAX_ZOOM = 18
# Follow the walker here. A destination flies closer than this, onto the place they named.
_FOLLOW_ZOOM = 16
_DEST_ZOOM = 18
_ROUTE_PAD = 36
_FLIGHT_S = 0.45
_EARTH_RADIUS_M = 6378137.0
_EARTH_CIRCUMFERENCE_M = 2.0 * math.pi * _EARTH_RADIUS_M
_MAX_LAT = math.degrees(math.atan(math.sinh(math.pi)))

_BG = "#1e1e1e"
_ROUTE = "#ff3154"
_ROUTE_WIDTH = 6
_TILE_URL = "https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/256/{z}/{x}/{y}?access_token={token}"
_USER_AGENT = "KiteWalk/1.0 (educational; local desktop)"

_WORKERS = 4
_CACHE_MAX = 256
_FAIL_RETRY_S = 20.0
_MAX_TILE_BYTES = 2_000_000
_DRAG_PX = 4


def _clamp_lat(lat: float) -> float:
    """Web Mercator ends at about ±85.05°. Past that, tan() overflows."""
    lat = float(lat)
    if not math.isfinite(lat):
        return 0.0
    if lat > _MAX_LAT:
        return _MAX_LAT
    if lat < -_MAX_LAT:
        return -_MAX_LAT
    return lat


def _finite_lng(lng: float) -> float:
    lng = float(lng)
    return lng if math.isfinite(lng) else 0.0


def world_px(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    """Pixel position in the world bitmap of 256px tiles. The origin is the northwest corner."""
    scale = _TILE * (2.0 ** int(zoom))
    x = (_finite_lng(lng) + 180.0) / 360.0 * scale
    lat_rad = math.radians(_clamp_lat(lat))
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return (x, y)


def latlng_from_world(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Inverse of world_px. x and y are world-pixel coordinates."""
    scale = _TILE * (2.0 ** int(zoom))
    lng = float(x) / scale * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(y) / scale)))
    return (math.degrees(lat_rad), lng)


def tile_xy(lat: float, lng: float, zoom: int) -> tuple[int, int]:
    """OSM tile column and row at zoom, clamped into the zoom's [0, 2**zoom) square."""
    z = int(zoom)
    n = 1 << z if z >= 0 else 1
    x, y = world_px(lat, lng, z)
    return (_clamp_index(x / _TILE, n), _clamp_index(y / _TILE, n))


def _clamp_index(index: float, n: int) -> int:
    if not math.isfinite(index):
        return 0 if index < 0 else n - 1
    return min(n - 1, max(0, int(math.floor(index))))


def _bounds_center(points, zoom: int) -> tuple[float, float]:
    xs, ys = zip(*(world_px(lat, lng, zoom) for lat, lng in points))
    return latlng_from_world((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, zoom)


def _span_fits(points, width: int, height: int, zoom: int, pad: int) -> bool:
    xs, ys = zip(*(world_px(lat, lng, zoom) for lat, lng in points))
    return (max(xs) - min(xs)) <= width - 2 * pad and (max(ys) - min(ys)) <= height - 2 * pad


def route_view(points, width: int, height: int, pad: int = _ROUTE_PAD):
    """(center, zoom) after a destination is set, or None when there is nothing to frame.

    The camera zooms in to ``_DEST_ZOOM``. A short walk stays entirely in the pane, centered
    on the route. A longer one is framed on the destination, the last point, so the place
    they named is what fills the map.
    """
    if not points or width < pad * 2 + 8 or height < pad * 2 + 8:
        return None
    zoom = min(_MAX_ZOOM, max(_MIN_ZOOM, _DEST_ZOOM))
    if _span_fits(points, width, height, zoom, pad):
        return _bounds_center(points, zoom), zoom
    return points[-1], zoom


def _same_place(a, b) -> bool:
    """True when two coordinates are the same street corner. A reroute of one destination is not a new one."""
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) < 1e-4 and abs(a[1] - b[1]) < 1e-4


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Ground metres covered by one pixel of a 256px Web Mercator tile."""
    lat = float(lat)
    if not math.isfinite(lat):
        lat = 0.0
    lat = max(-90.0, min(90.0, lat))
    return _EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (_TILE * (2.0 ** zoom))


def _format_age(age_s: float | None) -> str | None:
    """'12 s' under 90s, '4 min' under an hour, then whole hours. None if age was not given."""
    if age_s is None or not math.isfinite(age_s):
        return None
    age = max(0.0, float(age_s))
    if age < 90:
        return f"{int(age)} s"
    if age < 3600:
        return f"{int(age // 60)} min"
    return f"{int(age // 3600)} h"


def mapbox_token() -> str:
    """MAPBOX_TOKEN from the environment, or the repo-root .env. Empty when neither has it."""
    token = os.environ.get("MAPBOX_TOKEN", "").strip()
    if token:
        return token
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() == "MAPBOX_TOKEN":
                    token = value.strip().strip('"').strip("'")
                    break
    except OSError:
        token = ""
    if token:
        os.environ.setdefault("MAPBOX_TOKEN", token)
    return token


def tile_url(z: int, x: int, y: int, token: str) -> str | None:
    """One Mapbox streets tile, or None when there is no token to send."""
    token = (token or "").strip()
    if not token:
        return None
    return _TILE_URL.format(z=int(z), x=int(x), y=int(y), token=token)


def _download_tile(url: str) -> tuple[tuple[int, int], bytes] | None:
    """Fetch one tile and decode it to raw RGB. Runs off the UI thread. None on any failure."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=10) as response:
        data = response.read(_MAX_TILE_BYTES + 1)
    if not data or len(data) > _MAX_TILE_BYTES:
        return None
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        if rgb.size != (_TILE, _TILE):
            rgb = rgb.resize((_TILE, _TILE))
        return rgb.size, rgb.tobytes()


def _puck_image(master) -> ImageTk.PhotoImage:
    """The walker's mark: a red ring, a white ring, and a black center, with a soft shadow."""
    size = 46
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 14, 38, 40), fill=(0, 0, 0, 55))
    draw.ellipse((6, 4, 38, 36), fill=(255, 49, 84, 255))
    draw.ellipse((11, 9, 33, 31), fill=(255, 255, 255, 255))
    draw.ellipse((17, 15, 27, 25), fill=(24, 24, 24, 255))
    return ImageTk.PhotoImage(image, master=master)


def _font_family(widget: tk.Misc) -> str:
    try:
        families = set(tkfont.families(widget))
    except tk.TclError:
        return "TkDefaultFont"
    for name in ("Segoe UI", "SF Pro Text", "SF Pro", "Helvetica Neue", "Helvetica", "DejaVu Sans"):
        if name in families:
            return name
    return "TkDefaultFont"


class MapPane(tk.Frame):
    """Slippy map: Mapbox streets tiles, a walking route, and the latest GPS fix.

    Constructed on the Tk thread. set_fix and set_route may be called from anywhere.
    """

    def __init__(self, master):
        super().__init__(master, bg=_BG, bd=0, highlightthickness=0, takefocus=0)
        self._alive = True
        self._token = mapbox_token()
        self._ui_thread = threading.get_ident()
        self._zoom = _FOLLOW_ZOOM
        self._center: tuple[float, float] | None = None
        self._user_panned = False
        self._dest_view = False
        self._flight: dict | None = None
        self._flight_i = 0
        self._fix: tuple[float, float] | None = None
        self._accuracy: float | None = None
        self._age_s: float | None = None
        self._route: tuple[tuple[float, float], ...] | None = None
        self._tiles: OrderedDict[tuple[int, int, int], ImageTk.PhotoImage] = OrderedDict()
        self._inflight: set[tuple[int, int, int]] = set()
        self._failed: dict[tuple[int, int, int], float] = {}
        self._visible: set[tuple[int, int, int]] = set()
        self._inbox: queue.Queue = queue.Queue()
        self._jobs: queue.Queue = queue.Queue()
        self._drag: tuple[float, float, float, float] | None = None
        self._pointer = (0.0, 0.0)
        self._press_recenter = False
        self._dragging = False
        self._grabbed = False
        self._recenter_box: tuple[float, float, float, float] | None = None
        self._wheel_bound = False
        self._retry_wait = False
        self._reported = False
        self._canvas_size: tuple[int, int] | None = None
        family = _font_family(self)
        self._font = (family, 10)
        self._font_small = (family, 9)
        self._font_tiny = (family, 8)
        self._fonts: dict[tuple, tkfont.Font] = {}
        self._puck = _puck_image(self)
        self.canvas = tk.Canvas(self, bg=_BG, highlightthickness=0, bd=0, takefocus=0)
        self._bind_events()
        for i in range(_WORKERS):
            threading.Thread(target=self._tile_loop, name=f"map-tile-{i}", daemon=True).start()
        self.canvas.pack(fill="both", expand=True)
        self.after(30, self._poll)
        self.after_idle(self._redraw)

    def destroy(self) -> None:
        self._alive = False
        self._release_wheel()
        super().destroy()

    def set_fix(self, lat, lng, accuracy=None, age_s=None) -> None:
        """Show this fix. Safe from any thread. Recenters unless the user has dragged the map."""
        lat_f, lng_f = float(lat), float(lng)
        if not math.isfinite(lat_f) or not math.isfinite(lng_f):
            return
        self._post(("fix", lat_f, lng_f, _opt_float(accuracy), _opt_float(age_s)))

    def set_route(self, points) -> None:
        """Draw the walking route, a list of (lat, lng), or None to clear it. Any thread.

        A new destination zooms the map in on that place. A reroute to the same place does not
        move the camera again.
        """
        if points is None:
            copied = None
        else:
            copied_list = []
            for point in points:
                lat, lng = float(point[0]), float(point[1])
                if math.isfinite(lat) and math.isfinite(lng):
                    copied_list.append((lat, lng))
            copied = tuple(copied_list)
        self._post(("route", copied))

    def recenter(self) -> None:
        """Clear the user's pan and jump back to the fix."""
        self._post(("recenter",))

    # ------------------------------------------------------------------ threads
    def _post(self, message: tuple) -> None:
        if threading.get_ident() == self._ui_thread:
            self._handle(message)
            self._redraw()
        else:
            self._inbox.put(message)

    def _poll(self) -> None:
        if not self._alive:
            return
        try:
            self._drain()
        except tk.TclError:
            self._alive = False
            return
        except Exception as exc:
            # One bad tile must not stop the map. Say so once; the app's own refresh keeps going.
            if not self._reported:
                self._reported = True
                print(f"map pane: {exc!r}", file=sys.stderr, flush=True)
        if not self._alive:
            return
        try:
            self.after(30, self._poll)
        except tk.TclError:
            self._alive = False

    def _drain(self) -> None:
        dirty = False
        while True:
            try:
                message = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._handle(message)
            dirty = True
        if dirty:
            self._redraw()

    def _handle(self, message: tuple) -> None:
        kind = message[0]
        if kind == "fix":
            _, lat, lng, accuracy, age_s = message
            self._fix = (lat, lng)
            self._accuracy = accuracy
            self._age_s = age_s
            if not self._user_panned:
                self._center = (lat, lng)
        elif kind == "route":
            previous = self._route[-1] if self._route else None
            self._route = message[1]
            end = self._route[-1] if self._route else None
            if end is not None and not _same_place(previous, end):
                self._dest_view = True
            elif end is None:
                self._cancel_flight()
        elif kind == "recenter":
            self._cancel_flight()
            self._dest_view = False
            self._user_panned = False
            self._drag = None
            self._dragging = False
            if self._fix is not None:
                self._center = self._fix
        elif kind == "tile":
            self._store_tile(message[1], message[2])

    def _tile_loop(self) -> None:
        while True:
            try:
                key = self._jobs.get()
            except Exception:
                return
            payload = None
            try:
                z, x, y = key
                url = tile_url(z, x, y, self._token)
                payload = _download_tile(url) if url else None
            except Exception:
                payload = None
            try:
                self._inbox.put(("tile", key, payload))
            except Exception:
                return

    def _request_tile(self, key: tuple[int, int, int]) -> None:
        if key in self._tiles or key in self._inflight:
            return
        retry_at = self._failed.get(key)
        if retry_at is not None and time.monotonic() < retry_at:
            return
        self._inflight.add(key)
        self._jobs.put(key)

    def _store_tile(self, key: tuple[int, int, int], payload) -> None:
        self._inflight.discard(key)
        if key in self._tiles:
            return
        if payload is None:
            self._mark_failed(key)
            return
        try:
            size, raw = payload
            image = Image.frombytes("RGB", size, raw)
            photo = ImageTk.PhotoImage(image, master=self)
        except Exception:
            self._mark_failed(key)
            return
        self._failed.pop(key, None)
        # The canvas does not keep a PhotoImage alive. This dict does.
        self._tiles[key] = photo
        self._tiles.move_to_end(key)

    def _mark_failed(self, key: tuple[int, int, int]) -> None:
        self._failed[key] = time.monotonic() + _FAIL_RETRY_S
        if len(self._failed) > 512:
            self._failed.clear()
            self._failed[key] = time.monotonic() + _FAIL_RETRY_S
        if self._retry_wait or not self._alive:
            return
        self._retry_wait = True
        try:
            self.after(int(_FAIL_RETRY_S * 1000), self._retry_tiles)
        except tk.TclError:
            self._alive = False

    def _retry_tiles(self) -> None:
        self._retry_wait = False
        if not self._alive:
            return
        try:
            self._redraw()
        except tk.TclError:
            self._alive = False

    # ------------------------------------------------------------------ input
    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<Configure>", self._on_configure)
        c.bind("<Enter>", self._on_enter)
        c.bind("<Leave>", self._on_leave)
        c.bind("<Button-4>", self._on_wheel_up)
        c.bind("<Button-5>", self._on_wheel_down)
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Motion>", self._on_motion)
        self.bind("<Destroy>", self._on_destroy)

    def _on_configure(self, event) -> None:
        if event.widget is not self.canvas:
            return
        size = (int(event.width), int(event.height))
        if size == self._canvas_size or size[0] < 2 or size[1] < 2:
            return
        self._canvas_size = size
        self._redraw()

    def _on_enter(self, _event) -> None:
        # Wheel events go to the focused widget on Windows. Bind only while the pointer is here
        # so the console entry keeps the keyboard.
        if self._wheel_bound:
            return
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._wheel_bound = True

    def _on_leave(self, _event) -> None:
        if self._dragging:
            return
        self._release_wheel()

    def _release_wheel(self) -> None:
        if not self._wheel_bound:
            return
        self._wheel_bound = False
        try:
            self.canvas.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass

    def _on_wheel(self, event):
        steps = _wheel_steps(int(event.delta))
        if steps:
            self._zoom_by(steps, self._zoom_anchor())
        return "break"

    def _on_wheel_up(self, _event) -> None:
        self._zoom_by(1, self._zoom_anchor())

    def _on_wheel_down(self, _event) -> None:
        self._zoom_by(-1, self._zoom_anchor())

    def _zoom_anchor(self) -> tuple[float, float] | None:
        c = self.canvas
        px = c.winfo_pointerx() - c.winfo_rootx()
        py = c.winfo_pointery() - c.winfo_rooty()
        width, height = c.winfo_width(), c.winfo_height()
        if width < 2 or height < 2 or not (0 <= px < width and 0 <= py < height):
            return None
        return (px, py)

    def _zoom_by(self, steps: int, anchor: tuple[float, float] | None) -> None:
        self._cancel_flight()
        new = min(_MAX_ZOOM, max(_MIN_ZOOM, self._zoom + steps))
        if new == self._zoom:
            return
        # While the map is following the fix, keep the fix in the middle. Once the user has
        # dragged, keep the place under the pointer from sliding away.
        if self._center is not None and self._user_panned and anchor is not None:
            px, py = anchor
            width = max(self.canvas.winfo_width(), 1)
            height = max(self.canvas.winfo_height(), 1)
            origin_x, origin_y = self._origin(width, height)
            lat, lng = latlng_from_world(origin_x + px, origin_y + py, self._zoom)
            self._zoom = new
            ax, ay = world_px(lat, lng, new)
            self._center = latlng_from_world(ax - px + width / 2.0, ay - py + height / 2.0, new)
        else:
            self._zoom = new
        # A drag stores world pixels at the zoom it started at. Re-pin it so the next
        # motion doesn't apply that old offset again.
        if self._drag is not None and self._center is not None:
            cwx, cwy = world_px(self._center[0], self._center[1], self._zoom)
            px, py = anchor if anchor is not None else self._pointer
            self._drag = (px, py, cwx, cwy)
        self._redraw()

    def _on_press(self, event) -> None:
        self._press_recenter = self._hit_recenter(event.x, event.y)
        self._pointer = (event.x, event.y)
        self._dragging = False
        try:
            self.canvas.grab_set()
            self._grabbed = True
        except tk.TclError:
            self._grabbed = False
        if self._press_recenter or self._center is None:
            self._drag = None
            return
        wx, wy = world_px(self._center[0], self._center[1], self._zoom)
        self._drag = (event.x, event.y, wx, wy)

    def _on_drag(self, event) -> None:
        if self._drag is None or self._press_recenter:
            return
        self._pointer = (event.x, event.y)
        x0, y0, wx, wy = self._drag
        dx, dy = event.x - x0, event.y - y0
        if not self._dragging and abs(dx) < _DRAG_PX and abs(dy) < _DRAG_PX:
            return
        self._dragging = True
        self._cancel_flight()
        self._user_panned = True
        self.canvas.configure(cursor="fleur")
        self._center = latlng_from_world(wx - dx, wy - dy, self._zoom)
        self._redraw()

    def _on_release(self, event) -> None:
        if self._grabbed:
            self._grabbed = False
            try:
                self.canvas.grab_release()
            except tk.TclError:
                pass
        self._dragging = False
        self._drag = None
        if self._press_recenter:
            self._press_recenter = False
            if self._hit_recenter(event.x, event.y):
                self.recenter()
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if not (0 <= event.x < width and 0 <= event.y < height):
            # Leave is swallowed while the pointer is grabbed, so drop the wheel bind here.
            self._release_wheel()
        self._set_cursor(event.x, event.y)

    def _on_motion(self, event) -> None:
        if self._dragging:
            return
        self._set_cursor(event.x, event.y)

    def _set_cursor(self, x: int, y: int) -> None:
        self.canvas.configure(cursor="hand2" if self._hit_recenter(x, y) else "")

    def _hit_recenter(self, x: float, y: float) -> bool:
        box = self._recenter_box
        if box is None:
            return False
        x1, y1, x2, y2 = box
        return x1 - 2 <= x <= x2 + 2 and y1 - 2 <= y <= y2 + 2

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        self._alive = False
        self._release_wheel()

    # ------------------------------------------------------------------ drawing
    def _origin(self, width: int, height: int) -> tuple[float, float]:
        cx, cy = world_px(self._center[0], self._center[1], self._zoom)
        return cx - width / 2.0, cy - height / 2.0

    def _redraw(self) -> None:
        if not self._alive:
            return
        c = self.canvas
        try:
            width, height = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            self._alive = False
            return
        self._frame_destination()
        c.delete("all")
        self._recenter_box = None
        self._visible = set()
        if width >= 2 and height >= 2 and self._center is not None and self._token:
            self._draw_map(width, height)
        elif width >= 2 and height >= 2 and not self._token:
            self.canvas.create_text(width / 2, height / 2, text="Mapbox token is not set",
                                    fill="#8a8782", font=self._font, tags="chrome")
        if width >= 2 and height >= 2:
            self._draw_attribution(width, height)
            self._draw_chip()
            if self._user_panned:
                self._draw_recenter(width, height)

    def _draw_map(self, width: int, height: int) -> None:
        origin_x, origin_y = self._origin(width, height)
        n = 1 << self._zoom
        raw_x0 = int(math.floor(origin_x / _TILE))
        raw_y0 = int(math.floor(origin_y / _TILE))
        raw_x1 = int(math.floor((origin_x + width) / _TILE))
        raw_y1 = int(math.floor((origin_y + height) / _TILE))
        x0, y0 = max(0, raw_x0), max(0, raw_y0)
        x1, y1 = min(n - 1, raw_x1), min(n - 1, raw_y1)
        visible: set[tuple[int, int, int]] = set()
        if x0 <= x1 and y0 <= y1:
            for tx in range(x0, x1 + 1):
                for ty in range(y0, y1 + 1):
                    key = (self._zoom, tx, ty)
                    visible.add(key)
                    photo = self._tiles.get(key)
                    if photo is None:
                        self._request_tile(key)
                        continue
                    self._tiles.move_to_end(key)
                    sx = tx * _TILE - origin_x
                    sy = ty * _TILE - origin_y
                    self.canvas.create_image(sx, sy, image=photo, anchor="nw", tags="tile")
        self._visible = visible
        self._draw_route(origin_x, origin_y)
        self._draw_fix(origin_x, origin_y)
        self._trim(visible)

    def _draw_route(self, origin_x: float, origin_y: float) -> None:
        route = self._route
        if not route or len(route) < 2:
            return
        coords: list[float] = []
        last = None
        for lat, lng in route:
            sx, sy = world_px(lat, lng, self._zoom)
            point = (sx - origin_x, sy - origin_y)
            if point == last:
                continue
            coords.extend(point)
            last = point
        if len(coords) < 4:
            return
        self.canvas.create_line(*coords, fill="#ffffff", width=_ROUTE_WIDTH + 4, capstyle=tk.ROUND,
                                joinstyle=tk.ROUND, tags="route")
        self.canvas.create_line(*coords, fill=_ROUTE, width=_ROUTE_WIDTH, capstyle=tk.ROUND,
                                joinstyle=tk.ROUND, tags="route")
        ex, ey = coords[-2], coords[-1]
        self.canvas.create_oval(ex - 8, ey - 8, ex + 8, ey + 8, fill="#ffffff", outline="#ffffff", tags="route")
        self.canvas.create_oval(ex - 5, ey - 5, ex + 5, ey + 5, fill=_ROUTE, outline=_ROUTE, tags="route")

    def _draw_fix(self, origin_x: float, origin_y: float) -> None:
        if self._fix is None:
            return
        lat, lng = self._fix
        sx, sy = world_px(lat, lng, self._zoom)
        self.canvas.create_image(sx - origin_x, sy - origin_y, image=self._puck, anchor="center", tags="fix")

    def _draw_chip(self) -> None:
        if self._fix is None:
            text = "No GPS yet"
        else:
            age = _format_age(self._age_s)
            fresh = self._age_s is not None and self._age_s < 20
            text = "Live" if fresh else ("GPS" if age is None else f"GPS {age}")
            if self._accuracy is not None and self._accuracy > 0:
                text = f"{text}  ±{int(self._accuracy)} m"
        self._pill(text, 12, 12, "nw", "#ffffff", "#1a1918", self._font)

    def _draw_recenter(self, width: int, height: int) -> None:
        self._recenter_box = self._pill("Re-center", width - 12, height - 12, "se", "#1a1918", "#ffffff", self._font_small)

    def _draw_attribution(self, width: int, height: int) -> None:
        text = "© Mapbox  © OpenStreetMap"
        x, y = 10, height - 8
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.canvas.create_text(x + dx, y + dy, text=text, anchor="sw", fill=_BG, font=self._font_tiny, tags="chrome")
        self.canvas.create_text(x, y, text=text, anchor="sw", fill="#f0f0f0", font=self._font_tiny, tags="chrome")

    def _pill(self, text: str, x: int, y: int, anchor: str, fill: str, text_fill: str, font) -> tuple[int, int, int, int]:
        tw, th = self._measure(text, font)
        pad_x, pad_y = 9, 4
        bw, bh = tw + pad_x * 2, th + pad_y * 2
        if anchor == "se":
            left, top = x - bw, y - bh
        elif anchor == "ne":
            left, top = x - bw, y
        elif anchor == "sw":
            left, top = x, y - bh
        else:
            left, top = x, y
        self.canvas.create_rectangle(left, top, left + bw, top + bh, fill=fill, outline=fill, tags="chrome")
        self.canvas.create_text(left + pad_x, top + th / 2.0 + pad_y, text=text, anchor="w", font=font,
                                fill=text_fill, tags="chrome")
        return (left, top, left + bw, top + bh)

    def _measure(self, text: str, font) -> tuple[int, int]:
        cached = self._fonts.get(font)
        if cached is None:
            cached = tkfont.Font(self, font=font)
            self._fonts[font] = cached
        return cached.measure(text), cached.metrics("linespace")

    def _frame_destination(self) -> None:
        """Start the zoom onto a new destination once the pane has a size. A flight already running is left alone."""
        if not self._dest_view or not self._route or self._flight is not None:
            return
        try:
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        except tk.TclError:
            self._alive = False
            return
        view = route_view(self._route, width, height)
        if view is None:
            return
        self._dest_view = False
        center, zoom = view
        self._fly_to(center, zoom)

    def _fly_to(self, center: tuple[float, float], zoom: int) -> None:
        """Glide from the current view to ``center`` at ``zoom``. The walker stops pulling the camera until Re-center."""
        self._flight_i += 1
        token = self._flight_i
        self._flight = {
            "t0": time.monotonic(),
            "start": self._center or center,
            "start_z": self._zoom,
            "end": center,
            "end_z": zoom,
        }
        self._user_panned = True
        try:
            self.after(16, lambda t=token: self._tick_flight(t))
        except tk.TclError:
            self._alive = False

    def _tick_flight(self, token: int) -> None:
        flight = self._flight
        if not self._alive or token != self._flight_i or flight is None:
            return
        u = min(1.0, (time.monotonic() - flight["t0"]) / _FLIGHT_S)
        eased = u * u * (3.0 - 2.0 * u)
        start, end = flight["start"], flight["end"]
        self._center = (start[0] + (end[0] - start[0]) * eased, start[1] + (end[1] - start[1]) * eased)
        zoom = int(round(flight["start_z"] + (flight["end_z"] - flight["start_z"]) * eased))
        self._zoom = min(_MAX_ZOOM, max(_MIN_ZOOM, zoom))
        if u >= 1.0:
            self._center = end
            self._zoom = flight["end_z"]
            self._flight = None
        self._redraw()
        if self._flight is None or token != self._flight_i:
            return
        try:
            self.after(32, lambda t=token: self._tick_flight(t))
        except tk.TclError:
            self._alive = False

    def _cancel_flight(self) -> None:
        self._flight = None
        self._flight_i += 1

    def _trim(self, visible: set[tuple[int, int, int]]) -> None:
        if len(self._tiles) <= _CACHE_MAX:
            return
        kept: OrderedDict[tuple[int, int, int], ImageTk.PhotoImage] = OrderedDict()
        for key, photo in self._tiles.items():
            if key in visible:
                kept[key] = photo
        room = _CACHE_MAX - len(kept)
        if room > 0:
            extras = [item for item in self._tiles.items() if item[0] not in kept]
            for key, photo in extras[-room:]:
                kept[key] = photo
        self._tiles = kept


def _opt_float(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _wheel_steps(delta: int) -> int:
    """Windows sends ±120 per notch. Trackpads send small deltas; each of those is one zoom step."""
    if not delta:
        return 0
    if abs(delta) >= 120:
        return int(delta / 120) or (1 if delta > 0 else -1)
    return 1 if delta > 0 else -1
