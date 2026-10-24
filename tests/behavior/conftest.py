"""Fixtures for end-to-end drone tests against kite.testing's protocol-level Tello emulator.

    def test_something(tello_emulator):          # a connected kite TelloDrone on a fake Tello over real UDP
        rig = tello_emulator
        rig.takeoff()
        rig.fly(1.0, fb=40)                       # SafetyGovernor.check + rc at 10 Hz, like the recorder
        assert rig.emu.snapshot().north > 0.3     # ground truth

    @pytest.mark.emulator(time_scale=2, battery_pct=40, physics={"yaw_sign": -1}, drone={"yaw_sign": -1},
                          safety={"max_flight_s": 1.0}, connect=False)

tello_emulator_factory gives the same rig as a context manager, for tests that need several or custom ones.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import pytest

from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.tello import TelloDrone
from kite.testing import TelloEmulator, TelloPhysics, djitellopy_loopback

DEFAULT_TIME_SCALE = 4.0


def pytest_configure(config):
    config.addinivalue_line("markers", "emulator(**options): options for the tello_emulator fixture (TelloEmulator "
                            "kwargs, plus physics={...}, drone={...} TelloDrone kwargs, safety={...}, connect=bool)")


@dataclass
class TelloRig:
    emu: TelloEmulator
    link: object
    drone: TelloDrone
    gov: SafetyGovernor
    log: list = field(default_factory=list)

    def wait(self, timeout_s: float = 15.0) -> str | None:
        """Poll busy() (what completes a discrete command) until it is done; the result."""
        t_end = time.monotonic() + timeout_s
        while self.gov.busy():
            if time.monotonic() > t_end:
                raise AssertionError(f"still busy after {timeout_s} s: {self.emu.snapshot()}")
            time.sleep(0.005)
        return self.gov.last_result()

    def takeoff(self, fast: bool = True) -> None:
        """Take off and wait for the 'ok'. fast: run the emulator at >= 8x meanwhile (the takeoff itself is not
        what most tests look at; kite's timeouts are real time, so nothing else changes)."""
        scale = self.emu.time_scale
        if fast:
            self.emu.time_scale = max(scale, 8.0)
        try:
            self.gov.takeoff()
            res = self.wait()
        finally:
            self.emu.time_scale = scale
        assert res == "ok" and self.gov.flying, f"takeoff -> {res}"

    def land(self) -> str | None:
        self.gov.land()
        return self.wait()

    def fly(self, seconds: float, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0, hz: float = 10.0,
            governor: bool = True, frame_age_s: float = 0.0, on_tick=None) -> list[str]:
        """The recorder's control loop for `seconds`: safety check, then the sticks, at hz. Returns the actions."""
        actions = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            now = time.time()
            if governor:
                actions.append(self.gov.check(now, now - frame_age_s))
                self.gov.busy()
                self.gov.rc(lr, fb, ud, yaw)
            else:
                self.drone.busy()
                self.drone.rc(lr, fb, ud, yaw)
            if on_tick is not None:
                on_tick()
            time.sleep(1.0 / hz)
        return actions

    def fresh_telemetry(self, timeout_s: float = 2.0):
        """Telemetry from the next state packet that arrives (kite stamps a packet when it first sees it, so a packet
        that arrived earlier but was not looked at yet would look fresh)."""
        old = self.drone.tello.get_current_state()
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            if self.drone.tello.get_current_state() is not old:
                return self.drone.telemetry()
            time.sleep(0.002)
        raise AssertionError("no new state packet")


@contextlib.contextmanager
def open_rig(time_scale: float = DEFAULT_TIME_SCALE, physics: dict | None = None, drone: dict | None = None,
             safety: dict | None = None, connect: bool = True, video_port: int | None = None, **emu_kwargs):
    phys = TelloPhysics(**(physics or {}))
    with TelloEmulator(time_scale=time_scale, physics=phys, **emu_kwargs) as emu, \
            djitellopy_loopback(emu, video_port=video_port) as link:
        kw = {"video": False}
        kw.update(drone or {})
        d = TelloDrone(host=emu.host, **kw)
        gov = SafetyGovernor(d, SafetyCfg(**(safety or {})))
        rig = TelloRig(emu, link, d, gov)
        try:
            if connect:
                d.connect()
                rig.fresh_telemetry()
            yield rig
        finally:
            emu.clear_faults()
            link.wifi_up()
            emu.time_scale = max(emu.time_scale, 10.0)  # land quickly on close()
            if d.tello is not None:
                d.close()


def _options(request) -> dict:
    opts = {}
    for m in reversed(list(request.node.iter_markers("emulator"))):
        opts.update(m.kwargs)
    if hasattr(request, "param") and isinstance(request.param, dict):
        opts.update(request.param)
    return opts


@pytest.fixture
def tello_emulator(request):
    """A TelloRig: emulator + djitellopy loopback + connected kite TelloDrone (video off) + SafetyGovernor."""
    with open_rig(**_options(request)) as rig:
        yield rig


@pytest.fixture
def tello_emulator_factory():
    """open_rig(**options) as a context manager."""
    return open_rig
