"""All tunables in one place. Numbers that matter for safety carry their reasoning.

Bearings everywhere are degrees in the WEARER's frame: 0 = straight ahead,
+90 = right, -90 = left, +-180 = behind. Distances are metres.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """Load the repo-level .env once (keys are never stored in code)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


_load_env()


@dataclass
class AudioCfg:
    engine: str = field(default_factory=lambda: os.environ.get("KITE_AUDIO_ENGINE", "auto"))
    # "rust" (kite_audio, real-time thread outside the GIL),
    # "python" (numpy mixer), or "auto" = rust when the extension is built
    sample_rate: int = 48000          # AirPods A2DP native rate; Cartesia renders at it directly
    block: int = 1024                 # python engine: ~21 ms per callback, headroom against GIL stalls
    # (measured a 12.5 ms worst-case render at 512); Bluetooth itself adds 150-250 ms
    rust_block: int = 512             # rust engine: no GIL in the audio path, so a smaller buffer is safe
    output_substr: str = "AirPods"    # preferred output device (name substring)
    master_gain: float = 0.9
    limiter_ceiling: float = 0.89     # -1 dBFS: protects hearing and avoids codec clipping
    speech_gain: float = 1.0
    cue_gain: float = 0.55            # beeps sit under speech so words stay intelligible
    duck_cues_during_speech: float = 0.45  # non-critical cues drop ~7 dB while speech plays
    device_poll_s: float = 2.0        # how often to check for AirPods (dis)connects


@dataclass
class SonarCfg:
    """Proximity warnings. Walking speed ~1.3 m/s, so 4 s of look-ahead is ~5 m."""
    walk_speed_mps: float = 1.2       # assumed closing speed for static obstacles
    corridor_half_width_m: float = 0.55   # half shoulder width + margin
    corridor_margin_m: float = 0.35       # soft edge: partial urgency just outside the corridor
    max_range_m: float = 6.0
    ttc_start_s: float = 3.5          # start beeping below this time-to-contact (BBeep: 2.5-5 s)
    ttc_critical_s: float = 1.0       # below this (or inside critical_dist) = critical
    critical_dist_m: float = 0.75
    hysteresis: float = 0.15          # exit thresholds are 15 % looser than entry thresholds
    interval_slow_s: float = 1.0      # inter-beep gap at the edge of the zone (WatchOut: 1.2 s)
    interval_fast_s: float = 0.11     # at critical (WatchOut: 0.21 s; parking sensors go continuous)
    pitch_low_hz: float = 520.0
    pitch_high_hz: float = 1250.0
    max_sources: int = 2              # >2 simultaneous sonified sources stop being intelligible
    second_source_gain: float = 0.55
    stale_track_s: float = 0.6        # an object not refreshed for this long fades out
    speak_label_cooldown_s: float = 6.0   # per track: "person, left" at most this often
    speak_global_gap_s: float = 2.5       # between any two hazard labels


@dataclass
class BeaconCfg:
    """Soundscape-style audio beacon toward the next waypoint."""
    period_far_s: float = 1.1
    period_near_s: float = 0.45
    near_dist_m: float = 5.0
    far_dist_m: float = 40.0
    on_course_deg: float = 15.0       # inside this cone: bright "on course" tick
    near_axis_deg: float = 55.0       # Soundscape's 4-region beacon: 15 / 55 / 125 deg
    behind_deg: float = 125.0         # beyond this: "turn around" cue instead of a tick
    arrive_dist_m: float = 3.0


@dataclass
class SpeechCfg:
    cartesia_model: str = "sonic-3.6"
    cartesia_version: str = "2026-08-14"
    voice_id: str = "47c38ca4-5f35-497b-b1a3-415245fb35e1"   # "Daniel": clear, crisp assistant voice
    speed: float = 1.1                # blind users typically prefer faster-than-default speech
    connect_timeout_s: float = 4.0
    first_chunk_timeout_s: float = 2.5    # past this, fall back to the offline voice
    offline_voice: str = "Samantha"   # macOS `say` voice used without internet
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "earpiece" / ".cache" / "tts")
    default_ttl_s: float = 6.0        # navigation speech older than this is dropped, never spoken late
    dedupe_window_s: float = 4.0


@dataclass
class ListenCfg:
    input_substr: str = ""            # "" = system default mic. Using the MacBook mic keeps the
    # AirPods in high-quality stereo; their own mic forces mono "call" mode while open.
    sample_rate: int = 16000
    stt_model: str = "ink-whisper"
    wake_words: tuple[str, ...] = ("kite", "hey kite", "okay kite", "ok kite")
    hands_free: bool = False          # True = always listening for the wake word
    max_utterance_s: float = 8.0
    end_silence_s: float = 0.7
    no_speech_timeout_s: float = 4.0


@dataclass
class NetCfg:
    udp_port: int = 5010              # fire-and-forget JSON events from perception / nav
    ws_port: int = 5011               # bidirectional JSON (events in, state + voice requests out)
    bind_host: str = "127.0.0.1"
    # SpacetimeDB mirror + caregiver inbox: OFF unless `run --spacetime` or KITE_SPACETIME=1,
    # because every call spends Maincloud credits.
    spacetime_host: str = field(default_factory=lambda: os.environ.get("SPACETIME_HOST")
                                or "https://maincloud.spacetimedb.com")
    spacetime_db: str = field(default_factory=lambda: os.environ.get("SPACETIME_DB") or "kite-sync-pl6rp")
    spacetime_enabled: bool = field(default_factory=lambda: os.environ.get("KITE_SPACETIME", "").strip()
                                    .lower() in ("1", "true", "yes", "on"))


@dataclass
class WatchdogCfg:
    perception_stale_s: float = 1.5   # no obstacle frames for this long = detection is down
    announce_recovery: bool = True


@dataclass
class Cfg:
    audio: AudioCfg = field(default_factory=AudioCfg)
    sonar: SonarCfg = field(default_factory=SonarCfg)
    beacon: BeaconCfg = field(default_factory=BeaconCfg)
    speech: SpeechCfg = field(default_factory=SpeechCfg)
    listen: ListenCfg = field(default_factory=ListenCfg)
    net: NetCfg = field(default_factory=NetCfg)
    watchdog: WatchdogCfg = field(default_factory=WatchdogCfg)

    @property
    def cartesia_key(self) -> str | None:
        return os.environ.get("CARTESIA_API_KEY") or None

    @property
    def anthropic_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY") or None
