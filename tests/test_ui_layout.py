"""The console must not grow with the font and push the camera and the map into a strip."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run"))

from ui import console_lines  # noqa: E402


def test_a_tall_font_keeps_the_log_short_enough_for_the_panes():
    """200% scaling on a short window: an 11-line log would be most of it. Four lines leaves the panes the rest."""
    assert console_lines(900, 43) == 4


def test_a_tall_screen_still_grows_the_log_a_little():
    """The same large font in a window that uses a 1600px-tall screen keeps a short log and tall panes."""
    assert console_lines(1408, 43) == 5


def test_a_normal_font_still_gets_the_full_log():
    """100% scaling: 11 lines fit in a 900px window."""
    assert console_lines(900, 16) == 11


def test_a_short_window_never_drops_below_four_lines():
    assert console_lines(480, 40) == 4
