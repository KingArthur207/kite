"""The live segmentation window.

With no display, ``show`` gives up quietly so a headless run can still write
frames and print events.
"""

from __future__ import annotations

import numpy as np


class Display:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.headless = not enabled
        self._quit = False
        self._windows: set[str] = set()

    def show(self, image: np.ndarray | None, title: str = "Kite street") -> None:
        if self.headless or image is None:
            return
        import cv2

        try:
            if title not in self._windows:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                height, width = image.shape[:2]
                scale = min(1280 / float(width), 960 / float(height), 1.0)
                cv2.resizeWindow(title, max(1, int(width * scale)), max(1, int(height * scale)))
                cv2.moveWindow(title, 48, 48)
                self._windows.add(title)
            cv2.imshow(title, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._quit = True
        except cv2.error:
            self.headless = True

    def wait(self, ms: int) -> int:
        if self.headless:
            return -1
        import cv2

        try:
            key = cv2.waitKey(max(1, ms)) & 0xFF
        except cv2.error:
            self.headless = True
            return -1
        if key in (ord("q"), 27):
            self._quit = True
        return key

    def poll_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        if not self._windows:
            return
        import cv2

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self._windows.clear()
