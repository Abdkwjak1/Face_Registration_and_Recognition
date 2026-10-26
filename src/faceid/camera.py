"""Frame sources: a real webcam, or a scripted sequence of images for tests and demos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

from faceid.config import Settings


class CameraError(RuntimeError):
    """No usable camera could be opened."""


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None:
        """Next BGR frame, or ``None`` if no frame is available."""

    def release(self) -> None: ...


class Camera:
    """A ``cv2.VideoCapture`` wrapper that yields ``None`` instead of raising on bad reads."""

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480):
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"Cannot open camera {source!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def open_camera(settings: Settings | None = None) -> Camera:
    """Open the configured camera, or probe indices until one delivers a frame.

    The original code hard-coded index 1; laptops usually have the webcam at 0 and
    Linux exposes metadata nodes that open but never produce frames.
    """
    settings = settings or Settings()
    candidates = (
        [settings.camera_index]
        if settings.camera_index is not None
        else range(settings.camera_probe_range)
    )
    for index in candidates:
        try:
            cam = Camera(index, settings.frame_width, settings.frame_height)
        except CameraError:
            continue
        if any(cam.read() is not None for _ in range(5)):
            return cam
        cam.release()
    raise CameraError(
        "No working camera found. Plug one in, close other apps using it, "
        "or pick an index with --camera."
    )


class ImageSequenceSource:
    """Replays prepared frames, optionally looping. Stands in for a camera in tests/demos."""

    def __init__(self, frames: Sequence[np.ndarray], loop: bool = True):
        if not frames:
            raise ValueError("ImageSequenceSource needs at least one frame")
        self._frames = list(frames)
        self._loop = loop
        self._i = 0

    def read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            if not self._loop:
                return None
            self._i = 0
        frame = self._frames[self._i]
        self._i += 1
        return frame.copy()

    def release(self) -> None:
        pass
