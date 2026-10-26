"""Face detection (Haar cascade) and crop normalisation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceid.config import Settings

_CASCADE = "haarcascade_frontalface_default.xml"


@dataclass(frozen=True)
class Face:
    """A detected face as a bounding box in the coordinates of the original frame."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of ``image`` (BGR, BGRA or already gray)."""
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


class FaceDetector:
    """Haar-cascade detector tuned for a webcam stream.

    Frames are downscaled before detection (the cost of a Haar scan grows with pixel
    count), and boxes are mapped back to full-frame coordinates.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + _CASCADE)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load OpenCV cascade {_CASCADE!r}")

    def detect(self, frame: np.ndarray) -> list[Face]:
        """All faces in ``frame``, largest first."""
        gray = to_gray(frame)
        scale = min(1.0, self.settings.detect_max_side / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else gray
        min_side = max(20, int(self.settings.min_face_size * scale))
        boxes = self._cascade.detectMultiScale(
            small,
            scaleFactor=self.settings.scale_factor,
            minNeighbors=self.settings.min_neighbors,
            minSize=(min_side, min_side),
        )
        faces = [
            Face(*(int(round(v / scale)) for v in box)) for box in boxes
        ]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def largest(self, frame: np.ndarray) -> Face | None:
        """The biggest face in ``frame`` (the person closest to the camera), if any."""
        faces = self.detect(frame)
        return faces[0] if faces else None

    def crop(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """Gray, square-resized crop of ``face``, ready to store or recognise."""
        gray = to_gray(frame)
        h, w = gray.shape
        x0, y0 = max(face.x, 0), max(face.y, 0)
        x1, y1 = min(face.x + face.w, w), min(face.y + face.h, h)
        return resize_face(gray[y0:y1, x0:x1], self.settings.face_size)


def resize_face(gray: np.ndarray, size: int) -> np.ndarray:
    """Resize a gray crop to ``size`` x ``size`` (area interpolation when shrinking)."""
    if gray.shape == (size, size):
        return gray
    interp = cv2.INTER_AREA if gray.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(gray, (size, size), interpolation=interp)
