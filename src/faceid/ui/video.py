"""Helpers to draw on camera frames and hand them to Tk without a Pillow/ImageTk dependency."""

from __future__ import annotations

import cv2
import numpy as np

from faceid.detection import Face

WHITE = (255, 255, 255)  # BGR
ORANGE = (0, 74, 184)  # #b84a00, the app accent


def annotate(frame: np.ndarray, face: Face | None, color: tuple[int, int, int],
             label: str = "") -> np.ndarray:
    """Draw a face box (and optional caption) on a copy of ``frame``."""
    out = frame.copy()
    if face is not None:
        x, y, w, h = face.box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(out, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)
    return out


def placeholder(width: int, height: int, text: str) -> np.ndarray:
    """Dark frame with centred text, shown when there is no camera image."""
    img = np.full((height, width, 3), (17, 17, 17), np.uint8)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    org = ((width - size[0]) // 2, (height + size[1]) // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def to_ppm(frame: np.ndarray, width: int, height: int) -> bytes:
    """Scale ``frame`` to fit ``width`` x ``height`` and encode as binary PPM.

    PPM needs no compression step, so it is much cheaper per frame than PNG, and
    ``tk.PhotoImage(data=...)`` reads it natively.
    """
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    if scale != 1.0:
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()
