"""On-disk store of face samples, one file per sample: ``user.<guest_id>.<n>.png``."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

from faceid.detection import resize_face

log = logging.getLogger(__name__)
_NAME = re.compile(r"^user\.(\d+)\.(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


class FaceDataset:
    def __init__(self, directory: Path, face_size: int = 160):
        self.directory = Path(directory)
        self.face_size = face_size

    def _samples(self) -> list[tuple[int, int, Path]]:
        if not self.directory.is_dir():
            return []
        found = []
        for path in self.directory.iterdir():
            m = _NAME.match(path.name)
            if m:
                found.append((int(m[1]), int(m[2]), path))
        return sorted(found)

    def save(self, guest_id: int, face: np.ndarray) -> Path:
        """Store a gray face crop and return its path."""
        self.directory.mkdir(parents=True, exist_ok=True)
        n = max((n for gid, n, _ in self._samples() if gid == guest_id), default=0) + 1
        path = self.directory / f"user.{guest_id}.{n}.png"
        if not cv2.imwrite(str(path), face):
            raise OSError(f"Could not write {path}")
        return path

    def count(self, guest_id: int) -> int:
        return sum(1 for gid, _, _ in self._samples() if gid == guest_id)

    def counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for gid, _, _ in self._samples():
            out[gid] = out.get(gid, 0) + 1
        return out

    def load(self) -> tuple[list[np.ndarray], np.ndarray]:
        """All samples as ``(faces, ids)``; unreadable files are skipped with a warning."""
        faces, ids = [], []
        for gid, _, path in self._samples():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                log.warning("Skipping unreadable sample %s", path)
                continue
            faces.append(resize_face(img, self.face_size))
            ids.append(gid)
        return faces, np.array(ids, dtype=np.int32)

    def delete_guest(self, guest_id: int) -> int:
        removed = 0
        for gid, _, path in self._samples():
            if gid == guest_id:
                path.unlink()
                removed += 1
        return removed
