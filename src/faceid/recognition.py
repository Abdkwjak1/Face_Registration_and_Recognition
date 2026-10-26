"""LBPH face recognition with an "unknown" decision and multi-frame voting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from faceid.config import Settings
from faceid.detection import resize_face


@dataclass(frozen=True)
class Prediction:
    """Best match for one face. ``distance`` is LBPH's score: lower is better."""

    label: int
    distance: float


@dataclass(frozen=True)
class Identification:
    """Verdict over several frames. ``guest_id`` is ``None`` when nobody matched."""

    guest_id: int | None
    distance: float  # mean distance of the frames that voted for the winner (inf if none)
    votes: int
    frames: int

    @property
    def known(self) -> bool:
        return self.guest_id is not None


class NotTrainedError(RuntimeError):
    pass


class FaceRecognizer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._lbph = cv2.face.LBPHFaceRecognizer_create()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._trained = False

    def reset(self) -> None:
        """Forget everything learned (used when the last guest is removed)."""
        self._lbph = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """Fixed size + local contrast equalisation, so lighting changes matter less."""
        return self._clahe.apply(resize_face(face, self.settings.face_size))

    def train(self, faces: list[np.ndarray], ids: np.ndarray) -> None:
        if not faces:
            raise ValueError("No face samples to train on")
        self._lbph.train([self.preprocess(f) for f in faces], np.asarray(ids, dtype=np.int32))
        self._trained = True

    def predict(self, face: np.ndarray) -> Prediction:
        if not self._trained:
            raise NotTrainedError("Train or load a model first")
        label, distance = self._lbph.predict(self.preprocess(face))
        return Prediction(int(label), float(distance))

    def identify(self, faces: Iterable[np.ndarray]) -> Identification:
        """Vote over ``faces``: a frame votes for its best match only if it is close enough.

        The winner must collect at least ``min_votes`` of all frames; otherwise the person
        is reported as unknown. Voting makes one bad frame (blur, blink) harmless.
        """
        predictions = [self.predict(f) for f in faces]
        return self.decide(predictions)

    def decide(self, predictions: list[Prediction]) -> Identification:
        total = len(predictions)
        accepted = [p for p in predictions if p.distance <= self.settings.confidence_threshold]
        if not accepted:
            return Identification(None, float("inf"), 0, total)
        label, votes = Counter(p.label for p in accepted).most_common(1)[0]
        mean = float(np.mean([p.distance for p in accepted if p.label == label]))
        if votes < self.settings.min_votes * total:
            return Identification(None, mean, votes, total)
        return Identification(label, mean, votes, total)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lbph.write(str(path))

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise NotTrainedError(f"No trained model at {path}")
        self._lbph.read(str(path))
        self._trained = True
