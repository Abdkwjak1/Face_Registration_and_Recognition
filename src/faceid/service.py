"""Application workflows that tie camera frames, detection, storage and recognition together.

The sessions are *incremental* (``feed`` one frame at a time) so a GUI can drive them from
its event loop without blocking, while the CLI and tests simply loop over a frame source.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from faceid.camera import FrameSource
from faceid.config import Settings
from faceid.dataset import FaceDataset
from faceid.detection import Face, FaceDetector
from faceid.guests import Guest, GuestRepository
from faceid.recognition import FaceRecognizer, Identification, NotTrainedError

Clock = Callable[[], float]


class EnrollmentSession:
    """Collects ``target`` face samples for one guest, spaced ``interval`` seconds apart."""

    def __init__(self, service: FaceIDService, guest_id: int, target: int, interval: float,
                 clock: Clock = time.monotonic):
        self._service = service
        self.guest_id = guest_id
        self.target = target
        self.saved = 0
        self._interval = interval
        self._clock = clock
        self._last = float("-inf")

    @property
    def done(self) -> bool:
        return self.saved >= self.target

    def feed(self, frame: np.ndarray) -> Face | None:
        """Process one frame; returns the detected face (for drawing) or ``None``."""
        if self.done:
            return None
        face = self._service.detector.largest(frame)
        now = self._clock()
        if face is not None and now - self._last >= self._interval:
            crop = self._service.detector.crop(frame, face)
            self._service.dataset.save(self.guest_id, crop)
            self.saved += 1
            self._last = now
        return face


class RecognitionSession:
    """Collects up to ``frames`` face predictions (or until timeout) and votes."""

    def __init__(self, service: FaceIDService, clock: Clock = time.monotonic):
        if not service.recognizer.is_trained:
            raise NotTrainedError("No trained model yet: register a guest and train first")
        self._service = service
        s = service.settings
        self.target = s.recognition_frames
        self._deadline = clock() + s.recognition_timeout_s
        self._clock = clock
        self._predictions = []

    @property
    def collected(self) -> int:
        return len(self._predictions)

    @property
    def timed_out(self) -> bool:
        return self._clock() > self._deadline

    @property
    def done(self) -> bool:
        return self.collected >= self.target or self.timed_out

    def feed(self, frame: np.ndarray) -> Face | None:
        if self.done:
            return None
        face = self._service.detector.largest(frame)
        if face is not None:
            crop = self._service.detector.crop(frame, face)
            self._predictions.append(self._service.recognizer.predict(crop))
        return face

    def result(self) -> Identification:
        """Verdict so far. With zero faces seen this is an unknown with ``frames == 0``."""
        return self._service.recognizer.decide(self._predictions)


@dataclass(frozen=True)
class Status:
    guests: int
    samples: int
    model_ready: bool


class FaceIDService:
    def __init__(
        self,
        settings: Settings | None = None,
        detector: FaceDetector | None = None,
        recognizer: FaceRecognizer | None = None,
    ):
        self.settings = settings or Settings()
        self.detector = detector or FaceDetector(self.settings)
        self.recognizer = recognizer or FaceRecognizer(self.settings)
        self.dataset = FaceDataset(self.settings.dataset_dir, self.settings.face_size)
        self.guests = GuestRepository(self.settings.guests_csv)
        if self.settings.model_path.is_file():
            self.recognizer.load(self.settings.model_path)

    # -- registration -------------------------------------------------------
    def register(self, guest: Guest) -> Guest:
        return self.guests.add(guest)

    def start_enrollment(self, guest_id: int, clock: Clock = time.monotonic) -> EnrollmentSession:
        s = self.settings
        return EnrollmentSession(self, guest_id, s.samples_per_guest, s.sample_interval_s, clock)

    def enroll(self, guest_id: int, source: FrameSource, max_frames: int = 600) -> int:
        """Blocking helper: pull frames from ``source`` until enough samples are saved."""
        session = self.start_enrollment(guest_id)
        for _ in range(max_frames):
            if session.done:
                break
            frame = source.read()
            if frame is not None:
                session.feed(frame)
        return session.saved

    def train(self) -> int:
        """(Re)train on every stored sample and persist the model. Returns the sample count.

        With no samples left the model is discarded, so nobody is recognised.
        """
        faces, ids = self.dataset.load()
        if not faces:
            self.recognizer.reset()
            self.settings.model_path.unlink(missing_ok=True)
            return 0
        self.recognizer.train(faces, ids)
        self.recognizer.save(self.settings.model_path)
        return len(faces)

    def remove_guest(self, guest_id: int) -> bool:
        """Delete a guest's record and samples. The model must be retrained afterwards."""
        self.dataset.delete_guest(guest_id)
        return self.guests.delete(guest_id)

    # -- recognition --------------------------------------------------------
    def start_recognition(self, clock: Clock = time.monotonic) -> RecognitionSession:
        return RecognitionSession(self, clock)

    def identify(self, source: FrameSource, max_frames: int = 600) -> Identification:
        """Blocking helper: vote over live frames from ``source``."""
        session = self.start_recognition()
        for _ in range(max_frames):
            if session.done:
                break
            frame = source.read()
            if frame is not None:
                session.feed(frame)
        return session.result()

    def lookup(self, result: Identification) -> Guest | None:
        return self.guests.get(result.guest_id) if result.guest_id is not None else None

    def status(self) -> Status:
        counts = self.dataset.counts()
        return Status(len(self.guests.all()), sum(counts.values()), self.recognizer.is_trained)
