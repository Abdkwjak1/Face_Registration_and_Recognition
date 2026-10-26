"""Accuracy and speed measurements for the recognition pipeline.

Accuracy is measured two ways on a labelled folder of face crops:

* **Genuine** (stratified k-fold): train on k-1 folds, test on the held-out samples of
  people the model knows. Gives the true-accept rate (TAR) and identification accuracy.
* **Impostor** (leave-one-identity-out): train without person P, test on P. Every such
  sample *should* be rejected as unknown; the share that is accepted is the false-accept
  rate (FAR).

Sweeping the distance threshold over both gives the trade-off used to pick
``Settings.confidence_threshold``.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from faceid.config import Settings
from faceid.dataset import FaceDataset
from faceid.detection import FaceDetector
from faceid.recognition import FaceRecognizer


@dataclass(frozen=True)
class Distances:
    """Raw LBPH outputs for held-out samples."""

    genuine: list[tuple[int, int, float]]  # (true_id, predicted_id, distance)
    impostor: list[float]  # distance of the best (necessarily wrong) match


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    tar: float  # genuine samples correctly identified AND accepted
    far: float  # impostor samples accepted


def _folds(ids: np.ndarray, k: int) -> list[np.ndarray]:
    """Stratified round-robin folds (deterministic)."""
    folds: list[list[int]] = [[] for _ in range(k)]
    for person in np.unique(ids):
        for j, idx in enumerate(np.flatnonzero(ids == person)):
            folds[j % k].append(int(idx))
    return [np.array(f, dtype=int) for f in folds]


def collect_distances(
    faces: list[np.ndarray], ids: np.ndarray, k: int = 5, settings: Settings | None = None
) -> Distances:
    settings = settings or Settings()
    ids = np.asarray(ids)
    genuine: list[tuple[int, int, float]] = []
    for test_idx in _folds(ids, k):
        mask = np.ones(len(ids), dtype=bool)
        mask[test_idx] = False
        rec = FaceRecognizer(settings)
        rec.train([f for f, m in zip(faces, mask, strict=True) if m], ids[mask])
        for i in test_idx:
            p = rec.predict(faces[i])
            genuine.append((int(ids[i]), p.label, p.distance))

    impostor: list[float] = []
    people = np.unique(ids)
    if len(people) >= 2:
        for person in people:
            mask = ids != person
            rec = FaceRecognizer(settings)
            rec.train([f for f, m in zip(faces, mask, strict=True) if m], ids[mask])
            impostor += [rec.predict(faces[i]).distance for i in np.flatnonzero(~mask)]
    return Distances(genuine, impostor)


def sweep(d: Distances, thresholds: list[float]) -> list[ThresholdPoint]:
    out = []
    for t in thresholds:
        tar = np.mean([true == pred and dist <= t for true, pred, dist in d.genuine])
        far = np.mean([dist <= t for dist in d.impostor]) if d.impostor else float("nan")
        out.append(ThresholdPoint(t, float(tar), float(far)))
    return out


def identification_accuracy(d: Distances) -> float:
    """Closed-set accuracy ignoring the threshold: was the nearest person the right one?"""
    return float(np.mean([true == pred for true, pred, _ in d.genuine]))


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    p95_ms: float

    @property
    def per_second(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms else float("inf")


def time_it(fn: Callable[[], object], repeats: int = 50, warmup: int = 3) -> Timing:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    return Timing(statistics.fmean(samples), p95)


def benchmark_pipeline(dataset_dir: Path, settings: Settings | None = None) -> dict[str, Timing]:
    """Time each stage on a labelled folder of face crops.

    Detection is timed on a synthetic 640x480 frame in which a sample face is pasted on a
    plain background, so the numbers are reproducible without a camera.
    """
    settings = settings or Settings()
    faces, ids = FaceDataset(dataset_dir, settings.face_size).load()
    if not faces:
        raise ValueError(f"No face samples in {dataset_dir}")
    detector = FaceDetector(settings)
    frame = synthetic_frame(faces[0])
    for candidate in faces:
        frame = synthetic_frame(candidate)
        if detector.largest(frame) is not None:
            break
    face = detector.largest(frame)
    rec = FaceRecognizer(settings)
    out = {
        f"train ({len(faces)} samples)": time_it(
            lambda: rec.train(faces, ids), repeats=5, warmup=1),
        "detect (640x480 frame)": time_it(lambda: detector.detect(frame)),
    }
    crop = detector.crop(frame, face) if face else faces[0]
    out["crop + preprocess"] = time_it(lambda: rec.preprocess(crop))
    out["predict (1 face)"] = time_it(lambda: rec.predict(crop))
    out["detect + crop + predict"] = time_it(
        lambda: [rec.predict(detector.crop(frame, f)) for f in detector.detect(frame)[:1]]
    )
    return out


def synthetic_frame(face: np.ndarray, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """A BGR frame with ``face`` centred on a flat gray background."""
    w, h = size
    canvas = np.full((h, w), 127, np.uint8)
    side = min(h * 3 // 5, 280)
    patch = cv2.resize(face, (side, side))
    y0, x0 = (h - side) // 2, (w - side) // 2
    canvas[y0:y0 + side, x0:x0 + side] = patch
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
