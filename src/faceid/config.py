"""Runtime configuration. Everything tunable lives here, nothing is hard-coded elsewhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_home() -> Path:
    return Path(os.environ.get("FACEID_HOME", "data")).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    """Tunable parameters and storage locations.

    ``home`` defaults to ``$FACEID_HOME`` or ``./data``; the guest CSV, the face
    samples and the trained model all live below it so one folder is the whole state.
    """

    home: Path = field(default_factory=_default_home)

    # Camera
    camera_index: int | None = None  # None = probe indices 0..camera_probe_range
    camera_probe_range: int = 4
    frame_width: int = 640
    frame_height: int = 480

    # Detection (Haar cascade)
    scale_factor: float = 1.2
    min_neighbors: int = 5
    min_face_size: int = 80
    detect_max_side: int = 480  # frames are downscaled to this before detection

    # Normalised face crop fed to the recognizer
    face_size: int = 160

    # Registration
    samples_per_guest: int = 20
    sample_interval_s: float = 0.15  # min gap between saved samples (pose variety)

    # Recognition. LBPH returns a distance: lower is a better match.
    confidence_threshold: float = 65.0  # distance above this => "unknown" (see docs/BENCHMARKS.md)
    recognition_frames: int = 12  # frames voted over per check
    min_votes: float = 0.5  # share of frames that must agree on one guest
    recognition_timeout_s: float = 8.0

    @property
    def guests_csv(self) -> Path:
        return self.home / "guests.csv"

    @property
    def dataset_dir(self) -> Path:
        return self.home / "dataset"

    @property
    def model_path(self) -> Path:
        return self.home / "model.yml"

    def ensure_dirs(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
