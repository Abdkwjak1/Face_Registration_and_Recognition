"""Shared fixtures.

``tests/fixtures/faces`` holds real face crops of two people (``user.<id>.<n>.jpg``).
Frames for end-to-end tests are built by pasting those crops on a plain 640x480 canvas;
only crops the Haar detector actually finds are kept, so tests don't depend on luck.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import pytest

from faceid.camera import ImageSequenceSource
from faceid.config import Settings
from faceid.dataset import FaceDataset
from faceid.detection import FaceDetector
from faceid.evaluation import synthetic_frame
from faceid.service import FaceIDService

FIXTURES = Path(__file__).parent / "fixtures" / "faces"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def labelled_faces():
    """``(faces, ids)`` for every fixture crop."""
    return FaceDataset(FIXTURES).load()


@pytest.fixture(scope="session")
def detector() -> FaceDetector:
    return FaceDetector(Settings())


@pytest.fixture(scope="session")
def person_frames(detector):
    """``{person_id: [BGR frames with a detectable face]}``."""
    frames: dict[int, list] = {}
    for path in sorted(FIXTURES.glob("user.*.jpg")):
        person = int(path.name.split(".")[1])
        frame = synthetic_frame(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        if detector.largest(frame) is not None:
            frames.setdefault(person, []).append(frame)
    assert all(len(v) >= 8 for v in frames.values()), "fixtures no longer detectable"
    return frames


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated storage and no inter-sample delay, so enrolment is instant."""
    return dataclasses.replace(Settings(), home=tmp_path / "home", sample_interval_s=0.0)


@pytest.fixture
def service(settings, detector) -> FaceIDService:
    return FaceIDService(settings, detector=detector)


@pytest.fixture
def source_for(person_frames):
    def make(person: int) -> ImageSequenceSource:
        return ImageSequenceSource(person_frames[person], loop=True)

    return make


@pytest.fixture
def form_data() -> dict[str, str]:
    return {
        "name": "Ada Lovelace", "phone": "+44 20 7946 0958", "reservation_days": "3",
        "reservation_date": "2030-05-17", "floor": "2", "room": "204", "room_capacity": "2",
    }
