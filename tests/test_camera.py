import dataclasses

import numpy as np
import pytest

from faceid.camera import Camera, CameraError, ImageSequenceSource, open_camera
from faceid.config import Settings


def frame(v):
    return np.full((4, 4, 3), v, np.uint8)


def test_sequence_loops():
    src = ImageSequenceSource([frame(1), frame(2)])
    assert [int(src.read()[0, 0, 0]) for _ in range(5)] == [1, 2, 1, 2, 1]


def test_sequence_without_loop_ends_with_none():
    src = ImageSequenceSource([frame(1)], loop=False)
    assert src.read() is not None and src.read() is None


def test_sequence_returns_copies():
    src = ImageSequenceSource([frame(1)])
    src.read()[:] = 99
    assert int(src.read()[0, 0, 0]) == 1


def test_empty_sequence_rejected():
    with pytest.raises(ValueError):
        ImageSequenceSource([])


def test_camera_error_for_missing_device():
    with pytest.raises(CameraError):
        Camera(987)


def test_open_camera_reports_clear_error_when_none_work():
    settings = dataclasses.replace(Settings(), camera_index=987)
    with pytest.raises(CameraError, match="No working camera"):
        open_camera(settings)
