import numpy as np

from faceid.detection import Face
from faceid.ui import video


def test_to_ppm_header_and_payload():
    frame = np.zeros((100, 200, 3), np.uint8)
    data = video.to_ppm(frame, 100, 100)  # fits width: 100 x 50
    header, payload = data.split(b"\n", 1)
    assert header == b"P6 100 50 255"
    assert len(payload) == 100 * 50 * 3


def test_to_ppm_swaps_bgr_to_rgb():
    frame = np.zeros((2, 2, 3), np.uint8)
    frame[..., 0] = 255  # blue in BGR
    payload = video.to_ppm(frame, 2, 2).split(b"\n", 1)[1]
    assert payload[:3] == bytes([0, 0, 255])


def test_annotate_does_not_mutate_input():
    frame = np.zeros((100, 100, 3), np.uint8)
    out = video.annotate(frame, Face(10, 20, 30, 30), video.WHITE, "x")
    assert frame.sum() == 0 and out.sum() > 0
    assert video.annotate(frame, None, video.WHITE).sum() == 0


def test_placeholder_size():
    assert video.placeholder(64, 48, "hi").shape == (48, 64, 3)
