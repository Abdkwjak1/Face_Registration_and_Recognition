import cv2
import numpy as np

from faceid.config import Settings
from faceid.detection import Face, FaceDetector, resize_face, to_gray


def test_blank_frame_has_no_face(detector):
    assert detector.detect(np.full((480, 640, 3), 127, np.uint8)) == []
    assert detector.largest(np.zeros((480, 640), np.uint8)) is None


def test_finds_face_in_bgr_and_gray(detector, person_frames):
    frame = person_frames[1][0]
    assert detector.largest(frame) is not None
    assert detector.largest(to_gray(frame)) is not None


def test_boxes_are_in_full_frame_coordinates(detector, person_frames):
    """Detection runs on a downscaled copy; boxes must map back to the original frame."""
    frame = person_frames[1][0]
    big = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    small_face, big_face = detector.largest(frame), detector.largest(big)
    assert big_face is not None
    assert abs(big_face.x - 2 * small_face.x) <= 0.15 * big_face.w
    assert abs(big_face.w - 2 * small_face.w) <= 0.25 * big_face.w
    h, w = big.shape[:2]
    assert big_face.x + big_face.w <= w and big_face.y + big_face.h <= h


def test_faces_sorted_largest_first(detector, person_frames):
    a = cv2.resize(person_frames[1][0], (320, 240))
    b = cv2.resize(person_frames[2][0], (320, 240))
    both = np.hstack([a, b])
    faces = detector.detect(both)
    assert [f.area for f in faces] == sorted((f.area for f in faces), reverse=True)


def test_min_face_size_filters_tiny_faces(person_frames):
    strict = FaceDetector(Settings(min_face_size=400))
    assert strict.detect(person_frames[1][0]) == []


def test_crop_is_square_gray_of_configured_size(detector, person_frames):
    frame = person_frames[1][0]
    crop = detector.crop(frame, detector.largest(frame))
    assert crop.shape == (detector.settings.face_size,) * 2 and crop.dtype == np.uint8


def test_crop_clamps_box_outside_frame(detector):
    frame = np.full((100, 100, 3), 90, np.uint8)
    crop = detector.crop(frame, Face(-20, -20, 80, 80))
    assert crop.shape == (detector.settings.face_size,) * 2


def test_resize_face_noop_when_already_sized():
    img = np.zeros((160, 160), np.uint8)
    assert resize_face(img, 160) is img
    assert resize_face(img, 80).shape == (80, 80)


def test_face_helpers():
    f = Face(1, 2, 30, 40)
    assert f.area == 1200 and f.box == (1, 2, 30, 40)
