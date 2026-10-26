"""Speed and accuracy budgets. Limits are deliberately loose (CI machines vary) but would
catch an order-of-magnitude regression. Real measurements live in docs/BENCHMARKS.md."""

import numpy as np
import pytest

from faceid.config import Settings
from faceid.evaluation import collect_distances, identification_accuracy, sweep, time_it
from faceid.recognition import FaceRecognizer

pytestmark = pytest.mark.perf


def test_detection_keeps_up_with_30fps(detector, person_frames):
    frame = person_frames[1][0]
    assert time_it(lambda: detector.detect(frame), repeats=30).mean_ms < 33


def test_prediction_latency(labelled_faces):
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    rec.train(faces, ids)
    assert time_it(lambda: rec.predict(faces[0]), repeats=50).mean_ms < 20


def test_training_time_for_small_deployment(labelled_faces):
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    assert time_it(lambda: rec.train(faces, ids), repeats=3, warmup=1).mean_ms < 1500


def test_training_scales_roughly_linearly(labelled_faces):
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    one = time_it(lambda: rec.train(faces, ids), repeats=3, warmup=1).mean_ms
    tenfold = time_it(lambda: rec.train(faces * 10, np.tile(ids, 10)), repeats=2, warmup=0).mean_ms
    assert tenfold < one * 25  # 10x the data must not cost anywhere near 100x


def test_prediction_cost_grows_with_enrolled_samples(labelled_faces):
    """LBPH compares against every stored histogram: 420 samples must still be real-time."""
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    rec.train(faces * 10, np.tile(ids, 10))
    assert time_it(lambda: rec.predict(faces[0]), repeats=20).mean_ms < 50


def test_end_to_end_frame_rate(detector, person_frames, labelled_faces):
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    rec.train(faces, ids)
    frame = person_frames[1][0]

    def step():
        face = detector.largest(frame)
        return rec.predict(detector.crop(frame, face))

    assert time_it(step, repeats=30).per_second > 20


def test_cross_validated_accuracy_and_operating_point(labelled_faces):
    d = collect_distances(*labelled_faces, k=5, settings=Settings())
    assert identification_accuracy(d) >= 0.95
    (at_default,) = sweep(d, [Settings().confidence_threshold])
    assert at_default.tar >= 0.95, "default threshold rejects too many genuine faces"
    assert at_default.far <= 0.05, "default threshold accepts too many strangers"
