import numpy as np
import pytest

from faceid.evaluation import (
    Distances,
    ThresholdPoint,
    _folds,
    benchmark_pipeline,
    collect_distances,
    identification_accuracy,
    sweep,
    synthetic_frame,
    time_it,
)


def test_folds_are_stratified_and_exhaustive():
    ids = np.array([1] * 10 + [2] * 5)
    folds = _folds(ids, 5)
    assert sorted(np.concatenate(folds).tolist()) == list(range(15))
    for fold in folds:
        assert set(ids[fold]) == {1, 2}


def test_sweep_math():
    d = Distances(genuine=[(1, 1, 30.0), (1, 1, 50.0), (2, 1, 40.0)], impostor=[45.0, 90.0])
    points = {p.threshold: p for p in sweep(d, [20, 45, 100])}
    assert points[20] == ThresholdPoint(20, 0.0, 0.0)
    assert points[45].tar == pytest.approx(1 / 3) and points[45].far == pytest.approx(0.5)
    assert points[100].tar == pytest.approx(2 / 3) and points[100].far == 1.0
    assert identification_accuracy(d) == pytest.approx(2 / 3)


def test_no_impostors_gives_nan_far():
    (p,) = sweep(Distances([(1, 1, 1.0)], []), [10])
    assert np.isnan(p.far)


def test_real_data_separates_genuine_from_impostor(labelled_faces):
    d = collect_distances(*labelled_faces, k=3)
    assert len(d.genuine) == 42 and len(d.impostor) == 42
    assert max(x[2] for x in d.genuine) < np.percentile(d.impostor, 25)


def test_time_it():
    t = time_it(lambda: sum(range(1000)), repeats=10)
    assert t.mean_ms > 0 and t.p95_ms >= 0 and t.per_second > 0


def test_synthetic_frame_shape(labelled_faces):
    f = synthetic_frame(labelled_faces[0][0], (320, 240))
    assert f.shape == (240, 320, 3)


def test_benchmark_rejects_empty_dataset(tmp_path):
    with pytest.raises(ValueError):
        benchmark_pipeline(tmp_path)
