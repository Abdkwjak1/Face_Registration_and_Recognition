import cv2
import numpy as np

from faceid.dataset import FaceDataset


def face(value=100, size=160):
    return np.full((size, size), value, np.uint8)


def test_save_numbers_samples_per_guest(tmp_path):
    ds = FaceDataset(tmp_path / "ds")
    names = [ds.save(1, face()).name, ds.save(1, face()).name, ds.save(2, face()).name]
    assert names == ["user.1.1.png", "user.1.2.png", "user.2.1.png"]
    assert ds.counts() == {1: 2, 2: 1}
    assert ds.count(1) == 2


def test_numbering_continues_after_gap(tmp_path):
    ds = FaceDataset(tmp_path)
    ds.save(1, face())
    ds.save(1, face())
    (tmp_path / "user.1.1.png").unlink()
    assert ds.save(1, face()).name == "user.1.3.png"


def test_load_returns_faces_and_ids_resized(tmp_path):
    ds = FaceDataset(tmp_path, face_size=64)
    ds.save(3, face(size=200))
    faces, ids = ds.load()
    assert ids.tolist() == [3]
    assert faces[0].shape == (64, 64)


def test_load_ignores_foreign_and_corrupt_files(tmp_path):
    ds = FaceDataset(tmp_path)
    ds.save(1, face())
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "user.x.1.png").write_bytes(b"")
    (tmp_path / "user.2.1.png").write_bytes(b"not an image")
    faces, ids = ds.load()
    assert ids.tolist() == [1] and len(faces) == 1


def test_loads_legacy_jpg_names(tmp_path, fixtures_dir):
    faces, ids = FaceDataset(fixtures_dir).load()
    assert len(faces) == 42
    assert sorted(set(ids.tolist())) == [1, 2]


def test_delete_guest_removes_only_theirs(tmp_path):
    ds = FaceDataset(tmp_path)
    for gid in (1, 1, 2):
        ds.save(gid, face())
    assert ds.delete_guest(1) == 2
    assert ds.counts() == {2: 1}


def test_missing_directory_is_empty(tmp_path):
    ds = FaceDataset(tmp_path / "nope")
    faces, ids = ds.load()
    assert faces == [] and len(ids) == 0 and ds.counts() == {}


def test_saved_png_is_lossless(tmp_path):
    ds = FaceDataset(tmp_path)
    img = (np.random.default_rng(0).random((160, 160)) * 255).astype(np.uint8)
    path = ds.save(1, img)
    assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), img)
