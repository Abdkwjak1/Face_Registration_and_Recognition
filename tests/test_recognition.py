import numpy as np
import pytest

from faceid.config import Settings
from faceid.recognition import FaceRecognizer, NotTrainedError, Prediction


@pytest.fixture(scope="module")
def trained(labelled_faces):
    faces, ids = labelled_faces
    rec = FaceRecognizer(Settings())
    rec.train(faces, ids)
    return rec


def test_untrained_recognizer_refuses_to_predict(labelled_faces):
    rec = FaceRecognizer()
    assert not rec.is_trained
    with pytest.raises(NotTrainedError):
        rec.predict(labelled_faces[0][0])


def test_training_on_nothing_is_an_error():
    with pytest.raises(ValueError):
        FaceRecognizer().train([], np.array([]))


def test_predicts_training_samples_as_their_owner(trained, labelled_faces):
    faces, ids = labelled_faces
    for face, gid in zip(faces, ids, strict=True):
        p = trained.predict(face)
        assert p.label == gid
        assert p.distance < 5  # seen sample: near-zero distance


def test_identify_accepts_known_person(trained, labelled_faces):
    faces, ids = labelled_faces
    mine = [f for f, i in zip(faces, ids, strict=True) if i == 2]
    r = trained.identify(mine)
    assert r.known and r.guest_id == 2 and r.votes == r.frames == len(mine)


def test_identify_with_no_faces_is_unknown(trained):
    r = trained.identify([])
    assert not r.known and r.frames == 0 and r.distance == float("inf")


def test_unseen_person_is_rejected(labelled_faces):
    """Train on person 1 only; person 2 must come back unknown, not as person 1."""
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    rec.train([f for f, i in zip(faces, ids, strict=True) if i == 1], ids[ids == 1])
    strangers = [f for f, i in zip(faces, ids, strict=True) if i == 2]
    assert not rec.identify(strangers).known


def test_noise_is_rejected(trained):
    rng = np.random.default_rng(1)
    noise = [(rng.random((160, 160)) * 255).astype(np.uint8) for _ in range(10)]
    assert not trained.identify(noise).known


class TestDecide:
    """Voting logic in isolation, with hand-made predictions."""

    @pytest.fixture
    def rec(self):
        return FaceRecognizer(Settings(confidence_threshold=60.0, min_votes=0.5))

    def test_majority_wins_and_outlier_ignored(self, rec):
        preds = [Prediction(1, 40)] * 7 + [Prediction(2, 45)] * 2 + [Prediction(1, 90)]
        r = rec.decide(preds)
        assert r.guest_id == 1 and r.votes == 7 and r.frames == 10
        assert r.distance == pytest.approx(40)

    def test_all_above_threshold_is_unknown(self, rec):
        r = rec.decide([Prediction(1, 61)] * 5)
        assert r.guest_id is None and r.votes == 0

    def test_threshold_is_inclusive(self, rec):
        assert rec.decide([Prediction(1, 60)] * 4).guest_id == 1

    def test_split_vote_below_min_votes_is_unknown(self, rec):
        preds = [Prediction(1, 40)] * 2 + [Prediction(2, 40)] * 2 + [Prediction(3, 40)] * 2
        assert rec.decide(preds).guest_id is None

    def test_few_accepted_among_many_rejected_is_unknown(self, rec):
        preds = [Prediction(1, 40)] * 2 + [Prediction(1, 99)] * 8
        assert rec.decide(preds).guest_id is None


def test_save_load_round_trip(trained, labelled_faces, tmp_path):
    faces, _ = labelled_faces
    path = tmp_path / "sub" / "model.yml"
    trained.save(path)
    clone = FaceRecognizer()
    clone.load(path)
    assert clone.is_trained
    for face in faces[::7]:
        assert clone.predict(face) == trained.predict(face)


def test_load_missing_model(tmp_path):
    with pytest.raises(NotTrainedError):
        FaceRecognizer().load(tmp_path / "nope.yml")


def test_reset_forgets_model(labelled_faces):
    faces, ids = labelled_faces
    rec = FaceRecognizer()
    rec.train(faces, ids)
    rec.reset()
    assert not rec.is_trained


def test_preprocess_is_size_invariant(trained):
    img = (np.random.default_rng(2).random((300, 300)) * 255).astype(np.uint8)
    assert trained.preprocess(img).shape == (160, 160)
