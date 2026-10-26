import cv2
import pytest

from faceid import cli
from faceid.guests import Guest


@pytest.fixture
def home_args(settings):
    return ["--home", str(settings.home)]


@pytest.fixture
def trained_home(service, form_data, source_for, home_args):
    guest = service.register(Guest.from_form(form_data))
    service.enroll(guest.id, source_for(1))
    service.train()
    return home_args


def test_status_on_empty_home(home_args, capsys):
    assert cli.main([*home_args, "status"]) == 0
    out = capsys.readouterr().out
    assert "guests:   0" in out and "not trained" in out


def test_train_without_samples_fails_cleanly(home_args, capsys):
    assert cli.main([*home_args, "train"]) == 1
    assert "No samples" in capsys.readouterr().out


def test_identify_without_model(home_args, capsys):
    assert cli.main([*home_args, "identify", "--image", "x.png"]) == 2
    assert "No trained model" in capsys.readouterr().err


def test_identify_image_known_and_unknown(trained_home, person_frames, tmp_path, capsys):
    mine, other = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(mine), person_frames[1][0])
    cv2.imwrite(str(other), person_frames[2][0])
    assert cli.main([*trained_home, "identify", "--image", str(mine)]) == 0
    assert "Ada Lovelace" in capsys.readouterr().out
    assert cli.main([*trained_home, "identify", "--image", str(other)]) == 1
    assert "Unknown" in capsys.readouterr().out


def test_identify_unreadable_image(trained_home, tmp_path, capsys):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert cli.main([*trained_home, "identify", "--image", str(bad)]) == 2


def test_identify_without_camera(trained_home, capsys):
    assert cli.main(["--camera", "987", *trained_home, "identify"]) == 2
    assert "No working camera" in capsys.readouterr().err


def test_evaluate_report(fixtures_dir, capsys):
    assert cli.main(["evaluate", str(fixtures_dir)]) == 0
    out = capsys.readouterr().out
    assert "identification accuracy: 100.0%" in out and "threshold" in out


def test_evaluate_needs_data(tmp_path, capsys):
    assert cli.main(["evaluate", str(tmp_path)]) == 2


def test_benchmark_prints_every_stage(fixtures_dir, capsys):
    assert cli.main(["benchmark", str(fixtures_dir)]) == 0
    out = capsys.readouterr().out
    for stage in ("train", "detect (640x480 frame)", "predict (1 face)", "detect + crop + predict"):
        assert stage in out
