"""End-to-end workflows through the service layer, using fixture faces as the 'camera'."""

import dataclasses

import pytest

from faceid.camera import ImageSequenceSource
from faceid.guests import Guest
from faceid.recognition import NotTrainedError
from faceid.service import FaceIDService


def register(service, form_data, source, name):
    guest = service.register(Guest.from_form({**form_data, "name": name}))
    saved = service.enroll(guest.id, source)
    return guest, saved


def test_register_enroll_train_identify(service, form_data, source_for):
    ada, n1 = register(service, form_data, source_for(1), "Ada")
    bob, n2 = register(service, form_data, source_for(2), "Bob")
    assert n1 == n2 == service.settings.samples_per_guest
    assert service.train() == n1 + n2

    for person, expected in ((1, ada), (2, bob)):
        result = service.identify(source_for(person))
        assert result.known and result.guest_id == expected.id
        assert service.lookup(result).name == expected.name


def test_stranger_is_unknown(service, form_data, source_for):
    register(service, form_data, source_for(1), "Ada")
    service.train()
    result = service.identify(source_for(2))
    assert not result.known and result.frames > 0
    assert service.lookup(result) is None


def test_identify_requires_model(service, source_for):
    with pytest.raises(NotTrainedError):
        service.identify(source_for(1))


def test_no_face_in_view(service, form_data, source_for):
    import numpy as np

    register(service, form_data, source_for(1), "Ada")
    service.train()
    blank = ImageSequenceSource([np.full((480, 640, 3), 127, np.uint8)])
    result = service.identify(blank, max_frames=30)
    assert result.frames == 0 and not result.known


def test_recognition_times_out_without_faces(service, form_data, source_for):
    import numpy as np

    register(service, form_data, source_for(1), "Ada")
    service.train()
    now = [0.0]
    session = service.start_recognition(clock=lambda: now[0])
    assert not session.done
    session.feed(np.full((480, 640, 3), 127, np.uint8))
    now[0] += service.settings.recognition_timeout_s + 1
    assert session.timed_out and session.done


def test_enrollment_respects_sample_interval(settings, detector, source_for, form_data):
    spaced = FaceIDService(dataclasses.replace(settings, sample_interval_s=0.5), detector=detector)
    guest = spaced.register(Guest.from_form(form_data))
    now = [0.0]
    session = spaced.start_enrollment(guest.id, clock=lambda: now[0])
    src = source_for(1)
    for _ in range(5):  # five frames in the same instant -> one sample
        session.feed(src.read())
    assert session.saved == 1
    now[0] += 0.6
    session.feed(src.read())
    assert session.saved == 2


def test_enrollment_stops_at_target(service, source_for, form_data):
    guest = service.register(Guest.from_form(form_data))
    session = service.start_enrollment(guest.id)
    src = source_for(1)
    while not session.done:
        session.feed(src.read())
    assert session.feed(src.read()) is None
    assert service.dataset.count(guest.id) == service.settings.samples_per_guest


def test_model_persists_across_restarts(service, settings, form_data, source_for, detector):
    register(service, form_data, source_for(1), "Ada")
    service.train()
    restarted = FaceIDService(settings, detector=detector)
    assert restarted.recognizer.is_trained
    assert restarted.identify(source_for(1)).known


def test_remove_guest_then_retrain(service, form_data, source_for):
    ada, _ = register(service, form_data, source_for(1), "Ada")
    bob, _ = register(service, form_data, source_for(2), "Bob")
    service.train()
    assert service.remove_guest(ada.id)
    assert service.dataset.count(ada.id) == 0 and service.guests.get(ada.id) is None
    service.train()
    assert not service.identify(source_for(1)).known
    assert service.identify(source_for(2)).guest_id == bob.id


def test_removing_last_guest_clears_model(service, settings, form_data, source_for):
    ada, _ = register(service, form_data, source_for(1), "Ada")
    service.train()
    service.remove_guest(ada.id)
    assert service.train() == 0
    assert not service.recognizer.is_trained and not settings.model_path.exists()


def test_status(service, form_data, source_for):
    assert (service.status().guests, service.status().model_ready) == (0, False)
    register(service, form_data, source_for(1), "Ada")
    service.train()
    s = service.status()
    assert (s.guests, s.samples, s.model_ready) == (1, service.settings.samples_per_guest, True)


def test_ids_are_stable_when_registering_after_delete(service, form_data, source_for):
    """Regression: IDs used to come from the CSV row count and collided after deletions."""
    a, _ = register(service, form_data, source_for(1), "A")
    b, _ = register(service, form_data, source_for(2), "B")
    service.remove_guest(a.id)
    c, _ = register(service, form_data, source_for(1), "C")
    assert len({b.id, c.id}) == 2
    assert service.dataset.count(b.id) == service.dataset.count(c.id) == 20


def test_v0_dataset_and_csv_can_be_imported(service, settings, fixtures_dir, source_for):
    """README promise: old dataset/user.N.M.jpg + legacy CSV work after copying them in."""
    import shutil

    settings.dataset_dir.mkdir(parents=True)
    for path in fixtures_dir.glob("user.*.jpg"):
        shutil.copy(path, settings.dataset_dir)
    settings.guests_csv.write_text(
        "Name,Phone,Res_D,Date_R,Floor_N,Room_N,Room_C,ID\n"
        "Old One,555123456,2,2024-01-02,3,301,4,1\n"
        "Old Two,555987654,1,2024-02-03,1,101,2,2\n", encoding="utf-8")
    assert service.train() == 42
    result = service.identify(source_for(2))
    assert service.lookup(result).name == "Old Two"
