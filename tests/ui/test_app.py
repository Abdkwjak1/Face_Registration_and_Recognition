from tkinter import messagebox

import pytest

from faceid.guests import Guest
from faceid.ui.app import Mode


def test_window_builds_with_three_tabs(make_app, source_for):
    d = make_app(source_for(1))
    assert [d.app.tabs.tab(t, "text") for t in d.app.tabs.tabs()] == [
        "Check in", "Register guest", "Guests"]
    assert d.app.mode is Mode.PREVIEW
    assert d.app.status_var.get() == "Camera ready"


def test_preview_renders_frames(make_app, source_for):
    d = make_app(source_for(1))
    for _ in range(3):
        d.app.tick()
    assert d.app._photo.width() > 0 and d.app._photo.height() > 0


def test_no_camera_degrades_gracefully(make_app, form_data):
    d = make_app(None)
    assert d.app.camera is None
    assert "no camera" in d.app.status_var.get().lower()
    assert d.app.camera_btn.winfo_manager() == "pack"  # retry button visible
    d.app.tick()  # must not raise
    d.fill(form_data)
    assert d.app.start_registration() is False
    assert d.app.service.guests.all() == []  # no half-registered guest
    assert d.app.start_checkin() is False


def test_retry_camera_after_failure(make_app, source_for, root, service):
    state = {"source": None}
    from faceid.camera import CameraError
    from faceid.ui.app import FaceIDApp

    def factory():
        if state["source"] is None:
            raise CameraError("unplugged")
        return state["source"]

    app = FaceIDApp(root, service, camera_factory=factory)
    assert app.camera is None
    state["source"] = source_for(1)
    assert app.open_camera() is True
    assert app.camera is not None and app.camera_btn.winfo_manager() == ""
    app.stop()


def test_invalid_form_shows_inline_errors_and_saves_nothing(make_app, source_for):
    d = make_app(source_for(1))
    d.fill({"name": "", "phone": "12", "reservation_days": "0"})
    assert d.app.start_registration() is False
    errors = {k: v.get() for k, v in d.app.form_errors.items() if v.get()}
    assert {"name", "phone", "reservation_days", "room"} <= set(errors)
    assert d.app.mode is Mode.PREVIEW and d.app.service.guests.all() == []
    # errors clear once the form is fixed
    d.fill({"name": "Ada", "phone": "555 123 456", "reservation_days": "2", "floor": "1",
            "room": "5", "room_capacity": "2"})
    assert d.app.start_registration() is True
    assert not any(v.get() for v in d.app.form_errors.values())


def test_registration_flow_end_to_end(make_app, source_for, form_data):
    d = make_app(source_for(1))
    d.fill(form_data)
    assert d.app.start_registration() is True
    assert d.app.mode is Mode.ENROLL and str(d.app.cancel_btn["state"]) == "normal"
    assert str(d.app.register_btn["state"]) == "disabled"

    d.pump(until=lambda: d.app.mode is Mode.TRAINING)
    assert d.app.progress_var.get() == pytest.approx(1.0)
    d.settle()

    guests = d.app.service.guests.all()
    assert [g.name for g in guests] == ["Ada Lovelace"]
    assert d.app.service.dataset.count(guests[0].id) == 20
    assert d.app.service.recognizer.is_trained
    assert "registered as guest #1" in d.app.status_var.get()
    assert d.app.form_vars["name"].get() == ""  # form reset for the next guest
    assert d.app.tree.get_children() == ("1",)
    assert str(d.app.register_btn["state"]) == "normal"


def test_cancel_registration_rolls_back(make_app, source_for, form_data):
    d = make_app(source_for(1))
    d.fill(form_data)
    d.app.start_registration()
    for _ in range(3):
        d.app.tick()
    d.app.cancel_registration()
    assert d.app.mode is Mode.PREVIEW
    assert d.app.service.guests.all() == []
    assert d.app.service.dataset.counts() == {}
    assert str(d.app.cancel_btn["state"]) == "disabled"


def test_registration_times_out_without_a_face(make_app, source_for, form_data, monkeypatch):
    import numpy as np

    from faceid.camera import ImageSequenceSource

    blank = ImageSequenceSource([np.full((480, 640, 3), 127, np.uint8)])
    d = make_app(blank)
    d.fill(form_data)
    d.app.start_registration()
    monkeypatch.setattr("faceid.ui.app.ENROLL_TIMEOUT_S", -1)
    d.settle()
    assert "cancelled" in d.app.status_var.get()
    assert d.app.service.guests.all() == []


def _register(d, form_data, name):
    d.fill({**form_data, "name": name})
    assert d.app.start_registration()
    d.settle()


def test_checkin_recognises_registered_guest(make_app, source_for, form_data):
    d = make_app(source_for(1))
    _register(d, form_data, "Ada Lovelace")
    assert d.app.start_checkin() is True
    assert d.app.mode is Mode.RECOGNIZE
    d.settle()
    assert d.app.result_var.get() == "Welcome back, Ada Lovelace"
    assert d.app.detail_vars["room"].get() == "204"
    assert d.app.detail_vars["reservation_date"].get() == "2030-05-17"
    assert "Recognised Ada Lovelace" in d.app.status_var.get()


def test_checkin_rejects_stranger(make_app, source_for, form_data, service):
    d = make_app(source_for(1))
    _register(d, form_data, "Ada Lovelace")
    d.app.camera = source_for(2)  # a different person steps up
    d.app.start_checkin()
    d.settle()
    assert "Not recognised" in d.app.result_var.get()
    assert d.app.detail_vars["name"].get() == ""


def test_checkin_result_clears_previous_guest(make_app, source_for, form_data):
    d = make_app(source_for(1))
    _register(d, form_data, "Ada Lovelace")
    d.app.start_checkin()
    d.settle()
    assert d.app.detail_vars["name"].get() == "Ada Lovelace"
    d.app.camera = source_for(2)
    d.app.start_checkin()
    assert d.app.detail_vars["name"].get() == ""  # cleared as soon as a new check starts


def test_checkin_without_registrations(make_app, source_for):
    d = make_app(source_for(1))
    assert d.app.start_checkin() is False
    assert "register a guest first" in d.app.status_var.get()


def test_checkin_when_nobody_in_view(make_app, source_for, form_data):
    import numpy as np

    from faceid.camera import ImageSequenceSource

    d = make_app(source_for(1))
    _register(d, form_data, "Ada")
    d.app.camera = ImageSequenceSource([np.full((480, 640, 3), 127, np.uint8)])
    d.app.start_checkin()
    t = {"now": 0.0}
    session = d.app._recognize
    session._clock = lambda: t["now"]
    session._deadline = 1.0
    t["now"] = 5.0
    d.settle()
    assert "No face detected" in d.app.result_var.get()


def test_guests_tab_lists_and_deletes(make_app, source_for, form_data, monkeypatch):
    d = make_app(source_for(1))
    _register(d, form_data, "Ada")
    _register(d, form_data, "Bob")
    assert d.app.tree.item("2", "values")[1] == "Bob"
    assert d.app.tree.item("2", "values")[3] in (20, "20")

    d.app.delete_selected()  # nothing selected
    assert "Select a guest" in d.app.status_var.get()

    d.app.tree.selection_set("1")
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: False)
    d.app.delete_selected()
    assert len(d.app.service.guests.all()) == 2  # declined

    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    d.app.delete_selected()
    d.settle()
    assert [g.name for g in d.app.service.guests.all()] == ["Bob"]
    assert d.app.tree.get_children() == ("2",)
    assert d.app.service.dataset.count(1) == 0
    assert "trained on 20 samples" in d.app.status_var.get()


def test_deleting_last_guest_clears_model(make_app, source_for, form_data, monkeypatch):
    d = make_app(source_for(1))
    _register(d, form_data, "Ada")
    d.app.tree.selection_set("1")
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    d.app.delete_selected()
    d.settle()
    assert not d.app.service.recognizer.is_trained
    assert d.app.start_checkin() is False


def test_training_failure_is_reported_not_raised(make_app, source_for, monkeypatch):
    d = make_app(source_for(1))

    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr(d.app.service, "train", boom)
    d.app.retrain()
    d.settle()
    assert "Training failed: disk full" in d.app.status_var.get()
    assert d.app.mode is Mode.PREVIEW


def test_existing_guests_are_listed_on_startup(make_app, service, form_data, source_for):
    g = service.register(Guest.from_form(form_data))
    service.enroll(g.id, source_for(1))
    d = make_app(source_for(1))
    assert d.app.tree.get_children() == (str(g.id),)
    assert d.app.tree.item(str(g.id), "values")[3] in (20, "20")
