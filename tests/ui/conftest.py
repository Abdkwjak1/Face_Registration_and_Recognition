"""UI tests drive the real Tk window with a scripted camera. Skipped when no display exists."""

from __future__ import annotations

import tkinter as tk

import pytest

from faceid.camera import CameraError
from faceid.ui.app import FaceIDApp, Mode


def _display_available() -> bool:
    try:
        root = tk.Tk()
    except tk.TclError:
        return False
    root.destroy()
    return True


DISPLAY_OK = _display_available()


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="no display available for Tk")
    for item in items:
        if "ui" in str(item.fspath).replace("\\", "/").split("/tests/")[-1].split("/")[:1]:
            item.add_marker(pytest.mark.ui)
            if not DISPLAY_OK:
                item.add_marker(skip)


@pytest.fixture
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


class Driver:
    """Test helper wrapping the app: pump ticks until a condition holds."""

    def __init__(self, app: FaceIDApp, root: tk.Tk):
        self.app, self.root = app, root

    def pump(self, until=lambda: False, limit: int = 400) -> int:
        for n in range(limit):
            self.app.tick()
            self.root.update()
            if until():
                return n
        raise AssertionError("condition not reached within tick budget")

    def settle(self) -> None:
        self.pump(until=lambda: self.app.mode is Mode.PREVIEW)

    def fill(self, data: dict[str, str]) -> None:
        for key, value in data.items():
            self.app.form_vars[key].set(value)


@pytest.fixture
def make_app(root, service):
    apps = []

    def make(source=None, service_=None):
        def factory():
            if source is None:
                raise CameraError("no camera for test")
            return source

        app = FaceIDApp(root, service_ or service, camera_factory=factory)
        apps.append(app)
        return Driver(app, root)

    yield make
    for app in apps:
        app.stop()
