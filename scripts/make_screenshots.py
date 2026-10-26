"""Render README screenshots of the real Tk window (needs a display).

The "camera" is a synthetic frame so no personal photos end up in the docs.
Needs ImageMagick's ``import`` on PATH.
Usage: python scripts/make_screenshots.py
"""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

from faceid.camera import ImageSequenceSource
from faceid.config import Settings
from faceid.guests import Guest
from faceid.recognition import Identification
from faceid.service import FaceIDService
from faceid.ui import video
from faceid.ui.app import FaceIDApp

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"


def scene_frame() -> np.ndarray:
    frame = video.placeholder(640, 480, "")
    cv2.putText(frame, "live camera preview", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (170, 170, 170), 1, cv2.LINE_AA)
    return frame


def grab(root: tk.Tk, name: str) -> None:
    root.update()
    root.update_idletasks()
    # PIL's ImageGrab fails on XWayland; ImageMagick's per-window capture works.
    subprocess.run(["import", "-window", hex(root.winfo_id()), str(OUT / name)], check=True)
    print("wrote", OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp())
    service = FaceIDService(dataclasses.replace(Settings(), home=home))
    demo = [
        ("Amira Haddad", "+49 151 2345 6789", 3, "2030-06-14", 2, "204", 2),
        ("Jonas Weber", "+49 170 9876 5432", 1, "2030-06-15", 0, "012", 1),
        ("Lina Petrova", "+359 88 123 4567", 5, "2030-06-16", 4, "418", 3),
    ]
    for name, phone, days, when, floor, room, cap in demo:
        guest = service.register(Guest.from_form(dict(
            name=name, phone=phone, reservation_days=str(days), reservation_date=when,
            floor=str(floor), room=room, room_capacity=str(cap))))
        for _ in range(20):
            service.dataset.save(guest.id, np.full((160, 160), 100 + guest.id, np.uint8))

    root = tk.Tk()
    app = FaceIDApp(root, service, camera_factory=lambda: ImageSequenceSource([scene_frame()]))
    root.update()
    app.tick()

    app.tabs.select(app.checkin_tab)
    app.show_result(Identification(1, 38.0, 12, 12))
    grab(root, "checkin.png")

    app.tabs.select(app.register_tab)
    for key, val in zip(app.form_vars, ["Noah Fischer", "+49 160 5550 123", "2", "2030-07-01",
                                        "3", "305", "2"], strict=True):
        app.form_vars[key].set(val)
    app.form_vars["phone"].set("12")
    app.start_registration()
    grab(root, "register.png")

    app.form_vars["phone"].set("+49 160 5550 123")
    app.tabs.select(app.guests_tab)
    app.refresh_guests()
    app.say("3 guests registered, model trained on 60 samples")
    grab(root, "guests.png")
    root.destroy()


if __name__ == "__main__":
    main()
