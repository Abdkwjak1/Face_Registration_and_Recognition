"""Main window: live camera preview, guest registration, check-in and guest management.

All camera work happens in :meth:`FaceIDApp.tick`, driven by Tk's ``after`` timer, so the
window never freezes (the original opened blocking OpenCV windows inside the Tk loop).
Model training runs on a worker thread; results come back through a queue polled by ``tick``.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from datetime import date
from enum import Enum
from tkinter import messagebox, ttk

from faceid.camera import CameraError, FrameSource, open_camera
from faceid.config import Settings
from faceid.guests import Guest, ValidationError
from faceid.recognition import Identification
from faceid.service import EnrollmentSession, FaceIDService, RecognitionSession
from faceid.ui import video

PREVIEW_W, PREVIEW_H = 520, 390
TICK_MS = 33
ENROLL_TIMEOUT_S = 30.0

# Palette: black, white, gray and one dark orange accent.
WHITE = "#ffffff"
INK = "#111111"
MUTED = "#6b6b6b"
LINE = "#dcdcdc"
PANEL = "#f4f4f4"
ACCENT = "#b84a00"
ACCENT_DARK = "#8f3a00"
DISABLED = "#bdbdbd"

# key, label, grid row, first column, column span (6-column grid)
FORM_FIELDS = [
    ("name", "Full name", 0, 0, 6),
    ("phone", "Phone number", 1, 0, 6),
    ("reservation_days", "Nights", 2, 0, 2),
    ("reservation_date", "Check-in date", 2, 2, 4),
    ("floor", "Floor", 3, 0, 2),
    ("room", "Room", 3, 2, 2),
    ("room_capacity", "Capacity", 3, 4, 2),
]
DETAIL_ROWS = [
    ("Name", "name"), ("Phone", "phone"), ("Nights", "reservation_days"),
    ("Check-in", "reservation_date"), ("Floor", "floor"), ("Room", "room"),
    ("Capacity", "room_capacity"),
]

CameraFactory = Callable[[], FrameSource]


class Mode(Enum):
    PREVIEW = "preview"
    ENROLL = "enroll"
    TRAINING = "training"
    RECOGNIZE = "recognize"


class FaceIDApp:
    def __init__(
        self,
        root: tk.Misc,
        service: FaceIDService | None = None,
        camera_factory: CameraFactory | None = None,
    ):
        self.root = root
        self.service = service or FaceIDService()
        settings: Settings = self.service.settings
        self._camera_factory = camera_factory or (lambda: open_camera(settings))
        self.camera: FrameSource | None = None
        self.mode = Mode.PREVIEW
        self._enroll: EnrollmentSession | None = None
        self._enroll_name = ""
        self._enroll_started = 0.0
        self._recognize: RecognitionSession | None = None
        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue()
        self._running = False
        self._photo: tk.PhotoImage | None = None

        self._build()
        self.refresh_guests()
        self.open_camera()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = self.root
        root.title("Face ID")
        root.configure(bg=WHITE)
        root.geometry("1100x600")
        root.minsize(1040, 580)

        base = tkfont.nametofont("TkDefaultFont")
        self._fonts = {name: base.copy() for name in ("title", "big", "small", "brand", "tab")}
        self._fonts["title"].configure(size=16, weight="bold")
        self._fonts["big"].configure(size=13, weight="bold")
        self._fonts["brand"].configure(size=12, weight="bold")
        self._fonts["small"].configure(size=9)
        self._fonts["tab"].configure(size=11)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=WHITE, foreground=INK, bordercolor=LINE,
                        focuscolor=WHITE, lightcolor=WHITE, darkcolor=WHITE)
        style.configure("TFrame", background=WHITE)
        style.configure("TLabel", background=WHITE, foreground=INK)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Err.TLabel", foreground=ACCENT, font=self._fonts["small"])
        style.configure("Title.TLabel", font=self._fonts["title"])
        style.configure("Good.TLabel", foreground=INK, font=self._fonts["big"])
        style.configure("Bad.TLabel", foreground=ACCENT, font=self._fonts["big"])

        style.configure("TButton", background=PANEL, foreground=INK, padding=(14, 8),
                        borderwidth=0, relief="flat")
        style.map("TButton", background=[("active", LINE)], foreground=[("disabled", DISABLED)])
        style.configure("Accent.TButton", background=ACCENT, foreground=WHITE, borderwidth=0,
                        bordercolor=ACCENT)
        style.map("Accent.TButton", background=[("disabled", DISABLED), ("active", ACCENT_DARK)],
                  foreground=[("disabled", WHITE)], bordercolor=[("active", ACCENT_DARK)])

        style.configure("TEntry", fieldbackground=WHITE, bordercolor=LINE, padding=6,
                        lightcolor=LINE, darkcolor=LINE, insertcolor=INK)
        style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
                  darkcolor=[("focus", ACCENT)])

        # Tabs are switched by our own flat nav row (see _build_nav); hide ttk's tab strip.
        style.layout("TNotebook.Tab", [])
        style.configure("TNotebook", background=WHITE, borderwidth=0, tabmargins=0,
                        bordercolor=WHITE, lightcolor=WHITE, darkcolor=WHITE)

        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, rowheight=30,
                        borderwidth=0, bordercolor=LINE)
        style.map("Treeview", background=[("selected", INK)], foreground=[("selected", WHITE)])
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED, relief="flat",
                        padding=(8, 6), borderwidth=0)
        style.map("Treeview.Heading", background=[("active", PANEL)])


        header = tk.Frame(root, bg=INK)
        header.pack(fill="x")
        tk.Frame(header, bg=ACCENT, width=10, height=10).pack(side="left", padx=(24, 10), pady=18)
        tk.Label(header, text="FACE ID", bg=INK, fg=WHITE, font=self._fonts["brand"]).pack(
            side="left")
        tk.Label(header, text="Guest check-in", bg=INK, fg=DISABLED).pack(side="left", padx=16)

        tk.Frame(root, bg=LINE, height=1).pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w",
                                     bg=PANEL, fg=MUTED, padx=24, pady=6)
        self.status_label.pack(fill="x", side="bottom")

        body = ttk.Frame(root, padding=(24, 24, 24, 16))
        body.pack(fill="both", expand=True)

        # Left: video
        left = ttk.Frame(body)
        left.pack(side="left", anchor="n")
        self.video_label = tk.Label(left, bg=INK, width=PREVIEW_W, height=PREVIEW_H, bd=0)
        self.video_label.pack()
        self.video_label.configure(image=self._set_image(
            video.placeholder(PREVIEW_W, PREVIEW_H, "Starting camera...")))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = tk.Canvas(left, height=4, bg=LINE, highlightthickness=0, bd=0)
        self.progress.pack(fill="x")
        self._bar = self.progress.create_rectangle(0, 0, 0, 4, fill=ACCENT, width=0)
        self.progress_var.trace_add("write", self._draw_progress)
        self.progress.bind("<Configure>", self._draw_progress)
        self.hint_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.hint_var, style="Muted.TLabel").pack(anchor="w",
                                                                                 pady=(8, 0))
        self.camera_btn = ttk.Button(left, text="Retry camera", command=self.open_camera)

        # Right: tabs
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(32, 0))
        self.tabs = ttk.Notebook(right)
        self.checkin_tab = ttk.Frame(self.tabs, padding=(0, 24, 0, 0))
        self.register_tab = ttk.Frame(self.tabs, padding=(0, 24, 0, 0))
        self.guests_tab = ttk.Frame(self.tabs, padding=(0, 24, 0, 0))
        self.tabs.add(self.checkin_tab, text="Check in")
        self.tabs.add(self.register_tab, text="Register guest")
        self.tabs.add(self.guests_tab, text="Guests")
        self._build_nav(right)
        self.tabs.pack(fill="both", expand=True)
        self._build_checkin()
        self._build_register()
        self._build_guests()

    def _build_nav(self, parent: ttk.Frame) -> None:
        """Flat tab strip: gray labels, the active one black with an orange underline."""
        nav = tk.Frame(parent, bg=WHITE)
        nav.pack(fill="x")
        self._nav: dict[str, tuple[tk.Label, tk.Frame]] = {}
        for tab_id in self.tabs.tabs():
            cell = tk.Frame(nav, bg=WHITE)
            cell.pack(side="left", padx=(0, 28))
            label = tk.Label(cell, text=self.tabs.tab(tab_id, "text"), bg=WHITE, fg=MUTED,
                             font=self._fonts["tab"], cursor="hand2", pady=8)
            label.pack()
            bar = tk.Frame(cell, bg=WHITE, height=2)
            bar.pack(fill="x")
            label.bind("<Button-1>", lambda _e, t=tab_id: self.tabs.select(t))
            self._nav[tab_id] = (label, bar)
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x")
        self.tabs.bind("<<NotebookTabChanged>>", self._sync_nav)
        self._sync_nav()

    def _sync_nav(self, _event: object = None) -> None:
        current = self.tabs.select()
        for tab_id, (label, bar) in self._nav.items():
            on = tab_id == current
            label.configure(fg=INK if on else MUTED)
            bar.configure(bg=ACCENT if on else WHITE)

    def _draw_progress(self, *_: object) -> None:
        width = self.progress.winfo_width()
        self.progress.coords(self._bar, 0, 0, width * self.progress_var.get(), 4)

    def _build_checkin(self) -> None:
        f = self.checkin_tab
        ttk.Label(f, text="Identify a returning guest", style="Title.TLabel").pack(anchor="w")
        ttk.Label(f, text="Look at the camera and press the button. The result is voted "
                          "over several frames.", style="Muted.TLabel",
                  wraplength=420).pack(anchor="w", pady=(2, 12))
        self.checkin_btn = ttk.Button(f, text="Start check-in", style="Accent.TButton",
                                      command=self.start_checkin)
        self.checkin_btn.pack(anchor="w")
        self.result_var = tk.StringVar(value="")
        self.result_label = ttk.Label(f, textvariable=self.result_var, style="Good.TLabel",
                                      wraplength=420)
        self.result_label.pack(anchor="w", pady=(16, 6))
        self.detail_vars: dict[str, tk.StringVar] = {}
        grid = ttk.Frame(f)
        grid.pack(anchor="w")
        for r, (label, key) in enumerate(DETAIL_ROWS):
            ttk.Label(grid, text=label, style="Muted.TLabel", width=10).grid(
                row=r, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="")
            self.detail_vars[key] = var
            ttk.Label(grid, textvariable=var).grid(row=r, column=1, sticky="w", pady=2)

    def _build_register(self) -> None:
        f = self.register_tab
        ttk.Label(f, text="Register a new guest", style="Title.TLabel").pack(anchor="w")
        ttk.Label(f, text="Fill in the details, then look at the camera while samples "
                          "are captured.", style="Muted.TLabel", wraplength=420).pack(
            anchor="w", pady=(2, 10))
        form = ttk.Frame(f)
        form.pack(anchor="w", fill="x")
        for col in range(6):
            form.columnconfigure(col, weight=1, uniform="form")
        self.form_vars: dict[str, tk.StringVar] = {}
        self.form_errors: dict[str, tk.StringVar] = {}
        self.form_entries: dict[str, ttk.Entry] = {}
        for key, label, row, col, span in FORM_FIELDS:
            cell = ttk.Frame(form)
            cell.grid(row=row, column=col, columnspan=span, sticky="ew", pady=(0, 10),
                      padx=(0, 0 if col + span == 6 else 12))
            head = ttk.Frame(cell)
            head.pack(fill="x")
            ttk.Label(head, text=label, style="Muted.TLabel").pack(side="left")
            err = tk.StringVar()
            ttk.Label(head, textvariable=err, style="Err.TLabel").pack(side="right")
            var = tk.StringVar()
            entry = ttk.Entry(cell, textvariable=var)
            entry.pack(fill="x", pady=(3, 0))
            self.form_vars[key], self.form_errors[key], self.form_entries[key] = var, err, entry
        self.form_vars["reservation_date"].set(date.today().isoformat())
        buttons = ttk.Frame(f)
        buttons.pack(anchor="w", pady=(14, 0))
        self.register_btn = ttk.Button(buttons, text="Save and capture face",
                                       style="Accent.TButton", command=self.start_registration)
        self.register_btn.pack(side="left")
        self.cancel_btn = ttk.Button(buttons, text="Cancel", command=self.cancel_registration,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))

    def _build_guests(self) -> None:
        f = self.guests_tab
        ttk.Label(f, text="Registered guests", style="Title.TLabel").pack(anchor="w")
        cols = ("id", "name", "room", "samples")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=12,
                                 selectmode="browse")
        for col, text, width in (("id", "ID", 40), ("name", "Name", 190),
                                 ("room", "Room", 80), ("samples", "Face samples", 100)):
            self.tree.heading(col, text=text, anchor="w")
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=10)
        row = ttk.Frame(f)
        row.pack(anchor="w")
        self.delete_btn = ttk.Button(row, text="Delete selected", command=self.delete_selected)
        self.delete_btn.pack(side="left")
        self.retrain_btn = ttk.Button(row, text="Retrain model", command=self.retrain)
        self.retrain_btn.pack(side="left", padx=(8, 0))

    # ----------------------------------------------------------------- camera
    def _set_image(self, frame) -> tk.PhotoImage:
        self._photo = tk.PhotoImage(data=video.to_ppm(frame, PREVIEW_W, PREVIEW_H))
        return self._photo

    def open_camera(self) -> bool:
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        try:
            self.camera = self._camera_factory()
        except CameraError as exc:
            self._show_frame(video.placeholder(PREVIEW_W, PREVIEW_H, "No camera available"))
            self.camera_btn.pack(anchor="w", pady=(6, 0))
            self.say(str(exc), error=True)
            return False
        self.camera_btn.pack_forget()
        self.say("Camera ready")
        return True

    def _show_frame(self, frame) -> None:
        self.video_label.configure(image=self._set_image(frame))

    def start(self) -> None:
        """Begin the preview loop (the caller then runs ``root.mainloop()``)."""
        self._running = True
        self.root.after(TICK_MS, self._scheduled_tick)

    def stop(self) -> None:
        self._running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def _scheduled_tick(self) -> None:
        if not self._running:
            return
        self.tick()
        self.root.after(TICK_MS, self._scheduled_tick)

    def tick(self) -> None:
        """One iteration of the event loop: run queued jobs, read a frame, advance the mode."""
        while True:
            try:
                self._jobs.get_nowait()()
            except queue.Empty:
                break
        if self.camera is None:
            return
        frame = self.camera.read()
        if frame is None:
            return
        detector = self.service.detector
        face, color, label = None, video.WHITE, ""
        if self.mode is Mode.ENROLL and self._enroll is not None:
            face, color = self._enroll.feed(frame), video.ORANGE
            label = f"{self._enroll.saved}/{self._enroll.target}"
            self.progress_var.set(self._enroll.saved / self._enroll.target)
            if self._enroll.done:
                self._finish_enrollment()
            elif time.monotonic() - self._enroll_started > ENROLL_TIMEOUT_S:
                self.cancel_registration("No face detected for 30 s, registration cancelled")
        elif self.mode is Mode.RECOGNIZE and self._recognize is not None:
            face, color = self._recognize.feed(frame), video.ORANGE
            label = "Checking..."
            self.progress_var.set(min(1.0, self._recognize.collected / self._recognize.target))
            if self._recognize.done:
                self._finish_checkin()
        else:
            face = detector.largest(frame)
        self._show_frame(video.annotate(frame, face, color, label))

    def run_in_background(self, work: Callable[[], object], done: Callable[[object], None],
                          fail: Callable[[BaseException], None]) -> threading.Thread:
        def target() -> None:
            try:
                result = work()
            except BaseException as exc:  # reported on the UI thread
                self._jobs.put(lambda exc=exc: fail(exc))
            else:
                self._jobs.put(lambda: done(result))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    # --------------------------------------------------------------- helpers
    def say(self, text: str, error: bool = False) -> None:
        self.status_var.set(text)
        self.status_label.configure(fg=ACCENT if error else MUTED)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for btn in (self.checkin_btn, self.register_btn, self.delete_btn, self.retrain_btn):
            btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if self.mode is Mode.ENROLL else "disabled")

    def _set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self._set_busy(mode is not Mode.PREVIEW)
        if mode is Mode.PREVIEW:
            self.progress_var.set(0)
            self.hint_var.set("")

    # ---------------------------------------------------------- registration
    def collect_form(self) -> dict[str, str]:
        return {key: var.get() for key, var in self.form_vars.items()}

    def start_registration(self) -> bool:
        """Validate the form, store the guest and begin capturing face samples."""
        for err in self.form_errors.values():
            err.set("")
        if self.camera is None:
            self.say("Camera not available: cannot capture a face", error=True)
            return False
        try:
            guest = Guest.from_form(self.collect_form())
        except ValidationError as exc:
            for key, message in exc.errors.items():
                self.form_errors[key].set(message)
            self.form_entries[next(iter(exc.errors))].focus_set()
            self.say("Please fix the highlighted fields", error=True)
            return False
        stored = self.service.register(guest)
        self._enroll = self.service.start_enrollment(stored.id)
        self._enroll_name = stored.name
        self._enroll_started = time.monotonic()
        self._set_mode(Mode.ENROLL)
        self.hint_var.set("Look at the camera and move your head slightly")
        self.say(f"Capturing face samples for {stored.name}...")
        return True

    def cancel_registration(self, message: str = "Registration cancelled") -> None:
        if self.mode is not Mode.ENROLL or self._enroll is None:
            return
        self.service.remove_guest(self._enroll.guest_id)
        self._enroll = None
        self._set_mode(Mode.PREVIEW)
        self.refresh_guests()
        self.say(message)

    def _finish_enrollment(self) -> None:
        name, guest_id = self._enroll_name, self._enroll.guest_id
        self._enroll = None
        self._set_mode(Mode.TRAINING)
        self.hint_var.set("Training the recognizer...")
        self.say("Training the recognizer...")

        def done(_: object) -> None:
            self._set_mode(Mode.PREVIEW)
            for var in self.form_vars.values():
                var.set("")
            self.form_vars["reservation_date"].set(date.today().isoformat())
            self.refresh_guests()
            self.say(f"{name} registered as guest #{guest_id}")

        def fail(exc: BaseException) -> None:
            self._set_mode(Mode.PREVIEW)
            self.say(f"Training failed: {exc}", error=True)

        self.run_in_background(self.service.train, done, fail)

    # --------------------------------------------------------------- check-in
    def start_checkin(self) -> bool:
        if self.camera is None:
            self.say("Camera not available", error=True)
            return False
        if not self.service.recognizer.is_trained:
            self.say("Nobody is registered yet: register a guest first", error=True)
            return False
        self._clear_result()
        self._recognize = self.service.start_recognition()
        self._set_mode(Mode.RECOGNIZE)
        self.hint_var.set("Hold still and look at the camera")
        self.say("Checking...")
        return True

    def _finish_checkin(self) -> None:
        result = self._recognize.result()
        self._recognize = None
        self._set_mode(Mode.PREVIEW)
        self.show_result(result)

    def _clear_result(self) -> None:
        self.result_var.set("")
        for var in self.detail_vars.values():
            var.set("")

    def show_result(self, result: Identification) -> None:
        self._clear_result()
        guest = self.service.lookup(result)
        if result.frames == 0:
            self.result_label.configure(style="Bad.TLabel")
            self.result_var.set("No face detected. Face the camera and try again.")
            self.say("No face detected", error=True)
        elif guest is None and result.known:
            self.result_label.configure(style="Bad.TLabel")
            self.result_var.set(f"Recognised guest #{result.guest_id}, but the record is missing.")
            self.say("Guest record missing", error=True)
        elif guest is None:
            self.result_label.configure(style="Bad.TLabel")
            self.result_var.set("Not recognised. Please register as a new guest.")
            self.say("Not recognised", error=True)
        else:
            self.result_label.configure(style="Good.TLabel")
            self.result_var.set(f"Welcome back, {guest.name}")
            row = guest.to_row()
            for key, var in self.detail_vars.items():
                var.set(row[key])
            self.say(f"Recognised {guest.name} ({result.votes}/{result.frames} frames, "
                     f"distance {result.distance:.0f})")

    # ----------------------------------------------------------------- guests
    def refresh_guests(self) -> None:
        self.tree.delete(*self.tree.get_children())
        counts = self.service.dataset.counts()
        for g in self.service.guests.all():
            self.tree.insert("", "end", iid=str(g.id),
                             values=(g.id, g.name, g.room, counts.get(g.id, 0)))

    def delete_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            self.say("Select a guest first")
            return
        guest_id = int(selected[0])
        guest = self.service.guests.get(guest_id)
        if guest is None or not messagebox.askyesno(
                "Delete guest", f"Delete {guest.name} and all their face samples?"):
            return
        self.service.remove_guest(guest_id)
        self.refresh_guests()
        self.retrain()

    def retrain(self) -> None:
        self._set_mode(Mode.TRAINING)
        self.say("Training the recognizer...")

        def done(count: object) -> None:
            self._set_mode(Mode.PREVIEW)
            self.say(f"Model trained on {count} samples" if count else "No samples: model cleared")

        def fail(exc: BaseException) -> None:
            self._set_mode(Mode.PREVIEW)
            self.say(f"Training failed: {exc}", error=True)

        self.run_in_background(self.service.train, done, fail)


def run(settings: Settings | None = None) -> None:
    root = tk.Tk()
    app = FaceIDApp(root, FaceIDService(settings))
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    app.start()
    root.mainloop()


__all__ = ["FaceIDApp", "Mode", "run"]
