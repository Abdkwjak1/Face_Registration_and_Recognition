# Reference

## Command line

Global options go before the subcommand: `faceid --home /srv/faceid --camera 2 gui`.

```bash
faceid                               # open the app (same as `faceid gui`)
faceid status                        # guests / samples / model state
faceid train                         # retrain from stored samples
faceid identify                      # who is in front of the camera?
faceid identify --image photo.jpg    # ...or in a still image
faceid evaluate path/to/crops        # k-fold accuracy and TAR/FAR by threshold
faceid benchmark path/to/crops       # per-stage timings
```

`evaluate` and `benchmark` read a folder of face crops named `user.<id>.<n>.png|jpg`.

## Python API

```python
from faceid import FaceIDService, Guest
from faceid.camera import open_camera

service = FaceIDService()                # data under ./data (or $FACEID_HOME)
guest = service.register(Guest.from_form({
    "name": "Ada Lovelace", "phone": "+44 20 7946 0958", "reservation_days": "3",
    "reservation_date": "2030-05-17", "floor": "2", "room": "204", "room_capacity": "2"}))

with open_camera() as cam:
    service.enroll(guest.id, cam)        # captures 20 face samples
    service.train()
    result = service.identify(cam)       # votes over live frames

print(service.lookup(result) or "unknown")
```

Any object with `read()` and `release()` works as a camera, e.g. `ImageSequenceSource` for recorded frames.
For a GUI, use `service.start_enrollment()` / `start_recognition()` and call `session.feed(frame)` per frame.

## Settings

Everything tunable is a field of [`Settings`](../src/faceid/config.py):

| Setting | Default | Meaning |
|---|---|---|
| `camera_index` | auto | Camera to open; `None` probes 0-3 |
| `samples_per_guest` | 20 | Face samples captured at registration |
| `confidence_threshold` | 65 | LBPH distance above which a frame doesn't count as a match (lower is stricter) |
| `recognition_frames` | 12 | Frames voted over per check-in |
| `min_votes` | 0.5 | Share of frames that must agree on one guest |
| `recognition_timeout_s` | 8 | Give up waiting for faces after this long |
| `min_face_size` | 80 | Faces smaller than this (px) are ignored |
| `face_size` | 160 | Side of the normalised face crop |

## Project layout

```
src/faceid/
  config.py       Settings
  camera.py       webcam wrapper, auto-probe, ImageSequenceSource
  detection.py    Haar detector, crop normalisation
  recognition.py  LBPH wrapper, unknown rejection, voting
  dataset.py      face-sample store (user.<id>.<n>.png)
  guests.py       Guest model, validation, CSV repository
  service.py      enrolment / recognition sessions, training, lookup
  evaluation.py   cross-validation, FAR/TAR sweep, timers
  cli.py          `faceid` command
  ui/             Tk app and frame helpers
tests/            unit, workflow, ui/ (real Tk), test_performance.py
docs/             this documentation and screenshots
scripts/          make_screenshots.py
```

## Development

```bash
pytest                      # everything
pytest -m "not ui"          # headless
pytest -m perf              # speed and accuracy budgets
pytest --cov=faceid         # coverage
ruff check .                # lint
python scripts/make_screenshots.py   # refresh docs/img (needs ImageMagick `import`)
```

The UI palette (black, white, gray, one dark orange) is defined at the top of `src/faceid/ui/app.py`.

## Coming from the first version

`Face_ID.py` and `Fc_Rec/` were replaced by the `faceid` package. Old `dataset/user.N.M.jpg` files and the
old CSV header (`Name,Phone,Res_D,...`) still load: copy them into `data/dataset/` and `data/guests.csv`,
then run `faceid train`. The old `trainingData.yml` is incompatible and not needed.
