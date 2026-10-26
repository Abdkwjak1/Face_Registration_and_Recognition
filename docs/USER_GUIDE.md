# User guide

## Install

```bash
conda env create -f environment.yml && conda activate face-id
pip install -e .
faceid
```

Linux needs read access to the camera (`ls -l /dev/video0`; add yourself to the `video` group if needed).
Close other apps that hold the camera (video calls, browser tabs).

## Daily use

### Registering a guest
1. Open **Register guest** and fill in all seven fields. Invalid fields get a red message under them.
   Dates can be typed as `2030-05-17`, `17/05/2030`, `17-05-2030` or `17.05.2030`.
2. Press **Save and capture face**. The box around your face turns orange and the counter shows `n/20`.
3. Look at the camera and move your head slightly (a little left, right, up, down) so the samples differ.
   Good, even lighting on the face matters more than anything else.
4. When it reaches 20/20 the model retrains by itself; the status bar confirms *"registered as guest #N"*.

**Cancel** during capture removes the half-registered guest. If no face is seen for 30 s the registration cancels itself.

### Checking a guest in
**Check in → Start check-in.** The orange box counts frames; the result card shows the reservation or
*"Not recognised"*. For an unrecognised returning guest, check lighting/distance and try again before re-registering.

### Managing guests
**Guests** tab: select a row → **Delete selected** (asks for confirmation; removes the record and all face
samples, then retrains). **Retrain model** rebuilds from stored samples, e.g. after copying in samples by hand.

## Where is my data?

Everything lives in the data folder (`./data`, or `$FACEID_HOME`, or `faceid --home PATH`):

```
data/
  guests.csv         guest records (opens in Excel/LibreOffice)
  dataset/           user.<id>.<n>.png face samples
  model.yml          trained LBPH model
```

Back up the folder to back up everything. Delete it to start clean. It is git-ignored.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No camera available" | Plug in / free the camera, press **Retry camera**, or force an index: `faceid --camera 2` |
| Camera works but the box never appears | Face the camera, add light in front of you, move closer (faces under 80 px are ignored; see `min_face_size`) |
| Known guests show as "Not recognised" | Re-register in the lighting you check in under, or raise `confidence_threshold` a little (check FAR with `faceid evaluate`) |
| Strangers are accepted | Lower `confidence_threshold`; register more varied samples |
| "Nobody is registered yet" | Register one guest first (training happens automatically) |
| UI tests skipped | No display detected; run on a desktop session or under `xvfb-run` |
