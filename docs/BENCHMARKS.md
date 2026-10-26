# Benchmarks

Reproduce: `faceid benchmark tests/fixtures/faces`, `faceid evaluate tests/fixtures/faces`, `pytest -m perf`.

**Machine**: AMD Ryzen 5 4600H (6 cores / 12 threads), Linux, Python 3.11, OpenCV 4.14 (CPU only).
Numbers are means of 50 runs after warm-up; three consecutive runs agreed within ±3 %.

## Speed

| Stage | Mean | p95 |
|---|---|---|
| Detect, 640×480 frame (Haar, downscaled to ≤ 480 px) | 6.3 ms | 8.9 ms |
| Crop + preprocess (resize + CLAHE) | 0.04 ms | 0.05 ms |
| Predict, one face, 2 guests | 1.9 ms | 2.1 ms |
| **Detect → crop → predict** | **7.9 ms** | 8.4 ms |
| Train, 42 samples | 60 ms | 62 ms |

A webcam delivers 30 fps (33 ms per frame); the pipeline uses about a quarter of that budget.

### Scaling with enrolled guests (20 samples each)

LBPH compares a probe against every stored histogram, so both costs grow linearly.

| Guests | Samples | Train | Predict (1 face) |
|---|---|---|---|
| 2 | 40 | 60 ms | 2.0 ms |
| 10 | 200 | 285 ms | 4.3 ms |
| 50 | 1 000 | 1.4 s | 13 ms |
| 100 | 2 000 | 2.8 s | 25 ms |

Up to a few hundred guests this is comfortably real-time. Beyond that, reduce `samples_per_guest` or
swap the recogniser (see *Limitations*). Timings for 10+ guests replicate the bundled samples, so they
measure cost, not accuracy.

## Accuracy

Data: 42 face crops of 2 people (21 each), `tests/fixtures/faces`. Face ID was evaluated two ways:

* **Genuine**: stratified 5-fold cross-validation, trained on 4/5, tested on the held-out 1/5.
* **Impostor**: leave-one-person-out. Train on person A only, test on person B. Every such sample *should* be rejected.

| Metric | Result |
|---|---|
| Closed-set identification accuracy (nearest guest is right) | 100 % (42/42) |
| Genuine distances (min / median / max) | 33.8 / 38.4 / 62.1 |
| Impostor distances (min / median / max) | 69.3 / 74.3 / 102.9 |

The two distributions don't overlap, which is why the default threshold sits between them:

| Threshold | True-accept | False-accept |
|---|---|---|
| 50 | 95.2 % | 0.0 % |
| 60 | 97.6 % | 0.0 % |
| **65 (default)** | **100 %** | **0.0 %** |
| 70 | 100 % | 11.9 % |
| 80 | 100 % | 71.4 % |

For comparison, v0 had no threshold at all: its false-accept rate was 100 % by construction.

### Read this before trusting those numbers

* **Tiny and optimistic.** Two people, and each person's 21 crops are consecutive frames from one short capture
  (same lighting, almost the same pose). Cross-validation folds are therefore near-duplicates of the training
  data, which flatters accuracy. Expect lower numbers across days, lighting and head poses.
* **Impostor evidence is thin.** One stranger per model. 0 % FAR here is "no failures in 42 trials", not a guarantee.
* **Threshold is deployment-specific.** If you enrol in one setting and check in in another (different lights,
  camera), run `faceid evaluate` on your own crops and re-pick it: lower = stricter, fewer false accepts,
  more "not recognised".

Voting over ~12 frames (`recognition_frames`) improves on the per-frame figures above, but that gain
is not quantified here.

## Performance regression tests

`pytest -m perf` fails if any of these drift badly (limits are loose to tolerate slow CI machines):

| Test | Budget |
|---|---|
| Detection | < 33 ms/frame (real time at 30 fps) |
| Prediction (42 samples) | < 20 ms |
| Prediction (420 samples) | < 50 ms |
| Training (42 samples) | < 1.5 s; 10× the data < 25× the time |
| Detect + crop + predict | > 20 fps |
| Cross-validated accuracy | ≥ 95 %; at the default threshold TAR ≥ 95 % and FAR ≤ 5 % |

## Limitations

LBPH is chosen for being CPU-light, explainable and trainable from a handful of images, not for being
state of the art. Deep-embedding recognisers (ArcFace, FaceNet…) are far more robust to pose and lighting, but need
a model download and more compute. `FaceRecognizer` has a small surface (`train`, `predict`, `decide`,
`save`, `load`), so replacing it is a contained change.
