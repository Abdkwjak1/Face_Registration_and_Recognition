# Privacy and limits

## Data
Face images are biometric personal data. Face ID keeps everything local, in the data folder:
`guests.csv`, `dataset/` (face samples) and `model.yml`. Nothing is sent anywhere.

- Get explicit consent before enrolling someone, and tell them how long you keep the data.
- Deleting a guest in the **Guests** tab removes their record and all their face samples, then retrains.
- Back up, protect or delete the data folder like any other personal data. It is git-ignored.
- Check local rules (for example GDPR) before using this with real guests.

## What it is not
- **Not a security control.** There is no liveness or anti-spoofing check; a photo of a guest can fool it. Use it to look up a reservation, never to grant access.
- **Sensitive to conditions.** Strong head turns, heavy occlusion and very different lighting between registration and check-in lower accuracy.
- **Classical method.** LBPH is light and explainable, but less robust than deep-embedding recognisers. See the limitations in [BENCHMARKS.md](BENCHMARKS.md).
