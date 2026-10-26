"""Command line entry point: ``faceid`` (or ``python -m faceid``)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2

from faceid.camera import CameraError, ImageSequenceSource, open_camera
from faceid.config import Settings
from faceid.service import FaceIDService


def _service(args: argparse.Namespace) -> FaceIDService:
    overrides = {}
    if args.home:
        overrides["home"] = Path(args.home).expanduser().resolve()
    if args.camera is not None:
        overrides["camera_index"] = args.camera
    return FaceIDService(dataclasses.replace(Settings(), **overrides))


def cmd_gui(args: argparse.Namespace) -> int:
    from faceid.ui.app import run

    service = _service(args)
    run(service.settings)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    service = _service(args)
    status = service.status()
    print(f"home:     {service.settings.home}")
    print(f"guests:   {status.guests}")
    print(f"samples:  {status.samples}")
    print(f"model:    {'ready' if status.model_ready else 'not trained'}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    count = _service(args).train()
    print(f"Trained on {count} samples" if count else "No samples found; model cleared")
    return 0 if count else 1


def cmd_identify(args: argparse.Namespace) -> int:
    service = _service(args)
    if not service.recognizer.is_trained:
        print("No trained model. Register a guest and run `faceid train` first.", file=sys.stderr)
        return 2
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image {args.image}", file=sys.stderr)
            return 2
        source = ImageSequenceSource([frame])
    else:
        try:
            source = open_camera(service.settings)
        except CameraError as exc:
            print(exc, file=sys.stderr)
            return 2
    try:
        result = service.identify(source, max_frames=300)
    finally:
        source.release()
    guest = service.lookup(result)
    if guest is None:
        print(f"Unknown ({result.votes}/{result.frames} frames matched)")
        return 1
    print(f"{guest.name} (guest #{guest.id}, room {guest.room}) "
          f"- {result.votes}/{result.frames} frames, distance {result.distance:.1f}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from faceid.dataset import FaceDataset
    from faceid.evaluation import collect_distances, identification_accuracy, sweep

    settings = Settings()
    faces, ids = FaceDataset(Path(args.dataset), settings.face_size).load()
    if len(faces) < 10:
        print("Need at least 10 labelled samples to evaluate", file=sys.stderr)
        return 2
    d = collect_distances(faces, ids, k=args.folds, settings=settings)
    print(f"{len(faces)} samples, {len(set(ids.tolist()))} people, {args.folds}-fold CV")
    print(f"Closed-set identification accuracy: {identification_accuracy(d):.1%}")
    print(f"{'threshold':>9}  {'TAR':>6}  {'FAR':>6}")
    for p in sweep(d, [40, 50, 55, 60, 65, 70, 75, 80, 90]):
        print(f"{p.threshold:>9.0f}  {p.tar:>6.1%}  {p.far:>6.1%}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from faceid.evaluation import benchmark_pipeline

    for name, timing in benchmark_pipeline(Path(args.dataset)).items():
        print(f"{name:<28} {timing.mean_ms:8.2f} ms  (p95 {timing.p95_ms:.2f}, "
              f"{timing.per_second:,.0f}/s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faceid", description=__doc__)
    parser.add_argument("--home", help="data folder (default: $FACEID_HOME or ./data)")
    parser.add_argument("--camera", type=int, help="camera index (default: auto-detect)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="open the desktop app (default)").set_defaults(func=cmd_gui)
    sub.add_parser("status", help="show guest / sample / model counts").set_defaults(
        func=cmd_status)
    sub.add_parser("train", help="retrain the model from stored samples").set_defaults(
        func=cmd_train)
    ident = sub.add_parser("identify", help="identify who is in front of the camera or in an image")
    ident.add_argument("--image", help="identify a face in this image instead of the camera")
    ident.set_defaults(func=cmd_identify)
    ev = sub.add_parser("evaluate", help="accuracy report on a folder of user.<id>.<n>.* crops")
    ev.add_argument("dataset")
    ev.add_argument("--folds", type=int, default=5)
    ev.set_defaults(func=cmd_evaluate)
    bench = sub.add_parser("benchmark", help="measure detection / recognition speed")
    bench.add_argument("dataset")
    bench.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return getattr(args, "func", cmd_gui)(args)


if __name__ == "__main__":
    sys.exit(main())
