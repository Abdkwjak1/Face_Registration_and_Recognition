"""faceid: offline face registration and recognition for guest check-in."""

from faceid.config import Settings
from faceid.detection import Face, FaceDetector
from faceid.guests import Guest, GuestRepository
from faceid.recognition import FaceRecognizer, Identification, Prediction
from faceid.service import FaceIDService

__version__ = "1.0.0"

__all__ = [
    "Face",
    "FaceDetector",
    "FaceIDService",
    "FaceRecognizer",
    "Guest",
    "GuestRepository",
    "Identification",
    "Prediction",
    "Settings",
]
