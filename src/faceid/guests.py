"""Guest records: validation and CSV persistence."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from pathlib import Path

FIELDS = [
    "id", "name", "phone", "reservation_days", "reservation_date", "floor", "room", "room_capacity",
]
# Column names used by the first version of the app; still accepted when reading.
_LEGACY = {
    "ID": "id", "Name": "name", "Phone": "phone", "Res_D": "reservation_days",
    "Date_R": "reservation_date", "Floor_N": "floor", "Room_N": "room", "Room_C": "room_capacity",
}
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
_PHONE = re.compile(r"^\+?[\d\s().-]{5,20}$")


class ValidationError(ValueError):
    """The submitted guest data is invalid. ``errors`` maps field name -> message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


@dataclass(frozen=True)
class Guest:
    name: str
    phone: str
    reservation_days: int
    reservation_date: date
    floor: int
    room: str
    room_capacity: int
    id: int = 0

    @classmethod
    def from_form(cls, raw: dict[str, str], id: int = 0) -> Guest:
        """Validate free-text form input; raises ``ValidationError`` listing every bad field."""
        errors: dict[str, str] = {}
        raw = {k: (raw.get(k) or "").strip() for k in FIELDS}

        def integer(key: str, minimum: int) -> int:
            try:
                value = int(raw[key])
            except ValueError:
                errors[key] = "must be a whole number"
                return 0
            if value < minimum:
                errors[key] = f"must be at least {minimum}"
            return value

        if not raw["name"]:
            errors["name"] = "required"
        if not _PHONE.match(raw["phone"]):
            errors["phone"] = "enter 5-20 digits (spaces, +, -, ( ) allowed)"
        days = integer("reservation_days", 1)
        floor = integer("floor", 0)
        capacity = integer("room_capacity", 1)
        if not raw["room"]:
            errors["room"] = "required"
        when = _parse_date(raw["reservation_date"])
        if when is None:
            errors["reservation_date"] = "use YYYY-MM-DD"
        if errors:
            raise ValidationError(errors)
        return cls(raw["name"], raw["phone"], days, when, floor, raw["room"], capacity, id)

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        row["reservation_date"] = self.reservation_date.isoformat()
        return {k: str(row[k]) for k in FIELDS}

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Guest:
        row = {_LEGACY.get(k, k): v for k, v in row.items()}
        when = _parse_date(row.get("reservation_date", ""))
        return cls(
            name=row["name"],
            phone=row["phone"],
            reservation_days=int(row["reservation_days"]),
            reservation_date=when or date.min,
            floor=int(row["floor"]),
            room=row["room"],
            room_capacity=int(row["room_capacity"]),
            id=int(row["id"]),
        )


def _parse_date(text: str) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


class GuestRepository:
    """CSV-backed guest store. IDs are ``max(existing) + 1`` and never derived from row counts."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def all(self) -> list[Guest]:
        if not self.path.exists():
            return []
        with self.path.open(newline="", encoding="utf-8") as f:
            return [Guest.from_row(row) for row in csv.DictReader(f) if any(row.values())]

    def get(self, guest_id: int) -> Guest | None:
        return next((g for g in self.all() if g.id == guest_id), None)

    def add(self, guest: Guest) -> Guest:
        guests = self.all()
        new_id = max((g.id for g in guests), default=0) + 1
        stored = Guest(**{**{f.name: getattr(guest, f.name) for f in fields(Guest)}, "id": new_id})
        self._write([*guests, stored])
        return stored

    def delete(self, guest_id: int) -> bool:
        guests = self.all()
        kept = [g for g in guests if g.id != guest_id]
        if len(kept) == len(guests):
            return False
        self._write(kept)
        return True

    def _write(self, guests: list[Guest]) -> None:
        """Write via temp file + rename so a crash can never leave a half-written CSV."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(g.to_row() for g in guests)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
