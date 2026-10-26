from datetime import date

import pytest

from faceid.guests import FIELDS, Guest, GuestRepository, ValidationError


def test_from_form_parses_and_trims(form_data):
    form_data["name"] = "  Ada Lovelace "
    g = Guest.from_form(form_data)
    assert g.name == "Ada Lovelace"
    assert g.reservation_date == date(2030, 5, 17)
    assert (g.reservation_days, g.floor, g.room_capacity) == (3, 2, 2)


@pytest.mark.parametrize("text", ["2030-05-17", "17/05/2030", "17-05-2030", "17.05.2030"])
def test_date_formats_normalised_to_iso(form_data, text):
    form_data["reservation_date"] = text
    assert Guest.from_form(form_data).to_row()["reservation_date"] == "2030-05-17"


def test_validation_reports_every_bad_field():
    with pytest.raises(ValidationError) as exc:
        Guest.from_form({"name": "", "phone": "abc", "reservation_days": "0",
                         "reservation_date": "tomorrow", "floor": "x", "room": "",
                         "room_capacity": "-1"})
    assert set(exc.value.errors) == {
        "name", "phone", "reservation_days", "reservation_date", "floor", "room", "room_capacity"}


@pytest.mark.parametrize(("key", "value"), [
    ("phone", "123"), ("reservation_days", "1.5"), ("floor", "-1"), ("room_capacity", "0"),
    ("reservation_date", "2030-13-40"),
])
def test_single_invalid_field(form_data, key, value):
    form_data[key] = value
    with pytest.raises(ValidationError) as exc:
        Guest.from_form(form_data)
    assert list(exc.value.errors) == [key]


def test_floor_zero_is_ground_floor(form_data):
    form_data["floor"] = "0"
    assert Guest.from_form(form_data).floor == 0


def test_repository_round_trip(tmp_path, form_data):
    repo = GuestRepository(tmp_path / "g.csv")
    assert repo.all() == []
    stored = repo.add(Guest.from_form(form_data))
    assert stored.id == 1
    assert repo.get(1) == stored
    assert repo.get(99) is None
    header = (tmp_path / "g.csv").read_text().splitlines()[0]
    assert header == ",".join(FIELDS)


def test_ids_survive_deletion(tmp_path, form_data):
    """The original derived IDs from the row count, so deleting then adding collided."""
    repo = GuestRepository(tmp_path / "g.csv")
    ids = [repo.add(Guest.from_form(form_data)).id for _ in range(3)]
    assert ids == [1, 2, 3]
    assert repo.delete(2) is True
    assert repo.add(Guest.from_form(form_data)).id == 4
    assert [g.id for g in repo.all()] == [1, 3, 4]


def test_delete_missing_returns_false(tmp_path, form_data):
    repo = GuestRepository(tmp_path / "g.csv")
    repo.add(Guest.from_form(form_data))
    assert repo.delete(42) is False


def test_commas_quotes_and_unicode_survive(tmp_path, form_data):
    form_data["name"] = 'Zoë "Z", Müller-Åström'
    repo = GuestRepository(tmp_path / "g.csv")
    repo.add(Guest.from_form(form_data))
    assert repo.get(1).name == 'Zoë "Z", Müller-Åström'


def test_reads_legacy_csv(tmp_path):
    legacy = tmp_path / "Data.csv"
    legacy.write_text(
        "Name,Phone,Res_D,Date_R,Floor_N,Room_N,Room_C,ID\n"
        "Sam,555123456,2,2024-01-02,3,301,4,7\n", encoding="utf-8")
    (g,) = GuestRepository(legacy).all()
    assert (g.id, g.name, g.room, g.room_capacity) == (7, "Sam", "301", 4)


def test_failed_write_leaves_original_intact(tmp_path, form_data, monkeypatch):
    repo = GuestRepository(tmp_path / "g.csv")
    repo.add(Guest.from_form(form_data))
    before = (tmp_path / "g.csv").read_text()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("faceid.guests.os.replace", boom)
    with pytest.raises(OSError):
        repo.add(Guest.from_form(form_data))
    assert (tmp_path / "g.csv").read_text() == before
    assert list(tmp_path.glob("*.tmp")) == []
