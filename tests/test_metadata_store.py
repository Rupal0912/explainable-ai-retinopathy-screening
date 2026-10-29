"""MongoDB integration coverage using the disposable local retinopathy_test database."""
from datetime import datetime
import time
import uuid

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError


MONGO_URI = "mongodb://localhost:27017"
DATABASE = "retinopathy_test"


@pytest.fixture(scope="module", autouse=True)
def require_mongodb_and_drop_database():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        client.close()
        pytest.skip("Local MongoDB at localhost:27017 is unavailable.")
    yield
    client.drop_database(DATABASE)
    client.close()


@pytest.fixture(autouse=True)
def configure_test_database(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", MONGO_URI)
    monkeypatch.setenv("MONGODB_DB", DATABASE)


def _demo_id() -> str:
    return f"demo-test-{uuid.uuid4().hex[:12]}"


def _save(demo_id: str, grade: int = 2, confidence: float = 0.8, **extra) -> None:
    from metadata_store import save_record

    ok, message = save_record({"demo_id": demo_id, "grade": grade,
                               "confidence": confidence, **extra})
    assert ok, message


def test_create_and_read_records_newest_first():
    from metadata_store import list_records

    demo_id = _demo_id()
    _save(demo_id, label="older", image_bytes=b"not persisted")
    time.sleep(0.02)
    _save(demo_id, grade=3, label="newer")

    records, error = list_records(demo_id)
    assert error is None
    assert [row["label"] for row in records] == ["newer", "older"]
    assert all(isinstance(row["id"], str) for row in records)
    assert all("image_bytes" not in row for row in records)
    assert records[0]["created_at"] >= records[1]["created_at"]


def test_update_note_and_delete_record():
    from metadata_store import delete_record, list_records, update_note

    demo_id = _demo_id()
    _save(demo_id)
    records, error = list_records(demo_id)
    assert error is None and len(records) == 1
    record_id = records[0]["id"]

    ok, message = update_note(record_id, "Reviewed for project demonstration.")
    assert ok, message
    updated, error = list_records(demo_id)
    assert error is None
    assert updated[0]["reviewer_note"] == "Reviewed for project demonstration."
    assert isinstance(updated[0]["updated_at"], datetime)
    assert len(updated[0]["reviewer_note"]) <= 500

    ok, message = delete_record(record_id)
    assert ok, message
    remaining, error = list_records(demo_id)
    assert error is None and remaining == []


def test_stats_group_by_grade_with_average_confidence():
    from metadata_store import stats

    demo_id = _demo_id()
    _save(demo_id, grade=2, confidence=0.6)
    _save(demo_id, grade=2, confidence=0.8)
    _save(demo_id, grade=4, confidence=0.5)

    rows, error = stats(demo_id)
    assert error is None
    assert rows == [
        {"grade": 2, "count": 2, "average_confidence": pytest.approx(0.7)},
        {"grade": 4, "count": 1, "average_confidence": pytest.approx(0.5)},
    ]


def test_invalid_demo_id_is_rejected():
    from metadata_store import list_records, save_record

    ok, message = save_record({"demo_id": "invalid-id", "grade": 1})
    assert not ok and "Invalid demo ID" in message
    records, error = list_records("invalid-id")
    assert records == [] and error is not None


def test_invalid_record_id_is_rejected():
    from metadata_store import delete_record, update_note

    ok, message = update_note("not-an-object-id", "note")
    assert not ok and "Invalid record ID" in message
    ok, message = delete_record("not-an-object-id")
    assert not ok and "Invalid record ID" in message


def test_unconfigured_mongodb_is_distinct_from_empty_history(monkeypatch):
    from metadata_store import history, list_records, save_record, stats

    monkeypatch.setenv("MONGODB_URI", "")
    demo_id = _demo_id()
    ok, message = save_record({"demo_id": demo_id, "grade": 0})
    assert not ok and "not configured" in message
    records, error = list_records(demo_id)
    assert records == [] and error is not None and "not configured" in error
    assert history(demo_id) == []
    rows, error = stats(demo_id)
    assert rows == [] and error is not None and "not configured" in error


def test_unreachable_mongodb_returns_an_error_not_empty_history(monkeypatch):
    from metadata_store import list_records

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:1")
    records, error = list_records(_demo_id())
    assert records == []
    assert error is not None and "unavailable" in error
