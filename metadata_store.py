"""MongoDB persistence for screening metadata; uploaded image bytes are never stored."""
import os
import re
from datetime import datetime, timezone
from threading import Lock

from bson import ObjectId
from bson.errors import InvalidDocument
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


DEMO_ID_PATTERN = re.compile(r"demo-[A-Za-z0-9_-]{1,48}")
_client: MongoClient | None = None
_client_uri: str | None = None
_client_generation = 0
_client_lock = Lock()
_indexed_databases: set[tuple[int, str]] = set()


def _screenings() -> tuple[Collection | None, str | None]:
    """Return the shared screenings collection, creating its index once per DB."""
    global _client, _client_uri, _client_generation
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None, "MongoDB is not configured; database operation was not completed."

    db_name = os.getenv("MONGODB_DB", "retinopathy_demo")
    try:
        with _client_lock:
            if _client is None or _client_uri != uri:
                if _client is not None:
                    _client.close()
                _client = MongoClient(uri, serverSelectionTimeoutMS=2500)
                _client_uri = uri
                _client_generation += 1
            key = (_client_generation, db_name)
            collection = _client[db_name]["screenings"]
            if key not in _indexed_databases:
                collection.create_index(
                    [("demo_id", ASCENDING), ("created_at", DESCENDING)],
                    name="demo_id_created_at_desc",
                )
                _indexed_databases.add(key)
        return collection, None
    except PyMongoError:
        return None, "MongoDB is unavailable; database operation was not completed."


def _valid_demo_id(demo_id: str) -> bool:
    return isinstance(demo_id, str) and DEMO_ID_PATTERN.fullmatch(demo_id) is not None


def _valid_record_id(record_id: str) -> bool:
    return isinstance(record_id, str) and ObjectId.is_valid(record_id)


def save_record(record: dict) -> tuple[bool, str]:
    """Insert screening metadata, excluding any uploaded image content."""
    demo_id = record.get("demo_id") if isinstance(record, dict) else None
    if not _valid_demo_id(demo_id):
        return False, "Invalid demo ID; use the format demo-001."
    collection, error = _screenings()
    if error:
        return False, error

    doc = {key: value for key, value in record.items() if key not in {"image_bytes", "_id"}}
    doc["created_at"] = datetime.now(timezone.utc)
    try:
        collection.insert_one(doc)
        return True, "Screening metadata saved."
    except (PyMongoError, InvalidDocument):
        return False, "MongoDB is unavailable; result was not saved."


def list_records(demo_id: str) -> tuple[list[dict], str | None]:
    """Return up to 50 newest records and distinguish errors from an empty result."""
    if not _valid_demo_id(demo_id):
        return [], "Invalid demo ID; use the format demo-001."
    collection, error = _screenings()
    if error:
        return [], error
    try:
        records = list(collection.find({"demo_id": demo_id}).sort("created_at", DESCENDING).limit(50))
        for record in records:
            record["id"] = str(record.pop("_id"))
        return records, None
    except PyMongoError:
        return [], "MongoDB is unavailable; records could not be loaded."


def history(demo_id: str) -> list[dict]:
    """Compatibility wrapper returning only records."""
    records, _error = list_records(demo_id)
    return records


def update_note(record_id: str, note: str) -> tuple[bool, str]:
    """Update a record's reviewer note; notes are limited to 500 characters."""
    if not _valid_record_id(record_id):
        return False, "Invalid record ID."
    if not isinstance(note, str) or len(note) > 500:
        return False, "Reviewer note must be text of at most 500 characters."
    collection, error = _screenings()
    if error:
        return False, error
    try:
        result = collection.update_one(
            {"_id": ObjectId(record_id)},
            {"$set": {"reviewer_note": note, "updated_at": datetime.now(timezone.utc)}},
        )
        if result.matched_count == 0:
            return False, "Screening record was not found."
        return True, "Reviewer note updated."
    except PyMongoError:
        return False, "MongoDB is unavailable; reviewer note was not updated."


def delete_record(record_id: str) -> tuple[bool, str]:
    """Delete one screening record by its MongoDB ObjectId."""
    if not _valid_record_id(record_id):
        return False, "Invalid record ID."
    collection, error = _screenings()
    if error:
        return False, error
    try:
        result = collection.delete_one({"_id": ObjectId(record_id)})
        if result.deleted_count == 0:
            return False, "Screening record was not found."
        return True, "Screening record deleted."
    except PyMongoError:
        return False, "MongoDB is unavailable; screening record was not deleted."


def stats(demo_id: str | None = None) -> tuple[list[dict], str | None]:
    """Group screening counts and mean confidence by grade, optionally per demo ID."""
    if demo_id is not None and not _valid_demo_id(demo_id):
        return [], "Invalid demo ID; use the format demo-001."
    collection, error = _screenings()
    if error:
        return [], error

    pipeline = []
    if demo_id is not None:
        pipeline.append({"$match": {"demo_id": demo_id}})
    pipeline.extend([
        {"$group": {
            "_id": "$grade",
            "count": {"$sum": 1},
            "average_confidence": {"$avg": "$confidence"},
        }},
        {"$sort": {"_id": ASCENDING}},
    ])
    try:
        rows = list(collection.aggregate(pipeline))
        return [{"grade": row["_id"], "count": row["count"],
                 "average_confidence": row.get("average_confidence")} for row in rows], None
    except PyMongoError:
        return [], "MongoDB is unavailable; statistics could not be calculated."
