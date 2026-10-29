"""Explicit MongoDB CRUD smoke check; run with ``python scripts/mongo_smoke.py``."""
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


def main() -> int:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("FAIL: MongoDB is not configured.")
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=2500)
    record_id = None
    collection = None
    try:
        client.admin.command("ping")
        collection = client[os.getenv("MONGODB_DB", "retinopathy_demo")]["screenings_smoke"]
        demo_id = f"demo-smoke-{uuid.uuid4().hex[:12]}"
        result = collection.insert_one({
            "demo_id": demo_id,
            "grade": 2,
            "created_at": datetime.now(timezone.utc),
        })
        record_id = result.inserted_id
        created = collection.find_one({"_id": record_id}) is not None
        updated = collection.update_one({"_id": record_id}, {"$set": {"grade": 3}}).modified_count == 1
        read_updated = collection.find_one({"_id": record_id, "grade": 3}) is not None
        deleted = collection.delete_one({"_id": record_id}).deleted_count == 1
        print(f"Create: {'PASS' if created else 'FAIL'}")
        print(f"Read: {'PASS' if read_updated else 'FAIL'}")
        print(f"Update: {'PASS' if updated else 'FAIL'}")
        print(f"Delete: {'PASS' if deleted else 'FAIL'}")
        return 0 if all((created, updated, read_updated, deleted)) else 1
    except PyMongoError as exc:
        print(f"FAIL: MongoDB smoke check could not complete ({type(exc).__name__}).")
        return 1
    finally:
        if record_id is not None and collection is not None:
            try:
                collection.delete_one({"_id": record_id})
            except PyMongoError:
                pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
