from datetime import datetime

from pymongo import MongoClient


try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["trafficvision_ai"]
    violations_collection = db["violations"]
except Exception as e:
    print("❌ MongoDB connection error:", e)
    violations_collection = None


def save_violation(data):
    print("🔥 SAVING TO DB:", data)
    if violations_collection is None:
        print("⚠️ DB not available, skipping save")
        return
    try:
        track_id = data.get("track_id")
        if track_id is not None:
            existing = violations_collection.find_one({"track_id": track_id})
            if existing:
                print(f"⚠️ Duplicate track_id {track_id}, skipping insert")
                return
        data["created_at"] = datetime.utcnow()
        violations_collection.insert_one(data)
    except Exception as e:
        print("❌ Insert error:", e)
