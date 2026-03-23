"""
database/mongo_manager.py
MongoDB interface for storing face metadata, events, and visitor counts.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

logger = logging.getLogger(__name__)


class MongoManager:
    """Handles all MongoDB operations for the face tracker system."""

    def __init__(self, uri: str, db_name: str, collections: Dict[str, str]):
        self.uri = uri
        self.db_name = db_name
        self.collection_names = collections
        self.client: Optional[MongoClient] = None
        self.db = None
        self._connect()
        self._setup_indexes()

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def _connect(self):
        """Establish MongoDB connection with retry logic."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")          # verify connection
            self.db = self.client[self.db_name]
            logger.info("Connected to MongoDB: %s / %s", self.uri, self.db_name)
        except ConnectionFailure as exc:
            logger.error("MongoDB connection failed: %s", exc)
            raise

    def _setup_indexes(self):
        """Create indexes for performance-critical queries."""
        faces_col = self.db[self.collection_names["faces"]]
        faces_col.create_index([("face_id", ASCENDING)], unique=True)
        faces_col.create_index([("registered_at", DESCENDING)])

        events_col = self.db[self.collection_names["events"]]
        events_col.create_index([("face_id", ASCENDING)])
        events_col.create_index([("timestamp", DESCENDING)])
        events_col.create_index([("event_type", ASCENDING)])

        visitors_col = self.db[self.collection_names["visitors"]]
        visitors_col.create_index([("date", ASCENDING)], unique=True)

        logger.info("MongoDB indexes ensured.")

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    # ------------------------------------------------------------------ #
    #  Face Registration                                                   #
    # ------------------------------------------------------------------ #

    def register_face(self, face_id: str, embedding: List[float],
                      image_path: str, metadata: Dict[str, Any]) -> bool:
        """Insert a new face into the faces collection."""
        doc = {
            "face_id": face_id,
            "embedding": embedding,
            "thumbnail_path": image_path,
            "registered_at": datetime.utcnow(),
            "visit_count": 1,
            "last_seen": datetime.utcnow(),
            **metadata,
        }
        try:
            self.db[self.collection_names["faces"]].insert_one(doc)
            logger.info("Registered new face: %s", face_id)
            self._increment_daily_visitor()
            return True
        except DuplicateKeyError:
            logger.warning("Face %s already registered.", face_id)
            return False

    def update_face_last_seen(self, face_id: str):
        """Update the last_seen timestamp and increment visit_count."""
        self.db[self.collection_names["faces"]].update_one(
            {"face_id": face_id},
            {
                "$set": {"last_seen": datetime.utcnow()},
                "$inc": {"visit_count": 1},
            },
        )

    def get_all_faces(self) -> List[Dict]:
        """Return all registered face documents (embedding included)."""
        return list(
            self.db[self.collection_names["faces"]].find(
                {}, {"_id": 0}
            )
        )

    def face_exists(self, face_id: str) -> bool:
        return (
            self.db[self.collection_names["faces"]].find_one(
                {"face_id": face_id}, {"_id": 1}
            )
            is not None
        )

    # ------------------------------------------------------------------ #
    #  Event Logging                                                       #
    # ------------------------------------------------------------------ #

    def log_event(
        self,
        face_id: str,
        event_type: str,          # "entry" | "exit"
        image_path: str,
        timestamp: Optional[datetime] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """Insert an entry/exit event and return the inserted document id."""
        doc = {
            "face_id": face_id,
            "event_type": event_type,
            "image_path": image_path,
            "timestamp": timestamp or datetime.utcnow(),
            **(extra or {}),
        }
        result = self.db[self.collection_names["events"]].insert_one(doc)
        logger.debug("Event logged – face=%s type=%s", face_id, event_type)
        return str(result.inserted_id)

    def get_events_for_face(self, face_id: str) -> List[Dict]:
        return list(
            self.db[self.collection_names["events"]].find(
                {"face_id": face_id}, {"_id": 0}
            ).sort("timestamp", ASCENDING)
        )

    # ------------------------------------------------------------------ #
    #  Visitor Counter                                                     #
    # ------------------------------------------------------------------ #

    def _increment_daily_visitor(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self.db[self.collection_names["visitors"]].update_one(
            {"date": today},
            {"$inc": {"unique_visitors": 1}},
            upsert=True,
        )

    def get_unique_visitor_count(self, date: Optional[str] = None) -> int:
        """Return unique visitor count for a given date (default: today)."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        doc = self.db[self.collection_names["visitors"]].find_one({"date": date})
        return doc["unique_visitors"] if doc else 0

    def get_total_registered_faces(self) -> int:
        return self.db[self.collection_names["faces"]].count_documents({})

    # ------------------------------------------------------------------ #
    #  Dwell Time                                                          #
    # ------------------------------------------------------------------ #

    def update_exit_event_dwell(
        self,
        face_id: str,
        exit_time: datetime,
        dwell_time: float,
        category: str,
    ):
        """Enrich the most-recent exit event for face_id with dwell analytics."""
        self.db[self.collection_names["events"]].find_one_and_update(
            {"face_id": face_id, "event_type": "exit"},
            {
                "$set": {
                    "exit_time":  exit_time,
                    "dwell_time": dwell_time,
                    "category":   category,
                }
            },
            sort=[("timestamp", -1)],
        )
        logger.debug(
            "Exit event dwell updated: face=%s dwell=%.1fs cat=%s",
            face_id, dwell_time, category,
        )

    def update_visitor_dwell_stats(self, face_id: str, dwell_time: float):
        """Update rolling avg_dwell_time and total_dwell_time on faces doc."""
        faces_col = self.db[self.collection_names["faces"]]
        doc = faces_col.find_one(
            {"face_id": face_id}, {"visit_count": 1, "total_dwell_time": 1}
        )
        if not doc:
            return
        total = doc.get("total_dwell_time", 0.0) + dwell_time
        count = doc.get("visit_count", 1)
        avg   = total / count if count > 0 else dwell_time
        faces_col.update_one(
            {"face_id": face_id},
            {"$set": {
                "total_dwell_time": round(total, 2),
                "avg_dwell_time":   round(avg, 2),
            }},
        )

    # ------------------------------------------------------------------ #
    #  Visitor Intelligence                                                #
    # ------------------------------------------------------------------ #

    def upsert_visitor_profile(self, face_id: str, profile: Dict[str, Any]):
        """Upsert returning-visitor fields (visit_count, last_seen, tag) into faces."""
        self.db[self.collection_names["faces"]].update_one(
            {"face_id": face_id},
            {
                "$set": {
                    "visit_count": profile["visit_count"],
                    "last_seen":   profile["last_seen"],
                    "tag":         profile["tag"],
                },
                "$setOnInsert": {"first_seen": profile["first_seen"]},
            },
            upsert=True,
        )

    # ------------------------------------------------------------------ #
    #  Alerts                                                              #
    # ------------------------------------------------------------------ #

    def log_alert(
        self,
        face_id: str,
        alert_type: str,
        reason: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Insert an alert document into the alerts collection."""
        doc = {
            "face_id":    face_id,
            "alert_type": alert_type,
            "reason":     reason,
            "timestamp":  timestamp or datetime.utcnow(),
        }
        result = self.db["alerts"].insert_one(doc)
        logger.warning(
            "ALERT logged: face=%s type=%s reason=%s", face_id, alert_type, reason
        )
        return str(result.inserted_id)

    # ------------------------------------------------------------------ #
    #  Stats / Reports                                                     #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        pipeline = [
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}}
        ]
        event_counts = {
            doc["_id"]: doc["count"]
            for doc in self.db[self.collection_names["events"]].aggregate(pipeline)
        }
        return {
            "total_registered_faces": self.get_total_registered_faces(),
            "today_unique_visitors": self.get_unique_visitor_count(today),
            "total_entries": event_counts.get("entry", 0),
            "total_exits": event_counts.get("exit", 0),
        }
