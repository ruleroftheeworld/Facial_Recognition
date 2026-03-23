"""
analytics/visitor_intelligence.py

Returning-visitor intelligence layer.

Tracks per-identity visit history and classifies visitors as:
  new        visit_count == 1
  occasional visit_count 2–4
  frequent   visit_count >= 5

Designed to work with the existing MongoManager `faces` collection by
adding new fields without breaking existing schema.

All mutations go through the pipeline's db_submit so async writes are
preserved end-to-end.
"""

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Visitor tag thresholds ───────────────────────────────────────────────────
TAG_OCCASIONAL_MIN = 2
TAG_FREQUENT_MIN   = 5


def classify_tag(visit_count: int) -> str:
    if visit_count >= TAG_FREQUENT_MIN:
        return "frequent"
    if visit_count >= TAG_OCCASIONAL_MIN:
        return "occasional"
    return "new"


class VisitorIntelligence:
    """
    Manages returning-visitor logic.

    On each new face recognition (entry event):
      - If face_id already in DB  → increment visit_count, update last_seen,
                                     recalculate tag
      - If face_id brand new      → initialize visitor profile

    An in-memory cache (face_id → visit_count) avoids DB reads on every frame.
    The cache is populated at startup from MongoManager.get_all_faces().
    """

    def __init__(self):
        # face_id → {visit_count, first_seen, last_seen, tag}
        self._cache: dict = {}

    # ------------------------------------------------------------------ #
    #  Startup                                                             #
    # ------------------------------------------------------------------ #

    def preload(self, all_faces: list) -> None:
        """
        Populate in-memory cache from faces loaded at startup.
        Call once after MongoManager.get_all_faces().
        """
        for doc in all_faces:
            fid = doc.get("face_id")
            if not fid:
                continue
            self._cache[fid] = {
                "visit_count": doc.get("visit_count", 1),
                "first_seen":  doc.get("first_seen", doc.get("registered_at")),
                "last_seen":   doc.get("last_seen"),
                "tag":         doc.get("tag", "new"),
            }
        logger.info("[VISITOR] Preloaded %d visitor profiles.", len(self._cache))

    # ------------------------------------------------------------------ #
    #  Entry handling                                                      #
    # ------------------------------------------------------------------ #

    def on_entry(
        self,
        face_id: str,
        is_new: bool,
        db_submit: Callable,
        upsert_visitor_fn: Callable,
    ) -> dict:
        """
        Record a face entry.  Updates cache + DB.

        Args:
            face_id:          identity
            is_new:           True if this is first-ever registration
            db_submit:        pipeline's async-aware _db_submit
            upsert_visitor_fn: MongoManager.upsert_visitor_profile

        Returns visitor profile dict.
        """
        now = datetime.utcnow()

        if is_new:
            profile = {
                "visit_count": 1,
                "first_seen":  now,
                "last_seen":   now,
                "tag":         "new",
            }
            self._cache[face_id] = profile
            logger.info(
                "[VISITOR] New visitor face_id=%s | tag=new", face_id
            )
        else:
            existing = self._cache.get(face_id, {})
            visit_count = existing.get("visit_count", 1) + 1
            tag = classify_tag(visit_count)
            profile = {
                "visit_count": visit_count,
                "first_seen":  existing.get("first_seen", now),
                "last_seen":   now,
                "tag":         tag,
            }
            self._cache[face_id] = profile
            logger.info(
                "[VISITOR] Returning visitor face_id=%s | visit_count=%d | tag=%s",
                face_id, visit_count, tag,
            )

        # Async upsert
        db_submit(upsert_visitor_fn, face_id=face_id, profile=profile)
        return profile

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def get_profile(self, face_id: str) -> Optional[dict]:
        return self._cache.get(face_id)

    def visit_count(self, face_id: str) -> int:
        return self._cache.get(face_id, {}).get("visit_count", 0)

    def tag(self, face_id: str) -> str:
        return self._cache.get(face_id, {}).get("tag", "new")
