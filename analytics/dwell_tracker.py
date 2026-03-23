"""
analytics/dwell_tracker.py

Tracks entry_time and exit_time per face_id within a session.
Computes dwell_time and categorises engagement level.

Categories:
  passerby        < 5 sec
  engaged         5 – 20 sec
  highly_engaged  > 20 sec

Thread-safe: designed to be called from the single pipeline thread only.
All DB writes go through the caller-supplied db_submit callable so async
writes are preserved end-to-end.
"""

import logging
import time
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Engagement thresholds (seconds) ─────────────────────────────────────────
PASSERBY_MAX = 5.0
ENGAGED_MAX  = 20.0


def categorise_dwell(dwell_secs: float) -> str:
    """Return engagement category string for a dwell duration."""
    if dwell_secs < PASSERBY_MAX:
        return "passerby"
    if dwell_secs <= ENGAGED_MAX:
        return "engaged"
    return "highly_engaged"


class DwellTracker:
    """
    Manages per-face session timing.

    Lifecycle per face_id:
        on_entry(face_id)   → records wall-clock entry_time
        on_exit(face_id)    → computes dwell_time, writes DB, returns summary
    """

    def __init__(self, dwell_threshold: float = 20.0):
        """
        Args:
            dwell_threshold: seconds above which a visit is 'highly_engaged'
                             (also used by LoiteringDetector for alert threshold).
        """
        self.dwell_threshold = dwell_threshold
        # face_id → (entry_wall_time, entry_datetime)
        self._active: Dict[str, Tuple[float, datetime]] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def on_entry(self, face_id: str) -> None:
        """Record entry time for a face.  Safe to call multiple times."""
        if face_id not in self._active:
            now_mono   = time.monotonic()
            now_dt     = datetime.utcnow()
            self._active[face_id] = (now_mono, now_dt)
            logger.debug("[DWELL] Entry recorded: face_id=%s at %s", face_id, now_dt.isoformat())

    def on_exit(
        self,
        face_id: str,
        db_submit: Callable,
        update_exit_event_fn: Callable,
        update_visitor_dwell_fn: Callable,
    ) -> Optional[dict]:
        """
        Compute dwell_time for face_id and persist results.

        Args:
            face_id:                  identity being exited
            db_submit:                pipeline's _db_submit callable (async-aware)
            update_exit_event_fn:     MongoManager.update_exit_event_dwell
            update_visitor_dwell_fn:  MongoManager.update_visitor_dwell_stats

        Returns dict with dwell analytics, or None if entry was never recorded.
        """
        entry_info = self._active.pop(face_id, None)
        if entry_info is None:
            logger.debug("[DWELL] No entry found for face_id=%s on exit.", face_id)
            return None

        entry_mono, entry_dt = entry_info
        exit_dt   = datetime.utcnow()
        dwell_sec = time.monotonic() - entry_mono
        category  = categorise_dwell(dwell_sec)

        result = {
            "face_id":    face_id,
            "entry_time": entry_dt,
            "exit_time":  exit_dt,
            "dwell_time": round(dwell_sec, 2),
            "category":   category,
        }

        logger.info(
            "[DWELL] face_id=%s | dwell=%.1fs | category=%s",
            face_id, dwell_sec, category,
        )

        # Async DB writes
        db_submit(
            update_exit_event_fn,
            face_id=face_id,
            exit_time=exit_dt,
            dwell_time=round(dwell_sec, 2),
            category=category,
        )
        db_submit(
            update_visitor_dwell_fn,
            face_id=face_id,
            dwell_time=round(dwell_sec, 2),
        )

        return result

    def get_current_dwell(self, face_id: str) -> Optional[float]:
        """Return live dwell seconds for an active face (None if not tracked)."""
        info = self._active.get(face_id)
        if info is None:
            return None
        return time.monotonic() - info[0]

    def active_face_ids(self):
        return list(self._active.keys())
