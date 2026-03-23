"""
analytics/loitering_detector.py

Loitering and suspicious-behaviour detector.

Rules:
  1. LOITERING   — current dwell_time > dwell_threshold (live check)
  2. SUSPICIOUS  — visit_count within a rolling time window exceeds
                   suspicious_visit_threshold

Alerts are:
  - Logged to events.log with prefix [ALERT]
  - Written to MongoDB alerts collection (async)
  - Deduplicated per face_id so the same alert is not spammed every frame
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LoiteringDetector:
    """
    Stateful per-session alert engine.

    Usage (called once per frame from pipeline):
        detector.check(
            face_id=face_id,
            current_dwell=dwell_tracker.get_current_dwell(face_id),
            visit_count=visitor_intelligence.visit_count(face_id),
            db_submit=self._db_submit,
            log_alert_fn=self.db.log_alert,
            event_logger=self.event_logger,
        )
    """

    # How often (seconds) the same alert type can fire per face_id
    ALERT_COOLDOWN = 30.0

    def __init__(
        self,
        dwell_threshold: float = 20.0,
        suspicious_visit_threshold: int = 5,
        suspicious_window_seconds: float = 3600.0,
    ):
        self.dwell_threshold      = dwell_threshold
        self.suspicious_threshold = suspicious_visit_threshold
        self.suspicious_window    = suspicious_window_seconds

        # face_id → {alert_type → last_alert_time (monotonic)}
        self._last_alert: Dict[str, Dict[str, float]] = defaultdict(dict)

    # ------------------------------------------------------------------ #
    #  Main check (call every frame for each active face)                  #
    # ------------------------------------------------------------------ #

    def check(
        self,
        face_id: str,
        current_dwell: Optional[float],
        visit_count: int,
        db_submit: Callable,
        log_alert_fn: Callable,
        event_logger,
    ) -> None:
        """
        Evaluate alert conditions and fire if triggered.
        All DB writes are async via db_submit.
        """
        if current_dwell is not None and current_dwell > self.dwell_threshold:
            self._fire(
                face_id=face_id,
                alert_type="loitering",
                reason=f"dwell_time={current_dwell:.1f}s > threshold={self.dwell_threshold}s",
                db_submit=db_submit,
                log_alert_fn=log_alert_fn,
                event_logger=event_logger,
            )

        if visit_count >= self.suspicious_threshold:
            self._fire(
                face_id=face_id,
                alert_type="suspicious_frequent_visits",
                reason=(
                    f"visit_count={visit_count} >= "
                    f"threshold={self.suspicious_threshold}"
                ),
                db_submit=db_submit,
                log_alert_fn=log_alert_fn,
                event_logger=event_logger,
            )

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _fire(
        self,
        face_id: str,
        alert_type: str,
        reason: str,
        db_submit: Callable,
        log_alert_fn: Callable,
        event_logger,
    ) -> None:
        now = time.monotonic()
        last = self._last_alert[face_id].get(alert_type, 0.0)
        if now - last < self.ALERT_COOLDOWN:
            return    # deduplicate

        self._last_alert[face_id][alert_type] = now
        ts = datetime.utcnow()

        msg = f"[ALERT] Face ID {face_id} flagged | type={alert_type} | {reason}"
        logger.warning(msg)
        event_logger.log(msg, level="warning")

        db_submit(
            log_alert_fn,
            face_id=face_id,
            alert_type=alert_type,
            reason=reason,
            timestamp=ts,
        )

    def clear(self, face_id: str) -> None:
        """Reset alert state for a face that has exited."""
        self._last_alert.pop(face_id, None)
