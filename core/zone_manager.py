"""
core/zone_manager.py

Implements zone-based entry/exit detection instead of continuous per-frame detection.

Strategy:
  1. Frame is divided into: [entry zone | interior | exit zone]
  2. A face is only detected + embedded when its track centroid crosses INTO
     the entry or exit zone for the first time.
  3. Trajectory (direction of movement) validates the event to prevent
     false triggers from people walking parallel to the boundary.
  4. A per-face cooldown dict prevents the same face_id from firing another
     entry event until re_entry_cooldown_seconds have elapsed.
  5. Tracker runs every frame (cheap). Recognition only runs on zone crossings.

This reduces recognition calls from ~N_detections × FPS to ~2 per visit.
"""

import logging
import time
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Zone(Enum):
    ENTRY = auto()
    INTERIOR = auto()
    EXIT = auto()
    OUTSIDE = auto()    # beyond exit edge (already left)


class CrossingEvent(Enum):
    ENTERED_ENTRY_ZONE = auto()   # crossed into entry zone → fire entry detection
    ENTERED_EXIT_ZONE = auto()    # crossed into exit zone → fire exit detection
    NONE = auto()


class ZoneManager:
    """
    Manages spatial zones and crossing logic for a single camera view.

    Zone layout (left-to-right direction):
      x=0 ──[entry_zone]──[──── interior ────]──[exit_zone]── x=W

    For right-to-left simply set direction='right-to-left';
    entry and exit sides flip automatically.
    """

    def __init__(self, frame_width: int, frame_height: int, zone_cfg: dict):
        self.W = frame_width
        self.H = frame_height
        self.direction = zone_cfg.get("direction", "left-to-right")
        self.cooldown_secs = zone_cfg.get("re_entry_cooldown_seconds", 60)

        entry_ratio = zone_cfg.get("entry_zone_ratio", 0.20)
        exit_ratio  = zone_cfg.get("exit_zone_ratio",  0.20)

        # Pixel boundaries
        if self.direction == "left-to-right":
            self.entry_x_max = int(self.W * entry_ratio)
            self.exit_x_min  = int(self.W * (1.0 - exit_ratio))
        else:
            # flip: entry is on the right
            self.entry_x_max = self.W - int(self.W * exit_ratio)   # right edge
            self.exit_x_min  = int(self.W * entry_ratio)            # left edge

        # Track state: track_id → last Zone
        self._track_zones: Dict[int, Zone] = {}

        # Track state: track_id → recent centroid history (for trajectory check)
        # Stores last N centroids as [(x, y), ...]
        self._centroid_history: Dict[int, list] = {}
        self._history_len = 8

        # Cooldown: face_id → last entry timestamp
        self._entry_cooldown: Dict[str, float] = {}

        logger.info(
            "ZoneManager ready | W=%d direction=%s entry_x<=%d exit_x>=%d cooldown=%ds",
            self.W, self.direction, self.entry_x_max, self.exit_x_min, self.cooldown_secs,
        )

    # ------------------------------------------------------------------ #
    #  Zone classification                                                 #
    # ------------------------------------------------------------------ #

    def classify(self, cx: float) -> Zone:
        """Return which zone the x-centroid falls in."""
        if self.direction == "left-to-right":
            if cx <= self.entry_x_max:
                return Zone.ENTRY
            if cx >= self.exit_x_min:
                return Zone.EXIT
            return Zone.INTERIOR
        else:
            if cx >= self.entry_x_max:
                return Zone.ENTRY
            if cx <= self.exit_x_min:
                return Zone.EXIT
            return Zone.INTERIOR

    # ------------------------------------------------------------------ #
    #  Per-frame update                                                    #
    # ------------------------------------------------------------------ #

    def update_track(
        self, track_id: int, cx: float, cy: float
    ) -> CrossingEvent:
        """
        Call once per active track per frame.

        Returns CrossingEvent if the track just crossed into a notable zone,
        else CrossingEvent.NONE.
        """
        # Update centroid history
        history = self._centroid_history.setdefault(track_id, [])
        history.append((cx, cy))
        if len(history) > self._history_len:
            history.pop(0)

        current_zone = self.classify(cx)
        prev_zone = self._track_zones.get(track_id)

        # First observation — just record, no event
        if prev_zone is None:
            self._track_zones[track_id] = current_zone
            return CrossingEvent.NONE

        # No zone change
        if current_zone == prev_zone:
            return CrossingEvent.NONE

        self._track_zones[track_id] = current_zone
        event = CrossingEvent.NONE

        # Crossed INTO entry zone from exterior or from interior
        if current_zone == Zone.ENTRY:
            if self._valid_direction(history, entering=True):
                event = CrossingEvent.ENTERED_ENTRY_ZONE
                logger.debug("Track %d crossed into ENTRY zone", track_id)

        # Crossed INTO exit zone
        elif current_zone == Zone.EXIT:
            if self._valid_direction(history, entering=False):
                event = CrossingEvent.ENTERED_EXIT_ZONE
                logger.debug("Track %d crossed into EXIT zone", track_id)

        return event

    def remove_track(self, track_id: int):
        """Clean up state when a track is lost."""
        self._track_zones.pop(track_id, None)
        self._centroid_history.pop(track_id, None)

    # ------------------------------------------------------------------ #
    #  Cooldown                                                            #
    # ------------------------------------------------------------------ #

    def is_in_cooldown(self, face_id: str) -> bool:
        last = self._entry_cooldown.get(face_id)
        if last is None:
            return False
        return (time.time() - last) < self.cooldown_secs

    def set_entry_cooldown(self, face_id: str):
        self._entry_cooldown[face_id] = time.time()
        logger.debug("Cooldown set for face_id=%s (%ds)", face_id, self.cooldown_secs)

    def clear_cooldown(self, face_id: str):
        self._entry_cooldown.pop(face_id, None)

    # ------------------------------------------------------------------ #
    #  Trajectory validation                                               #
    # ------------------------------------------------------------------ #

    def _valid_direction(self, history: list, entering: bool) -> bool:
        """
        Check that the track is actually moving in the expected direction
        and not just jittering near a boundary.

        entering=True  → expect movement toward entry side
        entering=False → expect movement toward exit side
        """
        if len(history) < 3:
            return True   # not enough data, allow through

        xs = [p[0] for p in history]
        # Simple linear regression slope on x positions
        n = len(xs)
        indices = list(range(n))
        mean_i = sum(indices) / n
        mean_x = sum(xs) / n
        num = sum((i - mean_i) * (x - mean_x) for i, x in zip(indices, xs))
        den = sum((i - mean_i) ** 2 for i in indices)
        slope = num / den if den != 0 else 0.0

        # Minimum required horizontal displacement (pixels) to confirm movement
        net_displacement = abs(xs[-1] - xs[0])
        if net_displacement < 10:
            logger.debug("Trajectory rejected: displacement=%.1f < 10px", net_displacement)
            return False

        if self.direction == "left-to-right":
            # Entering → moving right (positive slope); exiting → also moving right
            # Both are positive slope; entering zone is on left, exit is on right
            if entering:
                # came from outside left, moving right → positive slope
                return slope > 0
            else:
                # moved from interior to right side → positive slope
                return slope > 0
        else:
            if entering:
                return slope < 0  # moving left (right-to-left direction)
            else:
                return slope < 0

    # ------------------------------------------------------------------ #
    #  Visualisation helpers                                               #
    # ------------------------------------------------------------------ #

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Overlay semi-transparent zone boxes on the frame."""
        import cv2
        overlay = frame.copy()

        if self.direction == "left-to-right":
            entry_rect  = (0, 0, self.entry_x_max, self.H)
            exit_rect   = (self.exit_x_min, 0, self.W, self.H)
        else:
            entry_rect  = (self.entry_x_max, 0, self.W, self.H)
            exit_rect   = (0, 0, self.exit_x_min, self.H)

        # Entry zone — green tint
        cv2.rectangle(overlay, (entry_rect[0], entry_rect[1]),
                      (entry_rect[2], entry_rect[3]), (0, 200, 80), -1)
        # Exit zone — red tint
        cv2.rectangle(overlay, (exit_rect[0], exit_rect[1]),
                      (exit_rect[2], exit_rect[3]), (0, 60, 220), -1)

        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Zone labels
        cv2.putText(frame, "ENTRY", (entry_rect[0] + 6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 1)
        cv2.putText(frame, "EXIT", (exit_rect[0] + 6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 220), 1)

        # Boundary lines
        cv2.line(frame, (self.entry_x_max, 0), (self.entry_x_max, self.H),
                 (0, 200, 80), 1)
        cv2.line(frame, (self.exit_x_min, 0), (self.exit_x_min, self.H),
                 (0, 80, 220), 1)
        return frame