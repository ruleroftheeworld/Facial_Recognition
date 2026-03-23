"""
core/tracker.py

Centroid-based multi-object tracker with ByteTrack-compatible interface.

Key design:
  - Assigns stable integer track_ids across frames via centroid distance matching.
  - track_id is TEMPORARY — it identifies a continuous motion trajectory.
  - face_id (assigned by pipeline via embedding) is the PERMANENT identity.
  - Exposes track state so pipeline can detect NEW vs EXISTING track_ids.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]  # x1, y1, x2, y2, conf


class TrackState:
    """Represents one tracked object across frames."""

    __slots__ = (
        "track_id", "bbox", "centroid", "disappeared",
        "face_id", "embedding", "frame_count", "is_new"
    )

    def __init__(self, track_id: int, bbox: BBox):
        self.track_id: int = track_id
        self.bbox: BBox = bbox
        self.centroid: np.ndarray = self._bbox_to_centroid(bbox)
        self.disappeared: int = 0
        self.face_id: Optional[str] = None       # set by pipeline after recognition
        self.embedding: Optional[np.ndarray] = None
        self.frame_count: int = 1                # how many frames this track has been alive
        self.is_new: bool = True                 # True only on the very first frame

    @staticmethod
    def _bbox_to_centroid(bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2, _ = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def update(self, bbox: BBox):
        self.bbox = bbox
        self.centroid = self._bbox_to_centroid(bbox)
        self.disappeared = 0
        self.frame_count += 1
        self.is_new = False


class CentroidTracker:
    """
    Greedy nearest-neighbour tracker using Euclidean centroid distance.

    Guarantees:
      - Each track_id is unique and monotonically increasing.
      - track.is_new is True only on the frame a track_id first appears.
      - Lost tracks (disappeared > max_disappeared) are returned via
        remove_lost_tracks() and cleaned up.
    """

    def __init__(self, max_disappeared: int = 40, max_distance: float = 80.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id: int = 0
        self.tracks: Dict[int, TrackState] = OrderedDict()
        # Track IDs that were new THIS frame (reset each update call)
        self.new_track_ids: Set[int] = set()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, detections: List[BBox]) -> List[TrackState]:
        """
        Update tracker state with detections from current frame.
        Returns list of all currently ACTIVE (non-lost) TrackState objects.
        """
        self.new_track_ids.clear()

        # Mark all existing tracks as disappeared; confirmed ones reset in match
        for t in self.tracks.values():
            t.disappeared += 1
            t.is_new = False

        if not detections:
            return self._active_tracks()

        if not self.tracks:
            for det in detections:
                self._register(det)
            return self._active_tracks()

        self._match_detections(detections)
        return self._active_tracks()

    def remove_lost_tracks(self) -> List[TrackState]:
        """
        Deregister tracks exceeding max_disappeared.
        Returns the list of just-removed tracks (pipeline uses these for exit events).
        """
        lost = [
            t for t in self.tracks.values()
            if t.disappeared > self.max_disappeared
        ]
        for t in lost:
            del self.tracks[t.track_id]
        return lost

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _register(self, bbox: BBox) -> TrackState:
        t = TrackState(self._next_id, bbox)
        self.tracks[self._next_id] = t
        self.new_track_ids.add(self._next_id)
        logger.debug("New track registered: id=%d", self._next_id)
        self._next_id += 1
        return t

    def _active_tracks(self) -> List[TrackState]:
        return [
            t for t in self.tracks.values()
            if t.disappeared <= self.max_disappeared
        ]

    def _match_detections(self, detections: List[BBox]):
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid].centroid for tid in track_ids]
        )
        det_centroids = np.array(
            [TrackState._bbox_to_centroid(d) for d in detections]
        )

        # Cost matrix: rows = existing tracks, cols = new detections
        cost = cdist(track_centroids, det_centroids)

        matched_rows: Set[int] = set()
        matched_cols: Set[int] = set()

        # Greedy assignment: sort all (row, col) pairs by cost ascending
        for row, col in sorted(
            np.ndindex(cost.shape), key=lambda rc: cost[rc]
        ):
            if row in matched_rows or col in matched_cols:
                continue
            if cost[row, col] > self.max_distance:
                break   # remaining pairs are all worse
            tid = track_ids[row]
            self.tracks[tid].update(detections[col])
            matched_rows.add(row)
            matched_cols.add(col)

        # Unmatched detections → new tracks
        for col_idx, det in enumerate(detections):
            if col_idx not in matched_cols:
                self._register(det)
        # Unmatched existing tracks → disappeared count already incremented above


class ByteTrackWrapper:
    """
    Public tracker interface used by the pipeline.
    Wraps CentroidTracker; swap internals for DeepSORT/ByteTrack if desired.
    """

    def __init__(self, max_disappeared: int = 40, max_distance: float = 80.0):
        self._tracker = CentroidTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
        )
        logger.info(
            "ByteTrackWrapper (CentroidTracker) initialised. "
            "max_disappeared=%d max_distance=%.0f",
            max_disappeared, max_distance,
        )

    def update(self, detections: List[BBox]) -> List[TrackState]:
        return self._tracker.update(detections)

    def remove_lost_tracks(self) -> List[TrackState]:
        return self._tracker.remove_lost_tracks()

    @property
    def tracks(self) -> Dict[int, TrackState]:
        return self._tracker.tracks

    @property
    def new_track_ids(self) -> Set[int]:
        return self._tracker.new_track_ids


def build_tracker(
    tracker_type: str,
    max_disappeared: int,
    max_distance: float,
) -> ByteTrackWrapper:
    logger.info("Building tracker: type=%s", tracker_type)
    return ByteTrackWrapper(
        max_disappeared=max_disappeared,
        max_distance=max_distance,
    )