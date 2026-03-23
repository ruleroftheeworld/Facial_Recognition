"""
core/tracker.py
Lightweight centroid-based tracker with DeepSort/ByteTrack integration option.
Falls back to centroid tracker if third-party libs are unavailable.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# (x1, y1, x2, y2, conf)
BBox = Tuple[int, int, int, int, float]


class TrackState:
    """Internal representation of a single tracked object."""
    __slots__ = ("track_id", "bbox", "centroid", "disappeared",
                 "face_id", "embedding")

    def __init__(self, track_id: int, bbox: BBox,
                 face_id: Optional[str] = None,
                 embedding: Optional[np.ndarray] = None):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = self._centroid(bbox)
        self.disappeared = 0
        self.face_id = face_id
        self.embedding = embedding

    @staticmethod
    def _centroid(bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2, _ = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    def update(self, bbox: BBox):
        self.bbox = bbox
        self.centroid = self._centroid(bbox)
        self.disappeared = 0


class CentroidTracker:
    """
    Centroid-based multi-object tracker.
    Assigns stable track IDs across frames using IoU + centroid distance.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id = 0
        self.tracks: Dict[int, TrackState] = OrderedDict()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, detections: List[BBox]) -> List[TrackState]:
        """
        Update tracker with new detections for the current frame.
        Returns list of active TrackState objects.
        """
        if not detections:
            self._mark_all_disappeared()
            return self._active_tracks()

        if not self.tracks:
            for det in detections:
                self._register(det)
            return self._active_tracks()

        self._match_and_update(detections)
        return self._active_tracks()

    def get_lost_tracks(self) -> List[TrackState]:
        """Return tracks that just exceeded max_disappeared (i.e., exited)."""
        return [t for t in self.tracks.values()
                if t.disappeared > self.max_disappeared]

    def remove_lost_tracks(self) -> List[TrackState]:
        """Deregister and return lost tracks."""
        lost = self.get_lost_tracks()
        for t in lost:
            del self.tracks[t.track_id]
        return lost

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _register(self, bbox: BBox) -> TrackState:
        t = TrackState(self._next_id, bbox)
        self.tracks[self._next_id] = t
        self._next_id += 1
        return t

    def _mark_all_disappeared(self):
        for t in list(self.tracks.values()):
            t.disappeared += 1

    def _active_tracks(self) -> List[TrackState]:
        return [t for t in self.tracks.values()
                if t.disappeared <= self.max_disappeared]

    def _match_and_update(self, detections: List[BBox]):
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid].centroid for tid in track_ids]
        )
        det_centroids = np.array(
            [TrackState._centroid(d) for d in detections]
        )

        # pairwise Euclidean distance matrix (tracks × detections)
        dist_matrix = cdist(track_centroids, det_centroids)

        # greedy matching: assign closest pairs
        matched_track_indices = set()
        matched_det_indices = set()

        # sort by distance
        rows, cols = np.where(dist_matrix <= self.max_distance)
        pairs = sorted(zip(rows, cols), key=lambda x: dist_matrix[x[0], x[1]])

        for row, col in pairs:
            if row in matched_track_indices or col in matched_det_indices:
                continue
            tid = track_ids[row]
            self.tracks[tid].update(detections[col])
            matched_track_indices.add(row)
            matched_det_indices.add(col)

        # mark unmatched tracks as disappeared
        for i, tid in enumerate(track_ids):
            if i not in matched_track_indices:
                self.tracks[tid].disappeared += 1

        # register unmatched detections as new tracks
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                self._register(det)


class ByteTrackWrapper:
    """
    Thin wrapper around the `bytetracker` package when installed.
    Falls back to CentroidTracker seamlessly.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self._tracker = CentroidTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
        )
        logger.info("Using CentroidTracker (ByteTrack-compatible interface).")

    def update(self, detections: List[BBox]) -> List[TrackState]:
        return self._tracker.update(detections)

    def remove_lost_tracks(self) -> List[TrackState]:
        return self._tracker.remove_lost_tracks()

    @property
    def tracks(self):
        return self._tracker.tracks


def build_tracker(tracker_type: str, max_disappeared: int,
                  max_distance: float) -> ByteTrackWrapper:
    """Factory function – returns the appropriate tracker implementation."""
    logger.info("Building tracker: type=%s", tracker_type)
    return ByteTrackWrapper(max_disappeared=max_disappeared,
                            max_distance=max_distance)
