"""
core/pipeline.py

Full-frame, identity-stable face tracking pipeline.

Architecture
============

                    ┌─────────────────────────────────────────┐
                    │            Per-frame loop                │
                    │                                         │
  Frame ──► Detector (every N frames) ──► raw BBoxes         │
                    │                          │               │
                    │                    Tracker.update()     │
                    │                          │               │
                    │                   active TrackStates    │
                    │                          │               │
                    │              ┌───────────┴────────────┐  │
                    │              │   For each track:      │  │
                    │              │                        │  │
                    │              │  track_id already      │  │
                    │              │  in cache?             │  │
                    │              │   YES → skip           │  │
                    │              │   NO  → embed + match  │  │
                    │              │         assign face_id │  │
                    │              └───────────┬────────────┘  │
                    │                          │               │
                    │              DB / Logger / Counts        │
                    └─────────────────────────────────────────┘

track_id vs face_id
-------------------
  track_id  — ephemeral integer from the tracker; unique per continuous
              motion trajectory.  Resets when a person leaves and comes back.

  face_id   — permanent UUID derived from ArcFace embedding; stable across
              re-entries, restarts, and tracker resets.  Stored in MongoDB.

Recognition trigger rules
--------------------------
  1. New track_id  →  ALWAYS run recognition once.
  2. Same track_id →  NEVER run recognition again (cache hit).
  3. Re-entry      →  new track_id is assigned by tracker → rule 1 applies →
                      embedding matches existing face_id in DB → no duplicate.

This means recognition runs at most once per track lifetime, regardless of
how long the person stays in frame.  Long-presence users are counted on
their FIRST appearance, not on exit.

Duplicate prevention
--------------------
  - Global in-memory face registry (loaded from MongoDB at startup).
  - Cosine similarity ≥ threshold  →  reuse existing face_id.
  - Below threshold               →  register new face_id.
  - Registry updated immediately after registration so subsequent tracks
    within the same session match correctly.

Active vs unique users
----------------------
  - active_face_ids  : set of face_ids whose track is currently alive.
  - unique_face_ids  : set of all face_ids ever seen this session.
  - Both are updated every frame and surfaced via get_stats().
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, Set

import cv2
import numpy as np

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.tracker import build_tracker, TrackState
from database.mongo_manager import MongoManager
from logging_system.event_logger import EventLogger

logger = logging.getLogger(__name__)


class FaceTrackerPipeline:
    """
    Full-frame identity-stable pipeline.

    Recognition is triggered ONCE per track_id, on first appearance.
    Long-presence users are counted immediately, not on exit.
    """

    def __init__(self, config: dict):
        self.config = config
        det_cfg = config["detection"]
        rec_cfg = config["recognition"]
        trk_cfg = config["tracking"]
        db_cfg  = config["database"]
        log_cfg = config["logging"]

        # ── Subsystems ──────────────────────────────────────────────────
        self.detector = FaceDetector(
            model_path=det_cfg["yolo_model"],
            conf_thresh=det_cfg["confidence_threshold"],
            iou_thresh=det_cfg["iou_threshold"],
        )
        self.recognizer = FaceRecognizer(
            model_name=rec_cfg["model_name"],
            embedding_threshold=rec_cfg["embedding_threshold"],
            min_face_size=rec_cfg["min_face_size"],
        )
        self.tracker = build_tracker(
            tracker_type=trk_cfg["tracker_type"],
            max_disappeared=trk_cfg["max_disappeared"],
            max_distance=trk_cfg["max_distance"],
        )
        self.db = MongoManager(
            uri=db_cfg["uri"],
            db_name=db_cfg["name"],
            collections=db_cfg["collections"],
        )
        self.event_logger = EventLogger(
            log_file=log_cfg["log_file"],
            image_base_dir=log_cfg["image_base_dir"],
        )

        # ── Configurable ────────────────────────────────────────────────
        self.frame_skip: int = det_cfg["frame_skip"]
        # Optionally re-run recognition after this many frames of absence
        # (handles the case where someone leaves, changes appearance, returns)
        self.re_recognition_frames: int = rec_cfg.get("re_recognition_frames", 150)

        # ── Identity cache ───────────────────────────────────────────────
        # PRIMARY CACHE: track_id → face_id
        # Prevents recognition from running more than once per track lifetime.
        self._track_to_face: Dict[int, str] = {}

        # GLOBAL REGISTRY: face_id → embedding (numpy array)
        # Loaded from DB at startup; updated on new registration.
        # Used for cosine matching to prevent duplicate face_ids.
        self._face_registry: Dict[str, np.ndarray] = {}

        # ── Active / unique tracking ─────────────────────────────────────
        # active_face_ids: face_ids with a live track right now
        self._active_face_ids: Set[str] = set()
        # unique_face_ids: every face_id seen this session (superset of active)
        self._unique_face_ids: Set[str] = set()

        # track_id → last frame number when recognition was attempted
        # (used for optional re-recognition on re-entry)
        self._track_last_recognition_frame: Dict[int, int] = {}

        self._frame_count: int = 0

        # Load existing faces from DB into registry
        self._load_face_registry()

        logger.info(
            "FaceTrackerPipeline initialised. Known faces in registry: %d",
            len(self._face_registry),
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Registry bootstrap
    # ──────────────────────────────────────────────────────────────────────

    def _load_face_registry(self):
        """
        Load all face embeddings from MongoDB into the in-memory registry.
        Called once at startup.  Registry is kept updated as new faces are
        registered during the session.
        """
        for doc in self.db.get_all_faces():
            fid = doc["face_id"]
            emb = np.array(doc["embedding"], dtype=np.float32)
            self._face_registry[fid] = emb
        logger.info("Face registry loaded: %d identities.", len(self._face_registry))

    def _add_to_registry(self, face_id: str, embedding: np.ndarray):
        """Add a newly registered face to the in-memory registry immediately."""
        self._face_registry[face_id] = embedding

    # ──────────────────────────────────────────────────────────────────────
    #  Main per-frame entry point
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one BGR video frame.
        Returns the annotated frame.
        """
        self._frame_count += 1

        # ── Detection (every frame_skip+1 frames) ─────────────────────
        if self._frame_count % (self.frame_skip + 1) == 0:
            detections = self.detector.detect(frame)
        else:
            detections = []

        # ── Tracking (every frame) ────────────────────────────────────
        active_tracks = self.tracker.update(detections)

        # ── Identity resolution (only for new track_ids) ──────────────
        current_active_face_ids: Set[str] = set()

        for track in active_tracks:
            face_id = self._resolve_identity(frame, track)
            if face_id:
                current_active_face_ids.add(face_id)

        # ── Handle lost tracks ────────────────────────────────────────
        lost_tracks = self.tracker.remove_lost_tracks()
        for track in lost_tracks:
            self._on_track_lost(track)

        # ── Update active set ─────────────────────────────────────────
        self._active_face_ids = current_active_face_ids

        # ── Annotate ──────────────────────────────────────────────────
        return self._draw_annotations(frame.copy(), active_tracks)

    # ──────────────────────────────────────────────────────────────────────
    #  Identity resolution  (THE CORE LOGIC)
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_identity(
        self, frame: np.ndarray, track: TrackState
    ) -> Optional[str]:
        """
        Determine the face_id for a track.

        Decision tree:
          1. track_id in cache?
             → return cached face_id immediately (no recognition).

          2. track_id NOT in cache (new track):
             → crop face → generate embedding
             → match against _face_registry (cosine similarity)
               a. Match found (sim ≥ threshold)
                  → assign existing face_id (prevents duplicate)
               b. No match
                  → register new face_id, add to registry, log to DB

        Returns face_id or None if embedding generation fails.
        """

        # ── Cache hit: skip recognition entirely ──────────────────────
        if track.track_id in self._track_to_face:
            return self._track_to_face[track.track_id]

        # ── New track: run recognition exactly once ───────────────────
        logger.debug(
            "New track_id=%d — running recognition (frame=%d)",
            track.track_id, self._frame_count,
        )

        crop = self.detector.crop_face(frame, track.bbox)
        embedding = self.recognizer.get_embedding(crop)

        if embedding is None:
            # Embedding failed (bad crop, too small, etc.).
            # Do NOT cache — will retry on next frame for this track_id.
            logger.debug(
                "Embedding failed for track_id=%d, will retry.", track.track_id
            )
            return None

        # ── Match against global registry ─────────────────────────────
        face_id = self._match_registry(embedding)

        if face_id:
            # Known identity re-entering (or track reset for same person)
            logger.info(
                "Re-identified: track_id=%d → face_id=%s", track.track_id, face_id
            )
            self.db.update_face_last_seen(face_id)
            self.event_logger.log(
                f"RE-ID | face_id={face_id} | track_id={track.track_id} "
                f"| frame={self._frame_count}"
            )
        else:
            # Brand new identity — register
            face_id = self._register_new_face(frame, track, embedding, crop)

        # ── Cache the result: track_id → face_id ──────────────────────
        self._track_to_face[track.track_id] = face_id
        self._track_last_recognition_frame[track.track_id] = self._frame_count

        # Update track object so callers can read face_id directly
        track.face_id = face_id
        track.embedding = embedding

        # Count as unique
        self._unique_face_ids.add(face_id)

        return face_id

    # ──────────────────────────────────────────────────────────────────────
    #  Registry matching
    # ──────────────────────────────────────────────────────────────────────

    def _match_registry(self, query_emb: np.ndarray) -> Optional[str]:
        """
        Compare query_emb against all embeddings in the global registry.

        Uses the same cosine similarity as the Recognizer but operates
        on the in-memory dict (no DB round-trip per frame).

        Returns the best-matching face_id if similarity ≥ threshold,
        else None.
        """
        if not self._face_registry:
            return None

        threshold = self.config["recognition"]["embedding_threshold"]
        best_id: Optional[str] = None
        best_sim: float = -1.0

        for fid, stored_emb in self._face_registry.items():
            # Both embeddings are L2-normalised → dot product = cosine sim
            sim = float(np.dot(query_emb, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = fid

        if best_sim >= threshold:
            logger.debug(
                "Registry match: face_id=%s sim=%.4f (threshold=%.2f)",
                best_id, best_sim, threshold,
            )
            return best_id

        logger.debug(
            "No registry match (best_sim=%.4f < threshold=%.2f)",
            best_sim, threshold,
        )
        return None

    # ──────────────────────────────────────────────────────────────────────
    #  New face registration
    # ──────────────────────────────────────────────────────────────────────

    def _register_new_face(
        self,
        frame: np.ndarray,
        track: TrackState,
        embedding: np.ndarray,
        crop: np.ndarray,
    ) -> str:
        """
        Register a brand-new identity.
        Saves to MongoDB, updates in-memory registry, logs entry event.
        Returns the new face_id.
        """
        face_id = "face_" + uuid.uuid4().hex[:12]

        # Save thumbnail
        image_path = self.event_logger.save_face_image(crop, face_id, "registration")

        # Persist to MongoDB
        self.db.register_face(
            face_id=face_id,
            embedding=embedding.tolist(),
            image_path=image_path,
            metadata={
                "track_id": track.track_id,
                "first_seen_frame": self._frame_count,
            },
        )

        # Update in-memory registry immediately so the NEXT new track
        # in this same session can match against this person
        self._add_to_registry(face_id, embedding)

        # Log entry event
        entry_img = self.event_logger.save_face_image(crop, face_id, "entry")
        self.db.log_event(
            face_id, "entry", entry_img,
            timestamp=datetime.utcnow(),
            extra={"track_id": track.track_id, "frame": self._frame_count},
        )
        self.event_logger.log(
            f"ENTRY | face_id={face_id} | track_id={track.track_id} "
            f"| frame={self._frame_count}"
        )

        logger.info(
            "NEW face registered: face_id=%s track_id=%d frame=%d",
            face_id, track.track_id, self._frame_count,
        )
        return face_id

    # ──────────────────────────────────────────────────────────────────────
    #  Track lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def _on_track_lost(self, track: TrackState):
        """
        Called when a track permanently disappears.
        - Logs exit event.
        - Removes track_id from cache (frees memory).
        - face_id remains in _unique_face_ids and _face_registry forever.
        """
        face_id = self._track_to_face.pop(track.track_id, None)
        self._track_last_recognition_frame.pop(track.track_id, None)

        if face_id:
            self._active_face_ids.discard(face_id)
            ts = datetime.utcnow()
            # Use last known crop from track bbox
            # (frame is not available here; log event without image if needed)
            self.db.log_event(
                face_id, "exit", "",
                timestamp=ts,
                extra={"track_id": track.track_id, "frame": self._frame_count},
            )
            self.event_logger.log(
                f"EXIT | face_id={face_id} | track_id={track.track_id} "
                f"| frame={self._frame_count}"
            )
            logger.info("EXIT: face_id=%s track_id=%d", face_id, track.track_id)

    # ──────────────────────────────────────────────────────────────────────
    #  Frame annotation
    # ──────────────────────────────────────────────────────────────────────

    def _draw_annotations(
        self, frame: np.ndarray, tracks: list
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and stats overlay."""
        for track in tracks:
            x1, y1, x2, y2, _ = track.bbox
            face_id = self._track_to_face.get(track.track_id)

            if face_id:
                label = f"{face_id[-8:]}  T:{track.track_id}"
                color = (0, 220, 80)
            else:
                label = f"T:{track.track_id} [pending]"
                color = (180, 180, 50)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        # Stats overlay
        unique  = len(self._unique_face_ids)
        active  = len(self._active_face_ids)
        cv2.putText(
            frame,
            f"Unique: {unique}  Active: {active}  Frame: {self._frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2,
        )
        return frame

    # ──────────────────────────────────────────────────────────────────────
    #  Public helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        db_stats = self.db.get_stats()
        return {
            **db_stats,
            "session_unique_faces": len(self._unique_face_ids),
            "session_active_faces": len(self._active_face_ids),
            "current_frame": self._frame_count,
            "registry_size": len(self._face_registry),
        }

    def shutdown(self):
        self.db.close()
        self.event_logger.close()
        logger.info("Pipeline shut down. Session unique faces: %d", len(self._unique_face_ids))