"""
core/pipeline.py

Zone-aware pipeline: recognition fires only on zone boundary crossings,
not on every frame. This eliminates duplicate entry events and dramatically
reduces compute load.

Flow per frame:
  1. Tracker updates every frame (cheap centroid math).
  2. ZoneManager checks each active track for zone transitions.
  3. Only when a track crosses INTO the entry or exit zone:
       a. Run YOLO detection on that face crop (confirm face present).
       b. Generate ArcFace embedding.
       c. Match against known faces DB.
       d. Register if new; fire entry/exit event with cooldown guard.
  4. Annotate frame and return.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.tracker import build_tracker, TrackState
from core.zone_manager import ZoneManager, CrossingEvent, Zone
from database.mongo_manager import MongoManager
from logging_system.event_logger import EventLogger

logger = logging.getLogger(__name__)


class FaceTrackerPipeline:
    """
    Orchestrates: Tracker → ZoneManager → Detector → Recognizer → DB → Logger
    Recognition is triggered only on zone-boundary crossings.
    """

    def __init__(self, config: dict):
        self.config = config
        det_cfg  = config["detection"]
        rec_cfg  = config["recognition"]
        trk_cfg  = config["tracking"]
        zone_cfg = config.get("zones", {})
        db_cfg   = config["database"]
        log_cfg  = config["logging"]

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

        # ZoneManager initialised on first frame (needs frame dimensions)
        self._zone_cfg = zone_cfg
        self.zone_mgr: Optional[ZoneManager] = None

        # ── Runtime state ────────────────────────────────────────────────
        self.frame_skip: int = det_cfg["frame_skip"]
        self._frame_count: int = 0

        # track_id → face_id  (assigned after recognition)
        self._track_to_face: Dict[int, str] = {}

        # track_id → last known bbox (for exit-event crop when tracker lost it)
        self._track_last_bbox: Dict[int, tuple] = {}

        # face_id → "inside" | "outside"  (prevents duplicate exit events)
        self._face_state: Dict[str, str] = {}

        # Embedding cache loaded from DB at startup
        self._face_cache: list = []
        self._refresh_face_cache()

        logger.info("FaceTrackerPipeline (zone-aware) initialised.")

    # ──────────────────────────────────────────────────────────────────────
    #  Cache helpers
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_face_cache(self):
        self._face_cache = self.db.get_all_faces()
        logger.debug("Face cache: %d known faces.", len(self._face_cache))

    # ──────────────────────────────────────────────────────────────────────
    #  Main per-frame entry point
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        h, w = frame.shape[:2]

        # Lazy-init ZoneManager (needs frame dimensions)
        if self.zone_mgr is None:
            self.zone_mgr = ZoneManager(w, h, self._zone_cfg)

        # ── Step 1: detect every N frames (for tracker bootstrapping)
        run_detection = (self._frame_count % (self.frame_skip + 1) == 0)
        if run_detection:
            detections = self.detector.detect(frame)
        else:
            detections = []

        # ── Step 2: update tracker
        active_tracks = self.tracker.update(detections)

        # ── Step 3: zone crossing check per active track
        for track in active_tracks:
            cx, cy = float(track.centroid[0]), float(track.centroid[1])
            self._track_last_bbox[track.track_id] = track.bbox

            crossing = self.zone_mgr.update_track(track.track_id, cx, cy)

            if crossing == CrossingEvent.ENTERED_ENTRY_ZONE:
                self._handle_entry_crossing(frame, track)

            elif crossing == CrossingEvent.ENTERED_EXIT_ZONE:
                self._handle_exit_crossing(frame, track)

        # ── Step 4: handle permanently lost tracks
        lost_tracks = self.tracker.remove_lost_tracks()
        for track in lost_tracks:
            self._on_track_lost(frame, track)

        # ── Step 5: annotate and return
        annotated = frame.copy()
        annotated = self.zone_mgr.draw_zones(annotated)
        annotated = self._draw_annotations(annotated, active_tracks)
        return annotated

    # ──────────────────────────────────────────────────────────────────────
    #  Zone crossing handlers
    # ──────────────────────────────────────────────────────────────────────

    def _handle_entry_crossing(self, frame: np.ndarray, track: TrackState):
        """
        A track just crossed into the ENTRY zone.
        Run recognition; register if new; fire entry event if not in cooldown.
        """
        face_id = self._resolve_face_id(frame, track)
        if face_id is None:
            return   # no valid face crop / embedding

        # Guard: cooldown prevents same person re-triggering entry immediately
        if self.zone_mgr.is_in_cooldown(face_id):
            logger.debug(
                "Entry suppressed for face=%s (cooldown active)", face_id
            )
            return

        # Guard: already marked inside (e.g. entered via interior on first frame)
        if self._face_state.get(face_id) == "inside":
            return

        self._face_state[face_id] = "inside"
        self.zone_mgr.set_entry_cooldown(face_id)
        self._fire_entry_event(frame, track, face_id)

    def _handle_exit_crossing(self, frame: np.ndarray, track: TrackState):
        """
        A track just crossed into the EXIT zone.
        Confirm face; fire exit event.
        """
        face_id = self._track_to_face.get(track.track_id)
        if face_id is None:
            # Not yet identified — attempt recognition now
            face_id = self._resolve_face_id(frame, track)
        if face_id is None:
            return

        if self._face_state.get(face_id) != "inside":
            return   # never saw this face enter; ignore

        self._face_state[face_id] = "outside"
        self.zone_mgr.clear_cooldown(face_id)   # allow re-entry after full exit
        self._fire_exit_event(frame, track, face_id)

    def _on_track_lost(self, frame: np.ndarray, track: TrackState):
        """
        Called when a track disappears (max_disappeared exceeded).
        If the face was 'inside', fire an exit event to keep counts consistent.
        """
        face_id = self._track_to_face.pop(track.track_id, None)
        self.zone_mgr.remove_track(track.track_id)
        self._track_last_bbox.pop(track.track_id, None)

        if face_id and self._face_state.get(face_id) == "inside":
            logger.info("Track %d lost while inside — firing exit for face=%s",
                        track.track_id, face_id)
            self._face_state[face_id] = "outside"
            # Use last known crop for the image
            last_bbox = self._track_last_bbox.get(track.track_id, track.bbox)
            crop = self.detector.crop_face(frame, last_bbox)
            self._do_fire_exit(face_id, crop)

    # ──────────────────────────────────────────────────────────────────────
    #  Recognition / Registration
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_face_id(
        self, frame: np.ndarray, track: TrackState
    ) -> Optional[str]:
        """
        Get (or create) the face_id for this track.
        Runs YOLO crop → ArcFace embedding → DB match.
        Returns face_id string or None if recognition fails.
        """
        # If already resolved for this track, return immediately (no re-detect)
        if track.track_id in self._track_to_face:
            return self._track_to_face[track.track_id]

        crop = self.detector.crop_face(frame, track.bbox)
        embedding = self.recognizer.get_embedding(crop)
        if embedding is None:
            logger.debug("No embedding for track %d", track.track_id)
            return None

        face_id, similarity = self.recognizer.find_best_match(
            embedding, self._face_cache
        )

        if face_id is None:
            # ── New face: register ─────────────────────────────────────
            face_id = self._generate_face_id()
            image_path = self.event_logger.save_face_image(
                crop, face_id, "registration"
            )
            self.db.register_face(
                face_id=face_id,
                embedding=embedding.tolist(),
                image_path=image_path,
                metadata={"track_id": track.track_id},
            )
            self._face_cache.append(
                {"face_id": face_id, "embedding": embedding.tolist()}
            )
            logger.info("Registered NEW face: %s", face_id)
            self.event_logger.log(
                f"REGISTER | face_id={face_id} | track_id={track.track_id}"
            )
        else:
            self.db.update_face_last_seen(face_id)
            logger.debug("Recognised face %s (sim=%.3f)", face_id, similarity)

        self._track_to_face[track.track_id] = face_id
        return face_id

    # ──────────────────────────────────────────────────────────────────────
    #  Event helpers
    # ──────────────────────────────────────────────────────────────────────

    def _fire_entry_event(
        self, frame: np.ndarray, track: TrackState, face_id: str
    ):
        ts = datetime.utcnow()
        crop = self.detector.crop_face(frame, track.bbox)
        image_path = self.event_logger.save_face_image(crop, face_id, "entry")
        self.db.log_event(face_id, "entry", image_path, timestamp=ts)
        self.event_logger.log(
            f"ENTRY | face_id={face_id} | track_id={track.track_id} | ts={ts.isoformat()}"
        )
        logger.info("ENTRY ← face=%s track=%d", face_id, track.track_id)

    def _fire_exit_event(
        self, frame: np.ndarray, track: TrackState, face_id: str
    ):
        crop = self.detector.crop_face(frame, track.bbox)
        self._do_fire_exit(face_id, crop)
        self.event_logger.log(
            f"EXIT | face_id={face_id} | track_id={track.track_id}"
        )
        logger.info("EXIT → face=%s track=%d", face_id, track.track_id)

    def _do_fire_exit(self, face_id: str, crop: np.ndarray):
        ts = datetime.utcnow()
        image_path = self.event_logger.save_face_image(crop, face_id, "exit")
        self.db.log_event(face_id, "exit", image_path, timestamp=ts)

    # ──────────────────────────────────────────────────────────────────────
    #  Frame annotation
    # ──────────────────────────────────────────────────────────────────────

    def _draw_annotations(
        self, frame: np.ndarray, tracks: list
    ) -> np.ndarray:
        for track in tracks:
            x1, y1, x2, y2, _ = track.bbox
            face_id = self._track_to_face.get(track.track_id)
            label = f"T:{track.track_id}"
            if face_id:
                label += f" | {face_id[-6:]}"
            state = self._face_state.get(face_id, "") if face_id else ""
            color = (0, 220, 80) if state == "inside" else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        stats = self.db.get_stats()
        cv2.putText(
            frame,
            f"Unique visitors: {stats['today_unique_visitors']}  "
            f"Active: {len(tracks)}",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2,
        )
        return frame

    # ──────────────────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_face_id() -> str:
        return "face_" + uuid.uuid4().hex[:12]

    def get_stats(self) -> dict:
        return self.db.get_stats()

    def shutdown(self):
        self.db.close()
        self.event_logger.close()
        logger.info("Pipeline shut down cleanly.")