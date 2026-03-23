"""
core/pipeline.py
Orchestrates detection → recognition → tracking → logging per frame.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

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
    Top-level controller that wires together every subsystem:
    Detector → Tracker → Recognizer → DB → Logger
    """

    def __init__(self, config: dict):
        self.config = config
        det_cfg = config["detection"]
        rec_cfg = config["recognition"]
        trk_cfg = config["tracking"]
        db_cfg = config["database"]
        log_cfg = config["logging"]

        # ---- subsystems ----
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

        # ---- runtime state ----
        self.frame_skip: int = det_cfg["frame_skip"]
        self._frame_count: int = 0
        # track_id → registered face_id
        self._track_to_face: Dict[int, str] = {}
        # face_id → True (already fired entry event)
        self._entry_fired: Dict[str, bool] = {}
        # cache of registered faces for matching
        self._face_cache: list = []
        self._refresh_face_cache()

        logger.info("FaceTrackerPipeline initialised.")

    # ------------------------------------------------------------------ #
    #  Cache                                                               #
    # ------------------------------------------------------------------ #

    def _refresh_face_cache(self):
        self._face_cache = self.db.get_all_faces()
        logger.debug("Face cache refreshed: %d known faces.", len(self._face_cache))

    # ------------------------------------------------------------------ #
    #  Per-frame processing                                                #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single BGR frame.
        Returns the annotated frame.
        """
        self._frame_count += 1
        run_detection = (self._frame_count % (self.frame_skip + 1) == 0)

        # ---------- Detection ----------
        if run_detection:
            detections = self.detector.detect(frame)
        else:
            detections = []   # tracker coasts on its own

        # ---------- Tracking ----------
        active_tracks = self.tracker.update(detections)

        # ---------- Recognition & Registration ----------
        if run_detection:
            for track in active_tracks:
                if track.track_id not in self._track_to_face:
                    self._identify_or_register(frame, track)

        # ---------- Handle exits ----------
        lost = self.tracker.remove_lost_tracks()
        for track in lost:
            face_id = self._track_to_face.pop(track.track_id, None)
            if face_id and self._entry_fired.get(face_id):
                self._fire_exit_event(frame, track, face_id)
                self._entry_fired.pop(face_id, None)

        # ---------- Annotate frame ----------
        annotated = self._draw_annotations(frame.copy(), active_tracks)
        return annotated

    # ------------------------------------------------------------------ #
    #  Recognition / Registration                                         #
    # ------------------------------------------------------------------ #

    def _identify_or_register(self, frame: np.ndarray, track: TrackState):
        crop = self.detector.crop_face(frame, track.bbox)
        embedding = self.recognizer.get_embedding(crop)
        if embedding is None:
            return

        face_id, similarity = self.recognizer.find_best_match(
            embedding, self._face_cache
        )

        if face_id is None:
            # ---- NEW face: register ----
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
            logger.info("New face registered: %s", face_id)
            self.event_logger.log(
                f"REGISTER | face_id={face_id} | track_id={track.track_id}"
            )
        else:
            self.db.update_face_last_seen(face_id)
            logger.debug("Recognised face %s (sim=%.3f)", face_id, similarity)

        self._track_to_face[track.track_id] = face_id
        track.face_id = face_id
        track.embedding = embedding

        # ---- Entry event ----
        if not self._entry_fired.get(face_id):
            self._fire_entry_event(frame, track, face_id)
            self._entry_fired[face_id] = True

    # ------------------------------------------------------------------ #
    #  Events                                                              #
    # ------------------------------------------------------------------ #

    def _fire_entry_event(self, frame: np.ndarray, track: TrackState,
                          face_id: str):
        ts = datetime.utcnow()
        crop = self.detector.crop_face(frame, track.bbox)
        image_path = self.event_logger.save_face_image(crop, face_id, "entry")
        self.db.log_event(face_id, "entry", image_path, timestamp=ts)
        self.event_logger.log(
            f"ENTRY | face_id={face_id} | track_id={track.track_id} | ts={ts.isoformat()}"
        )
        logger.info("ENTRY event – face=%s", face_id)

    def _fire_exit_event(self, frame: np.ndarray, track: TrackState,
                         face_id: str):
        ts = datetime.utcnow()
        crop = self.detector.crop_face(frame, track.bbox)
        image_path = self.event_logger.save_face_image(crop, face_id, "exit")
        self.db.log_event(face_id, "exit", image_path, timestamp=ts)
        self.event_logger.log(
            f"EXIT | face_id={face_id} | track_id={track.track_id} | ts={ts.isoformat()}"
        )
        logger.info("EXIT event – face=%s", face_id)

    # ------------------------------------------------------------------ #
    #  Annotation                                                          #
    # ------------------------------------------------------------------ #

    def _draw_annotations(self, frame: np.ndarray,
                           tracks: list) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        for track in tracks:
            x1, y1, x2, y2, _ = track.bbox
            face_id = self._track_to_face.get(track.track_id, "?")
            label = f"ID:{face_id[-6:] if face_id != '?' else '?'} T:{track.track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        # visitor counter overlay
        stats = self.db.get_stats()
        overlay_text = (
            f"Unique visitors: {stats['today_unique_visitors']}  "
            f"Active tracks: {len(tracks)}"
        )
        cv2.putText(
            frame, overlay_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )
        return frame

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_face_id() -> str:
        return "face_" + uuid.uuid4().hex[:12]

    def get_stats(self) -> dict:
        return self.db.get_stats()

    def shutdown(self):
        self.db.close()
        self.event_logger.close()
        logger.info("Pipeline shut down cleanly.")
