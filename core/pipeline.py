"""
core/pipeline.py

Optimised full-frame, crossing-robust face tracking pipeline.

Optimisations applied in this file
------------------------------------
  OPT-1  Frame resize — handled inside FaceDetector.detect(); pipeline passes
         the original frame and gets back original-scale boxes.

  OPT-2  frame_skip raised to 5 in config (detection every 6th frame).

  OPT-3  Delayed recognition — new tracks wait min_track_frames_for_recognition
         stable frames before embedding is attempted.  Unstable tracks (ones
         that appear and disappear quickly during crossings) are never sent to
         the GPU at all.

  OPT-5  Face quality filter — FaceDetector already drops small detections;
         FaceRecognizer rejects blurry crops via Laplacian check.

  OPT-6  Async DB writes — all MongoDB calls are submitted through
         AsyncDBWriter.submit() so the pipeline thread never blocks on I/O.

  OPT-7  ByteTrack note — ByteTrackWrapper can be swapped by pointing
         tracker_type to a real ByteTrack implementation; interface unchanged.

  OPT-8  Performance monitor — PerformanceMonitor.measure() wraps each stage;
         FPS + per-stage breakdown logged every fps_log_interval frames.

  OPT-9  Rendering optimised — bounding box drawing skipped when display
         is disabled; annotation is only called when needed.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional, Set

import cv2
import numpy as np

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.tracker import build_tracker, TrackState
from core.identity_registry import IdentityRegistry
from database.mongo_manager import MongoManager
from database.async_writer import AsyncDBWriter          # OPTIMIZATION 6
from logging_system.event_logger import EventLogger
from utils.performance_monitor import PerformanceMonitor  # OPTIMIZATION 8

logger = logging.getLogger(__name__)


class FaceTrackerPipeline:

    def __init__(self, config: dict):
        self.config = config
        det_cfg  = config["detection"]
        rec_cfg  = config["recognition"]
        trk_cfg  = config["tracking"]
        reid_cfg = config.get("reid", {})
        db_cfg   = config["database"]
        log_cfg  = config["logging"]
        perf_cfg = config.get("performance", {})

        # ── Core subsystems ─────────────────────────────────────────────
        # OPTIMIZATION 1 params forwarded to detector
        self.detector = FaceDetector(
            model_path=det_cfg["yolo_model"],
            conf_thresh=det_cfg["confidence_threshold"],
            iou_thresh=det_cfg["iou_threshold"],
            input_width=det_cfg.get("input_width", 640),      # OPT-1
            input_height=det_cfg.get("input_height", 360),    # OPT-1
            min_face_area=rec_cfg.get("min_face_area", 1600), # OPT-5
        )
        self.recognizer = FaceRecognizer(
            model_name=rec_cfg["model_name"],
            embedding_threshold=rec_cfg["embedding_threshold"],
            min_face_size=rec_cfg["min_face_size"],
            quality_laplacian_threshold=rec_cfg.get(          # OPT-5
                "quality_laplacian_threshold", 80.0
            ),
        )
        self.tracker = build_tracker(
            tracker_type=trk_cfg["tracker_type"],
            max_disappeared=trk_cfg["max_disappeared"],
            max_distance=trk_cfg["max_distance"],
        )

        # ── Database + async writer ──────────────────────────────────────
        self.db = MongoManager(
            uri=db_cfg["uri"],
            db_name=db_cfg["name"],
            collections=db_cfg["collections"],
        )
        # OPTIMIZATION 6 — wrap all writes through async queue
        if db_cfg.get("async_write", True):
            self._async_writer = AsyncDBWriter(
                max_queue_size=db_cfg.get("async_queue_size", 500)
            )
            self._async_writer.start()
            logger.info("Async DB writer enabled.")
        else:
            self._async_writer = None
            logger.info("Async DB writer disabled (synchronous mode).")

        self.event_logger = EventLogger(
            log_file=log_cfg["log_file"],
            image_base_dir=log_cfg["image_base_dir"],
        )

        # ── Identity registry ────────────────────────────────────────────
        self.registry = IdentityRegistry(rec_cfg, reid_cfg)
        self.registry.load_from_db(self.db.get_all_faces())

        # ── OPTIMIZATION 8 — performance monitor ────────────────────────
        self._monitor = PerformanceMonitor(
            log_interval=log_cfg.get("fps_log_interval", 100)
        )

        # ── OPTIMIZATION 3 — delayed recognition config ─────────────────
        self._min_stable_frames: int = rec_cfg.get(
            "min_track_frames_for_recognition", 5
        )
        # track_id → frame count (how many frames this track has been seen)
        self._track_frame_count: Dict[int, int] = {}

        # ── OPTIMIZATION 2 — frame skip ─────────────────────────────────
        self.frame_skip: int = det_cfg["frame_skip"]   # now default 5

        # ── Runtime state ────────────────────────────────────────────────
        self._frame_count: int = 0
        self._track_last_emb: Dict[int, np.ndarray] = {}
        self._track_last_centroid: Dict[int, np.ndarray] = {}
        self._unique_face_ids: Set[str] = set()
        self._active_face_ids: Set[str] = set()

        # Log GPU provider confirmation (OPT-4)
        logger.info(
            "InsightFace provider: %s", self.recognizer.active_provider
        )
        logger.info("Pipeline ready. frame_skip=%d min_stable_frames=%d",
                    self.frame_skip, self._min_stable_frames)

    # ──────────────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, display: bool = True) -> np.ndarray:
        """
        Process one BGR frame and return annotated output.
        OPT-9: pass display=False to skip annotation entirely.
        """
        self._monitor.tick()          # OPTIMIZATION 8
        self._frame_count += 1

        # ── OPT-2: Detection every frame_skip+1 frames ─────────────────
        with self._monitor.measure("detection"):          # OPTIMIZATION 8
            if self._frame_count % (self.frame_skip + 1) == 0:
                detections = self.detector.detect(frame)  # OPT-1 inside
            else:
                detections = []

        # ── Tracking ────────────────────────────────────────────────────
        with self._monitor.measure("tracking"):           # OPTIMIZATION 8
            active_tracks = self.tracker.update(detections)

        # ── Per-track identity resolution ───────────────────────────────
        current_active: Set[str] = set()
        with self._monitor.measure("recognition"):        # OPTIMIZATION 8
            for track in active_tracks:
                cx = float(track.centroid[0])
                cy = float(track.centroid[1])
                centroid = np.array([cx, cy], dtype=np.float32)
                self._track_last_centroid[track.track_id] = centroid

                # OPT-3: increment per-track frame counter
                self._track_frame_count[track.track_id] = \
                    self._track_frame_count.get(track.track_id, 0) + 1

                face_id = self._process_track(frame, track, centroid)
                if face_id:
                    current_active.add(face_id)

        # ── Lost track cleanup ───────────────────────────────────────────
        for lost in self.tracker.remove_lost_tracks():
            self._on_track_lost(lost)

        self._active_face_ids = current_active

        # ── OPT-9: skip annotation if display is off ────────────────────
        if display:
            with self._monitor.measure("annotation"):     # OPTIMIZATION 8
                frame = self._draw_annotations(frame.copy(), active_tracks)

        self._monitor.maybe_log()     # OPTIMIZATION 8 — log every N frames
        return frame

    # ──────────────────────────────────────────────────────────────────────
    #  Per-track processing
    # ──────────────────────────────────────────────────────────────────────

    def _process_track(
        self,
        frame: np.ndarray,
        track: TrackState,
        centroid: np.ndarray,
    ) -> Optional[str]:
        tid = track.track_id

        # Already resolved — refresh confidence opportunistically
        if not self.registry.needs_recognition(tid):
            face_id = self.registry.get_face_id(tid)
            if face_id:
                crop = self.detector.crop_face(frame, track.bbox)
                emb  = self.recognizer.get_embedding(crop)
                if emb is not None:
                    self._track_last_emb[tid] = emb
                    self.registry.update_confidence(tid, emb, self._frame_count)
            return face_id

        # OPTIMIZATION 3 — only attempt recognition once track is stable
        stable_frames = self._track_frame_count.get(tid, 0)
        if stable_frames < self._min_stable_frames:
            logger.debug(
                "Track %d not yet stable (%d/%d frames) — skipping recognition.",
                tid, stable_frames, self._min_stable_frames,
            )
            return None

        # Generate embedding (OPT-5 quality gates inside recognizer)
        crop    = self.detector.crop_face(frame, track.bbox)
        raw_emb = self.recognizer.get_embedding(crop)
        if raw_emb is None:
            return None

        self._track_last_emb[tid] = raw_emb
        ready_emb = self.registry.accumulate_embedding(tid, raw_emb)
        if ready_emb is None:
            return None

        face_id, confidence, is_new = self.registry.resolve(
            tid, ready_emb, centroid, self._frame_count
        )

        if is_new:
            self._register_new_face(face_id, ready_emb, crop, track)
        else:
            # OPTIMIZATION 6 — async DB update (non-blocking)
            self._db_submit(self.db.update_face_last_seen, face_id)
            self.event_logger.log(
                f"RE-ID | face_id={face_id} | track_id={tid} "
                f"| conf={confidence:.3f} | frame={self._frame_count}"
            )

        self._unique_face_ids.add(face_id)
        track.face_id = face_id
        return face_id

    # ──────────────────────────────────────────────────────────────────────
    #  Registration (async writes)
    # ──────────────────────────────────────────────────────────────────────

    def _register_new_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        crop: np.ndarray,
        track: TrackState,
    ):
        thumbnail  = self.event_logger.save_face_image(crop, face_id, "registration")
        entry_img  = self.event_logger.save_face_image(crop, face_id, "entry")
        ts         = datetime.utcnow()
        meta       = {"track_id": track.track_id, "first_seen_frame": self._frame_count}

        # OPTIMIZATION 6 — both writes are async; pipeline doesn't wait
        self._db_submit(
            self.db.register_face,
            face_id=face_id,
            embedding=embedding.tolist(),
            image_path=thumbnail,
            metadata=meta,
        )
        self._db_submit(
            self.db.log_event,
            face_id, "entry", entry_img,
            timestamp=ts,
            extra=meta,
        )

        self.event_logger.log(
            f"ENTRY | face_id={face_id} | track_id={track.track_id} "
            f"| frame={self._frame_count}"
        )
        logger.info("NEW face: %s track_id=%d", face_id, track.track_id)

    # ──────────────────────────────────────────────────────────────────────
    #  Track loss
    # ──────────────────────────────────────────────────────────────────────

    def _on_track_lost(self, track: TrackState):
        tid      = track.track_id
        face_id  = self.registry.get_face_id(tid)
        last_emb = self._track_last_emb.pop(tid, None)
        last_cen = self._track_last_centroid.pop(tid, None)
        self._track_frame_count.pop(tid, None)   # OPT-3 cleanup

        self.registry.on_track_lost(
            tid, last_emb,
            last_cen if last_cen is not None else track.centroid,
        )

        if face_id:
            self._active_face_ids.discard(face_id)
            ts = datetime.utcnow()
            # OPTIMIZATION 6 — async exit event write
            self._db_submit(
                self.db.log_event,
                face_id, "exit", "",
                timestamp=ts,
                extra={"track_id": tid, "frame": self._frame_count},
            )
            self.event_logger.log(
                f"EXIT | face_id={face_id} | track_id={tid} "
                f"| frame={self._frame_count}"
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Async DB helper
    # ──────────────────────────────────────────────────────────────────────

    def _db_submit(self, fn, *args, **kwargs):
        """
        OPTIMIZATION 6 — route write through async queue when enabled,
        fall back to synchronous call if async writer is off.
        """
        if self._async_writer:
            self._async_writer.submit(fn, *args, **kwargs)
        else:
            fn(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    #  Annotation  (OPT-9: only called when display=True)
    # ──────────────────────────────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        # OPTIMIZATION 9 — only draw resolved tracks to reduce text calls
        for track in tracks:
            x1, y1, x2, y2, _ = track.bbox
            face_id = self.registry.get_face_id(track.track_id)
            stable  = self._track_frame_count.get(track.track_id, 0)

            if face_id:
                label = f"{face_id[-8:]}  T:{track.track_id}"
                color = (0, 220, 80)
            elif stable < self._min_stable_frames:
                label = f"T:{track.track_id} [{stable}/{self._min_stable_frames}]"
                color = (120, 120, 120)   # grey = pending stability
            else:
                label = f"T:{track.track_id} [resolving]"
                color = (180, 180, 50)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
            )

        # Stats overlay — single putText call
        writer_stats = self._async_writer.stats if self._async_writer else {}
        cv2.putText(
            frame,
            f"Unique:{len(self._unique_face_ids)}  "
            f"Active:{len(self._active_face_ids)}  "
            f"FPS:{self._monitor.current_fps:.1f}  "
            f"Q:{writer_stats.get('queue_depth', 0)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2,
        )
        return frame

    # ──────────────────────────────────────────────────────────────────────
    #  Public
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        writer_stats = self._async_writer.stats if self._async_writer else {}
        return {
            **self.db.get_stats(),
            "session_unique_faces":  len(self._unique_face_ids),
            "session_active_faces":  len(self._active_face_ids),
            "registry_size":         self.registry.registry_size(),
            "current_frame":         self._frame_count,
            "current_fps":           round(self._monitor.current_fps, 2),
            "db_writer":             writer_stats,
        }

    def shutdown(self):
        logger.info("Shutting down pipeline...")
        if self._async_writer:
            self._async_writer.drain(timeout=5.0)
            self._async_writer.stop()
        self.db.close()
        self.event_logger.close()
        logger.info(
            "Shutdown complete. Unique faces: %d | Avg FPS: %.1f",
            len(self._unique_face_ids), self._monitor.current_fps,
        )