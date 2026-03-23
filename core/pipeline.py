"""
core/pipeline.py

High-performance real-time face tracking pipeline with intelligent analytics.

Performance optimisations (all retained from original):
  OPT-1  Frame resize before detection (handled inside FaceDetector)
  OPT-2  frame_skip — detection every N+1 frames
  OPT-3  Delayed recognition — min_track_frames_for_recognition
  OPT-4  InsightFace CUDAExecutionProvider (inside FaceRecognizer)
  OPT-5  Face filtering — small faces + blur check (detector + recognizer)
  OPT-6  Async MongoDB writes via AsyncDBWriter
  OPT-7  ByteTrack-compatible centroid tracker
  OPT-8  PerformanceMonitor — per-stage FPS breakdown
  OPT-9  No-display mode — annotation skipped when display=False

Analytics added (Part 2):
  A1  Dwell time — entry/exit timestamps, dwell_time, engagement category
  A2  Returning visitor intelligence — visit_count, tag (new/occasional/frequent)
  A3  Loitering / suspicious detection — alerts to DB + log file
  A4  Explainable AI logging (XAI) — reason strings on every resolve event
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
from database.async_writer import AsyncDBWriter
from logging_system.event_logger import EventLogger
from utils.performance_monitor import PerformanceMonitor

# Analytics modules (Part 2)
from analytics.dwell_tracker import DwellTracker
from analytics.visitor_intelligence import VisitorIntelligence
from analytics.loitering_detector import LoiteringDetector

logger = logging.getLogger(__name__)


class FaceTrackerPipeline:

    def __init__(self, config: dict):
        self.config   = config
        det_cfg       = config["detection"]
        rec_cfg       = config["recognition"]
        trk_cfg       = config["tracking"]
        reid_cfg      = config.get("reid", {})
        db_cfg        = config["database"]
        log_cfg       = config["logging"]
        analytics_cfg = config.get("analytics", {})

        # ── Core subsystems ──────────────────────────────────────────────
        self.detector = FaceDetector(
            model_path=det_cfg["yolo_model"],
            conf_thresh=det_cfg["confidence_threshold"],
            iou_thresh=det_cfg["iou_threshold"],
            input_width=det_cfg.get("input_width", 640),
            input_height=det_cfg.get("input_height", 360),
            min_face_area=rec_cfg.get("min_face_area", 1600),
        )
        self.recognizer = FaceRecognizer(
            model_name=rec_cfg["model_name"],
            embedding_threshold=rec_cfg["embedding_threshold"],
            min_face_size=rec_cfg["min_face_size"],
            quality_laplacian_threshold=rec_cfg.get("quality_laplacian_threshold", 80.0),
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
        all_faces = self.db.get_all_faces()
        self.registry.load_from_db(all_faces)

        # ── Performance monitor ──────────────────────────────────────────
        self._monitor = PerformanceMonitor(
            log_interval=log_cfg.get("fps_log_interval", 100)
        )

        # ── Analytics A1: Dwell time ─────────────────────────────────────
        self._dwell = DwellTracker(
            dwell_threshold=analytics_cfg.get("dwell_time_threshold", 20)
        )

        # ── Analytics A2: Returning visitor intelligence ─────────────────
        self._visitor = VisitorIntelligence()
        self._visitor.preload(all_faces)

        # ── Analytics A3: Loitering / suspicious detection ───────────────
        self._loitering = LoiteringDetector(
            dwell_threshold=analytics_cfg.get("dwell_time_threshold", 20),
            suspicious_visit_threshold=analytics_cfg.get("suspicious_visit_threshold", 5),
            suspicious_window_seconds=analytics_cfg.get("suspicious_window_seconds", 3600),
        )

        # ── Config cache ─────────────────────────────────────────────────
        self._min_stable_frames: int = analytics_cfg.get(
            "min_frames_for_recognition",
            rec_cfg.get("min_track_frames_for_recognition", 5),
        )
        self.frame_skip: int = det_cfg["frame_skip"]

        # ── Runtime state ────────────────────────────────────────────────
        self._frame_count: int                           = 0
        self._track_frame_count: Dict[int, int]          = {}
        self._track_last_emb: Dict[int, np.ndarray]      = {}
        self._track_last_centroid: Dict[int, np.ndarray] = {}
        self._unique_face_ids: Set[str]                  = set()
        self._active_face_ids: Set[str]                  = set()

        logger.info("InsightFace provider: %s", self.recognizer.active_provider)
        logger.info(
            "Pipeline ready. frame_skip=%d  min_stable_frames=%d  "
            "dwell_threshold=%.0fs  suspicious_visits=%d",
            self.frame_skip, self._min_stable_frames,
            self._dwell.dwell_threshold, self._loitering.suspicious_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, display: bool = True) -> np.ndarray:
        """
        Process one BGR frame.  Returns annotated frame (or original if display=False).
        OPT-9: pass display=False to skip annotation entirely for max throughput.
        """
        self._monitor.tick()
        self._frame_count += 1

        # OPT-2: run detector only every frame_skip+1 frames
        with self._monitor.measure("detection"):
            if self._frame_count % (self.frame_skip + 1) == 0:
                detections = self.detector.detect(frame)  # OPT-1 inside
            else:
                detections = []

        # Tracking
        with self._monitor.measure("tracking"):
            active_tracks = self.tracker.update(detections)

        # Per-track identity resolution + analytics
        current_active: Set[str] = set()
        with self._monitor.measure("recognition"):
            for track in active_tracks:
                cx, cy   = float(track.centroid[0]), float(track.centroid[1])
                centroid = np.array([cx, cy], dtype=np.float32)
                self._track_last_centroid[track.track_id] = centroid
                self._track_frame_count[track.track_id] = (
                    self._track_frame_count.get(track.track_id, 0) + 1
                )
                face_id = self._process_track(frame, track, centroid)
                if face_id:
                    current_active.add(face_id)

        # A3: loitering check for every currently resolved face
        with self._monitor.measure("analytics"):
            for track in active_tracks:
                face_id = self.registry.get_face_id(track.track_id)
                if not face_id:
                    continue
                self._loitering.check(
                    face_id=face_id,
                    current_dwell=self._dwell.get_current_dwell(face_id),
                    visit_count=self._visitor.visit_count(face_id),
                    db_submit=self._db_submit,
                    log_alert_fn=self.db.log_alert,
                    event_logger=self.event_logger,
                )

        # Lost track cleanup
        for lost in self.tracker.remove_lost_tracks():
            self._on_track_lost(lost)

        self._active_face_ids = current_active

        # OPT-9: skip annotation when display is off
        if display:
            with self._monitor.measure("annotation"):
                frame = self._draw_annotations(frame.copy(), active_tracks)

        self._monitor.maybe_log()
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

        # Already resolved — opportunistic confidence refresh
        if not self.registry.needs_recognition(tid):
            face_id = self.registry.get_face_id(tid)
            if face_id:
                crop = self.detector.crop_face(frame, track.bbox)
                emb  = self.recognizer.get_embedding(crop)
                if emb is not None:
                    self._track_last_emb[tid] = emb
                    self.registry.update_confidence(tid, emb, self._frame_count)
            return face_id

        # OPT-3: wait until track is stable before spending GPU time
        stable_frames = self._track_frame_count.get(tid, 0)
        if stable_frames < self._min_stable_frames:
            logger.debug(
                "Track %d not yet stable (%d/%d) — skipping recognition.",
                tid, stable_frames, self._min_stable_frames,
            )
            return None

        # Generate embedding (OPT-5 quality gates are inside recognizer)
        crop    = self.detector.crop_face(frame, track.bbox)
        raw_emb = self.recognizer.get_embedding(crop)
        if raw_emb is None:
            return None

        self._track_last_emb[tid] = raw_emb
        ready_emb = self.registry.accumulate_embedding(tid, raw_emb)
        if ready_emb is None:
            return None

        # A4: XAI — pass event_logger so reasoning is written to events.log
        face_id, confidence, is_new = self.registry.resolve(
            track_id=tid,
            embedding=ready_emb,
            centroid=centroid,
            frame_no=self._frame_count,
            event_logger=self.event_logger,
            match_threshold=self.recognizer.threshold,
        )

        if is_new:
            self._register_new_face(face_id, ready_emb, crop, track)
        else:
            # Returning visit
            self._db_submit(self.db.update_face_last_seen, face_id)
            # A2: update visitor intelligence
            self._visitor.on_entry(
                face_id=face_id,
                is_new=False,
                db_submit=self._db_submit,
                upsert_visitor_fn=self.db.upsert_visitor_profile,
            )
            # A1: start dwell timer for this visit
            self._dwell.on_entry(face_id)

            profile = self._visitor.get_profile(face_id)
            tag     = profile["tag"] if profile else "unknown"
            self.event_logger.log(
                f"RE-ID | face_id={face_id} | track_id={tid} "
                f"| conf={confidence:.3f} | visit_count={self._visitor.visit_count(face_id)} "
                f"| tag={tag} | frame={self._frame_count}"
            )

        self._unique_face_ids.add(face_id)
        track.face_id = face_id
        return face_id

    # ──────────────────────────────────────────────────────────────────────
    #  Registration
    # ──────────────────────────────────────────────────────────────────────

    def _register_new_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        crop: np.ndarray,
        track: TrackState,
    ):
        thumbnail = self.event_logger.save_face_image(crop, face_id, "registration")
        entry_img = self.event_logger.save_face_image(crop, face_id, "entry")
        ts        = datetime.utcnow()
        meta      = {"track_id": track.track_id, "first_seen_frame": self._frame_count}

        # OPT-6: async writes — pipeline doesn't block on I/O
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

        # A2: initialise new visitor profile
        self._visitor.on_entry(
            face_id=face_id,
            is_new=True,
            db_submit=self._db_submit,
            upsert_visitor_fn=self.db.upsert_visitor_profile,
        )

        # A1: start dwell timer
        self._dwell.on_entry(face_id)

        profile = self._visitor.get_profile(face_id)
        tag     = profile["tag"] if profile else "new"
        self.event_logger.log(
            f"ENTRY | face_id={face_id} | track_id={track.track_id} "
            f"| tag={tag} | frame={self._frame_count}"
        )
        logger.info("NEW face: %s  track_id=%d  tag=%s", face_id, track.track_id, tag)

    # ──────────────────────────────────────────────────────────────────────
    #  Track loss
    # ──────────────────────────────────────────────────────────────────────

    def _on_track_lost(self, track: TrackState):
        tid      = track.track_id
        face_id  = self.registry.get_face_id(tid)
        last_emb = self._track_last_emb.pop(tid, None)
        last_cen = self._track_last_centroid.pop(tid, None)
        self._track_frame_count.pop(tid, None)

        self.registry.on_track_lost(
            tid, last_emb,
            last_cen if last_cen is not None else track.centroid,
        )

        if not face_id:
            return

        self._active_face_ids.discard(face_id)
        ts = datetime.utcnow()

        # A1: compute and persist dwell analytics
        dwell_result = self._dwell.on_exit(
            face_id=face_id,
            db_submit=self._db_submit,
            update_exit_event_fn=self.db.update_exit_event_dwell,
            update_visitor_dwell_fn=self.db.update_visitor_dwell_stats,
        )

        dwell_str = ""
        if dwell_result:
            dwell_str = (
                f"| dwell={dwell_result['dwell_time']:.1f}s "
                f"| category={dwell_result['category']}"
            )

        # OPT-6: async exit event write
        self._db_submit(
            self.db.log_event,
            face_id, "exit", "",
            timestamp=ts,
            extra={"track_id": tid, "frame": self._frame_count},
        )

        # A3: clear alert cooldown state for this face
        self._loitering.clear(face_id)

        self.event_logger.log(
            f"EXIT | face_id={face_id} | track_id={tid} "
            f"| frame={self._frame_count} {dwell_str}"
        )
        logger.info("EXIT: %s  track_id=%d  %s", face_id, tid, dwell_str)

    # ──────────────────────────────────────────────────────────────────────
    #  Async DB helper
    # ──────────────────────────────────────────────────────────────────────

    def _db_submit(self, fn, *args, **kwargs):
        """OPT-6: route write through async queue when enabled, else synchronous."""
        if self._async_writer:
            self._async_writer.submit(fn, *args, **kwargs)
        else:
            fn(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    #  Annotation  (OPT-9: only called when display=True)
    # ──────────────────────────────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        for track in tracks:
            x1, y1, x2, y2, _ = track.bbox
            face_id = self.registry.get_face_id(track.track_id)
            stable  = self._track_frame_count.get(track.track_id, 0)

            if face_id:
                visit_count = self._visitor.visit_count(face_id)
                tag         = self._visitor.tag(face_id)
                dwell       = self._dwell.get_current_dwell(face_id)
                dwell_str   = f" {dwell:.0f}s" if dwell is not None else ""
                label       = f"{face_id[-8:]} v{visit_count}/{tag}{dwell_str}"
                color       = (0, 220, 80)
                # Red box = loitering alert
                if dwell is not None and dwell > self._dwell.dwell_threshold:
                    color = (0, 0, 230)
            elif stable < self._min_stable_frames:
                label = f"T:{track.track_id} [{stable}/{self._min_stable_frames}]"
                color = (120, 120, 120)
            else:
                label = f"T:{track.track_id} [resolving]"
                color = (180, 180, 50)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
            )

        # HUD overlay
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
            "session_unique_faces": len(self._unique_face_ids),
            "session_active_faces": len(self._active_face_ids),
            "registry_size":        self.registry.registry_size(),
            "current_frame":        self._frame_count,
            "current_fps":          round(self._monitor.current_fps, 2),
            "db_writer":            writer_stats,
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
