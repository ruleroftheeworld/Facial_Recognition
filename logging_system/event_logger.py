"""
logging_system/event_logger.py
Handles file-based event logging and structured face image storage.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Responsible for:
    1. Writing human-readable events to events.log.
    2. Saving cropped face images under logs/{entries|exits}/YYYY-MM-DD/.
    """

    def __init__(self, log_file: str, image_base_dir: str):
        self.image_base_dir = Path(image_base_dir)
        self._setup_file_logger(log_file)
        logger.info("EventLogger initialised. Log file: %s", log_file)

    # ------------------------------------------------------------------ #
    #  File logger setup                                                   #
    # ------------------------------------------------------------------ #

    def _setup_file_logger(self, log_file: str):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_logger = logging.getLogger("events")
        self._file_logger.setLevel(logging.INFO)
        self._file_logger.propagate = False   # don't double-log to root

        # rotating file handler: max 10 MB, keep 5 backups
        handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        self._file_logger.addHandler(handler)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def log(self, message: str, level: str = "info"):
        """Write a line to events.log."""
        getattr(self._file_logger, level.lower(), self._file_logger.info)(message)

    def save_face_image(
        self,
        face_crop: np.ndarray,
        face_id: str,
        event_type: str,           # "entry" | "exit" | "registration"
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Save face crop to disk.
        Path: logs/{event_type}s/YYYY-MM-DD/{face_id}_{HH-MM-SS-ms}.jpg

        Returns the relative path string.
        """
        ts = timestamp or datetime.utcnow()
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H-%M-%S-%f")[:-3]   # ms precision

        # event_type "entry" → folder "entries", "exit" → "exits"
        folder_name = event_type.rstrip("y") + "ies" if event_type == "entry" else event_type + "s"
        save_dir = self.image_base_dir / folder_name / date_str
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{face_id}_{time_str}.jpg"
        full_path = save_dir / filename

        if face_crop is not None and face_crop.size > 0:
            cv2.imwrite(str(full_path), face_crop)
        else:
            logger.warning("Empty crop for face %s – skipping image save.", face_id)

        relative_path = str(full_path.relative_to(Path(".")))
        self.log(
            f"IMAGE_SAVED | face_id={face_id} | type={event_type} | path={relative_path}"
        )
        return relative_path

    def close(self):
        """Flush and close all handlers."""
        for handler in self._file_logger.handlers:
            handler.flush()
            handler.close()
