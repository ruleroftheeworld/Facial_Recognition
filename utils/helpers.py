"""
utils/helpers.py
Utility functions: config loading, root logger setup, frame drawing helpers.
"""

import json
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any, Dict


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load and validate config.json. Raises on missing required keys."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_top_keys = ["detection", "recognition", "tracking", "database",
                         "logging", "camera"]
    missing = [k for k in required_top_keys if k not in cfg]
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")

    return cfg


# ------------------------------------------------------------------ #
#  Logging setup                                                       #
# ------------------------------------------------------------------ #

def setup_logging(level: str = "INFO", log_file: str = "logs/app.log"):
    """Configure root logger with console + rotating file handlers."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filename": log_file,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": level,
        },
    })


# ------------------------------------------------------------------ #
#  Video source helpers                                                #
# ------------------------------------------------------------------ #

def open_video_source(camera_config: Dict[str, Any]):
    """
    Open an OpenCV VideoCapture from config.
    Supports video file or RTSP stream.
    """
    import cv2

    use_rtsp = camera_config.get("use_rtsp", False)
    source = (
        camera_config["rtsp_url"] if use_rtsp
        else camera_config["source"]
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.getLogger(__name__).info(
        "Video source opened: %s  |  %dx%d @ %.1f FPS", source, width, height, fps
    )
    return cap, fps, width, height


# ------------------------------------------------------------------ #
#  Misc                                                                #
# ------------------------------------------------------------------ #

def ensure_dirs(*paths: str):
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
