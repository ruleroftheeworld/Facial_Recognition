"""
core/detector.py
YOLOv8-based face detector wrapper.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Detection result type: List of (x1, y1, x2, y2, confidence)
BBox = Tuple[int, int, int, int, float]


class FaceDetector:
    """
    Wraps YOLOv8 (ultralytics) for face detection.
    Falls back to OpenCV Haar cascades if the YOLO model is unavailable.
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.4):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.use_fallback = False
        self._load_model(model_path)

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            p = Path(model_path)
            if not p.exists():
                logger.warning(
                    "YOLO model not found at %s – downloading 'yolov8n-face'…",
                    model_path,
                )
                # ultralytics will auto-download on first use
                self.model = YOLO("yolov8n.pt")
            else:
                self.model = YOLO(str(p))
            logger.info("YOLOv8 face detector loaded: %s", model_path)
        except ImportError:
            logger.warning(
                "ultralytics not installed – falling back to Haar cascade detector."
            )
            self._load_haar_fallback()

    def _load_haar_fallback(self):
        import cv2
        self.use_fallback = True
        cascade_path = (
            Path(__file__).parent.parent
            / "models"
            / "haarcascade_frontalface_default.xml"
        )
        if not cascade_path.exists():
            # use OpenCV's bundled cascade
            import cv2
            cascade_path = str(
                Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
            )
        self.model = __import__("cv2").CascadeClassifier(str(cascade_path))
        logger.info("Haar cascade fallback loaded.")

    # ------------------------------------------------------------------ #
    #  Detection                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Run face detection on a BGR frame.
        Returns list of (x1, y1, x2, y2, conf).
        """
        if self.use_fallback:
            return self._haar_detect(frame)
        return self._yolo_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> List[BBox]:
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        detections: List[BBox] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def _haar_detect(self, frame: np.ndarray) -> List[BBox]:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        result: List[BBox] = []
        for (x, y, w, h) in faces:
            result.append((x, y, x + w, y + h, 1.0))
        return result

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def crop_face(frame: np.ndarray, bbox: BBox,
                  padding: int = 10) -> np.ndarray:
        """Return a padded crop of the face region."""
        x1, y1, x2, y2, _ = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]
