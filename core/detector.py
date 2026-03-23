"""
core/detector.py
YOLOv8-based face detector with frame-resize optimisation.

OPTIMIZATION 1 — Frame resize before detection:
  - Incoming frame is resized to (input_width × input_height) before YOLO.
  - YOLO runs on the smaller tensor → significantly fewer FLOPs.
  - Bounding boxes are scaled back to original resolution so all downstream
    code (tracker, crop, annotation) works on original coordinates.
  - Default: 640×360 (~4× fewer pixels than 1080p).

OPTIMIZATION 5 — Face filtering:
  - min_face_area rejects tiny detections before any embedding work.
  - Low-quality crops are caught later in the recognizer via Laplacian blur
    check, but oversized/noise detections are dropped here immediately.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]   # x1, y1, x2, y2, conf


class FaceDetector:
    """YOLOv8 face detector with resize-before-detect optimisation."""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.4,
        # OPTIMIZATION 1
        input_width: int = 640,
        input_height: int = 360,
        # OPTIMIZATION 5
        min_face_area: int = 1600,
    ):
        self.conf_thresh   = conf_thresh
        self.iou_thresh    = iou_thresh
        self.input_width   = input_width    # OPTIMIZATION 1
        self.input_height  = input_height   # OPTIMIZATION 1
        self.min_face_area = min_face_area  # OPTIMIZATION 5
        self.model         = None
        self.use_fallback  = False
        self._load_model(model_path)

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            p = Path(model_path)
            self.model = YOLO(str(p) if p.exists() else "yolov8n.pt")
            logger.info("YOLOv8 loaded: %s", model_path)
        except ImportError:
            logger.warning("ultralytics not installed — using Haar fallback.")
            self._load_haar_fallback()

    def _load_haar_fallback(self):
        import cv2 as _cv2
        self.use_fallback = True
        cascade_path = str(
            Path(_cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
        )
        self.model = _cv2.CascadeClassifier(cascade_path)
        logger.info("Haar cascade fallback loaded.")

    # ------------------------------------------------------------------ #
    #  Detection                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Detect faces in a BGR frame.

        OPTIMIZATION 1: resizes to (input_width × input_height) before
        inference, then scales boxes back to original resolution.

        OPTIMIZATION 5: drops detections smaller than min_face_area pixels².
        """
        orig_h, orig_w = frame.shape[:2]

        # -- OPTIMIZATION 1: resize for inference ----------------------
        if orig_w != self.input_width or orig_h != self.input_height:
            small = cv2.resize(
                frame, (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height
        else:
            small = frame
            scale_x = scale_y = 1.0

        raw = self._yolo_detect(small) if not self.use_fallback \
              else self._haar_detect(small)

        # -- Scale back + area filter ----------------------------------
        result: List[BBox] = []
        for (x1, y1, x2, y2, conf) in raw:
            x1s = int(x1 * scale_x)
            y1s = int(y1 * scale_y)
            x2s = int(x2 * scale_x)
            y2s = int(y2 * scale_y)
            # OPTIMIZATION 5: reject tiny detections
            area = (x2s - x1s) * (y2s - y1s)
            if area < self.min_face_area:
                continue
            result.append((x1s, y1s, x2s, y2s, conf))

        return result

    def _yolo_detect(self, frame: np.ndarray) -> List[BBox]:
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        out: List[BBox] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                out.append((x1, y1, x2, y2, float(box.conf[0])))
        return out

    def _haar_detect(self, frame: np.ndarray) -> List[BBox]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces]

    # ------------------------------------------------------------------ #
    #  Crop utility                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def crop_face(frame: np.ndarray, bbox: BBox, padding: int = 10) -> np.ndarray:
        x1, y1, x2, y2, _ = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]