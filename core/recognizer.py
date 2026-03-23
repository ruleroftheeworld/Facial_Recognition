"""
core/recognizer.py
InsightFace ArcFace embedding generator.

OPTIMIZATION 4 — GPU acceleration:
  - CUDAExecutionProvider is always listed first; ONNX Runtime picks it if
    a CUDA-capable GPU is present, falls back to CPU automatically.
  - Provider actually used is logged at startup so you can verify at runtime.

OPTIMIZATION 5 — Quality filtering:
  - Laplacian variance check rejects blurry crops before embedding.
  - Configurable quality_laplacian_threshold (default 80).
  - min_face_size gate already existed; kept and documented.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """InsightFace ArcFace embeddings with GPU + quality optimisations."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        embedding_threshold: float = 0.60,
        min_face_size: int = 40,
        # OPTIMIZATION 5
        quality_laplacian_threshold: float = 80.0,
    ):
        self.threshold   = embedding_threshold
        self.min_face_size = min_face_size
        self.quality_lap = quality_laplacian_threshold  # OPTIMIZATION 5
        self.app         = None
        self.use_fallback = False
        self._active_provider: str = "unknown"
        self._load_model(model_name)

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self, model_name: str):
        try:
            from insightface.app import FaceAnalysis

            # OPTIMIZATION 4 — list CUDA first; ONNX Runtime auto-selects
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.app  = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))

            # Verify which provider is actually active
            self._log_active_provider()
            logger.info("InsightFace '%s' loaded. Provider: %s",
                        model_name, self._active_provider)
        except ImportError:
            logger.warning("insightface not installed — using HOG fallback.")
            self.use_fallback = True

    def _log_active_provider(self):
        """
        OPTIMIZATION 4 — runtime provider verification.
        Checks the first model's session to confirm CUDA vs CPU.
        """
        try:
            session = self.app.models[list(self.app.models.keys())[0]].session
            providers = session.get_providers()
            self._active_provider = providers[0] if providers else "unknown"
            if "CUDA" in self._active_provider:
                logger.info("InsightFace is running on GPU (CUDA).")
            else:
                logger.warning(
                    "InsightFace is running on CPU. "
                    "Install onnxruntime-gpu for GPU acceleration."
                )
        except Exception:
            self._active_provider = "unknown"

    # ------------------------------------------------------------------ #
    #  Embedding                                                           #
    # ------------------------------------------------------------------ #

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate L2-normalised 512-d ArcFace embedding.

        OPTIMIZATION 5 — quality gates applied before expensive inference:
          1. Minimum size check (min_face_size).
          2. Laplacian blur check (quality_laplacian_threshold).
        """
        if face_crop is None or face_crop.size == 0:
            return None

        h, w = face_crop.shape[:2]

        # Gate 1: minimum size
        if h < self.min_face_size or w < self.min_face_size:
            return None

        # OPTIMIZATION 5 Gate 2: blur / quality check
        if not self._is_sharp(face_crop):
            logger.debug("Crop rejected: below blur threshold.")
            return None

        if self.use_fallback:
            return self._hog_embedding(face_crop)
        return self._insightface_embedding(face_crop)

    def _is_sharp(self, crop: np.ndarray) -> bool:
        """
        OPTIMIZATION 5 — Laplacian variance sharpness check.
        A blurry face crop produces a poor embedding and wastes GPU time.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= self.quality_lap

    def _insightface_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(crop)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb  = face.embedding
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-6 else emb

    def _hog_embedding(self, crop: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        hog     = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
        desc    = hog.compute(resized).flatten()
        desc    = desc[:128] if len(desc) >= 128 \
                  else np.pad(desc, (0, 128 - len(desc)))
        norm    = np.linalg.norm(desc)
        return desc / norm if norm > 1e-6 else desc

    # ------------------------------------------------------------------ #
    #  Matching helpers (used by IdentityRegistry)                        #
    # ------------------------------------------------------------------ #

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))   # both already L2-normalised

    def find_best_match(
        self, query: np.ndarray, faces: List[dict]
    ) -> Tuple[Optional[str], float]:
        best_id, best_sim = None, -1.0
        for face in faces:
            sim = self.cosine_similarity(query, np.array(face["embedding"], dtype=np.float32))
            if sim > best_sim:
                best_sim, best_id = sim, face["face_id"]
        return (best_id, best_sim) if best_sim >= self.threshold else (None, best_sim)

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    @property
    def active_provider(self) -> str:
        """OPTIMIZATION 4 — expose active ONNX provider for external checks."""
        return self._active_provider