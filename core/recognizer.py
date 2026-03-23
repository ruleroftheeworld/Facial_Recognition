"""
core/recognizer.py
InsightFace-based face embedding generator and matcher.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Generates 512-d ArcFace embeddings via InsightFace.
    Falls back to a simple ResNet feature extractor if InsightFace is absent.
    """

    def __init__(self, model_name: str = "buffalo_l",
                 embedding_threshold: float = 0.45,
                 min_face_size: int = 40):
        self.threshold = embedding_threshold
        self.min_face_size = min_face_size
        self.app = None
        self.use_fallback = False
        self._load_model(model_name)

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self, model_name: str):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model '%s' loaded.", model_name)
        except ImportError:
            logger.warning(
                "insightface not installed – using lightweight fallback embedder."
            )
            self._load_fallback()

    def _load_fallback(self):
        """Simple OpenCV DNN-based face descriptor (less accurate but dependency-free)."""
        self.use_fallback = True
        logger.info("Fallback face embedder active (lower accuracy).")

    # ------------------------------------------------------------------ #
    #  Embedding generation                                                #
    # ------------------------------------------------------------------ #

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Given a cropped face image (BGR, numpy array), return a normalised
        embedding vector, or None if no face detected in the crop.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        h, w = face_crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return None

        if self.use_fallback:
            return self._fallback_embedding(face_crop)
        return self._insightface_embedding(face_crop)

    def _insightface_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(face_crop)
        if not faces:
            return None
        # pick the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.embedding
        return emb / np.linalg.norm(emb)   # L2 normalise

    def _fallback_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        HOG-based 128-d descriptor as a last-resort fallback.
        Not production-grade but keeps the pipeline running.
        """
        import cv2
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        hog = cv2.HOGDescriptor(
            (64, 64), (16, 16), (8, 8), (8, 8), 9
        )
        descriptor = hog.compute(resized).flatten()
        # truncate / pad to 128
        descriptor = descriptor[:128] if len(descriptor) >= 128 else np.pad(descriptor, (0, 128 - len(descriptor)))
        norm = np.linalg.norm(descriptor)
        return descriptor / norm if norm > 0 else descriptor

    # ------------------------------------------------------------------ #
    #  Matching                                                            #
    # ------------------------------------------------------------------ #

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]; higher = more similar."""
        return float(np.dot(emb1, emb2))   # embeddings are already L2 normalised

    def find_best_match(
        self,
        query_emb: np.ndarray,
        registered_faces: List[dict],
    ) -> Tuple[Optional[str], float]:
        """
        Compare query embedding against all registered faces.

        Returns:
            (face_id, similarity) if above threshold, else (None, best_sim).
        """
        best_id: Optional[str] = None
        best_sim: float = -1.0

        for face in registered_faces:
            stored_emb = np.array(face["embedding"], dtype=np.float32)
            sim = self.cosine_similarity(query_emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = face["face_id"]

        if best_sim >= self.threshold:
            return best_id, best_sim
        return None, best_sim
