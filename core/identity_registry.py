"""
core/identity_registry.py

Central identity store that sits between the pipeline and MongoDB.

Responsibilities
----------------
1. In-memory face registry
   - Holds all known face_id → embedding mappings loaded from DB at startup.
   - Updated immediately when a new face is registered so intra-session
     crossings are matched without another DB round-trip.

2. Embedding averaging (robustness fix)
   - Accumulates up to `avg_frames` embeddings per track before committing.
   - The averaged, re-normalised embedding is far more stable than a single
     frame crop, especially near occlusion or during crossing.

3. Ghost buffer (short-term re-ID memory)
   - When a track disappears, its last known embedding + centroid + timestamp
     are placed in a "ghost" entry for `ghost_ttl_seconds`.
   - When a new track appears, it is first matched against ghosts using
     cosine similarity AND spatial proximity before hitting the full registry.
   - This is the primary fix for crossing: the new track_id that re-appears
     after the occlusion matches its own ghost instantly.

4. Duplicate guard (global uniqueness enforcement)
   - Before inserting any new face_id, a full registry scan is performed at
     a stricter threshold (duplicate_guard_threshold > embedding_threshold).
   - Prevents two slightly-different crops of the same person from creating
     separate identities.

5. Confidence-gated cache updates
   - Cache entries store (face_id, similarity_score).
   - A new match only overwrites the cached entry if its similarity is higher
     than the currently stored confidence. This prevents a low-confidence
     match during a crossing from corrupting a previously solid identity.
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class GhostEntry:
    """Recently-lost track kept alive for re-ID during crossings."""
    face_id: str
    embedding: np.ndarray
    centroid: np.ndarray          # (cx, cy) at last seen frame
    lost_at: float                # time.time() when track disappeared
    ttl: float                    # seconds to keep alive


@dataclass
class CacheEntry:
    """Per-track identity assignment with confidence score."""
    face_id: str
    confidence: float             # cosine similarity that produced this match
    frame_assigned: int


@dataclass
class PendingEmbedding:
    """Accumulates frames before committing the averaged embedding."""
    embeddings: List[np.ndarray] = field(default_factory=list)
    target_frames: int = 5


# ── Registry ─────────────────────────────────────────────────────────────────

class IdentityRegistry:
    """
    Thread-safe (single-process) identity management.

    Usage in pipeline:
        registry = IdentityRegistry(cfg, db)
        registry.load_from_db()

        # each new track_id:
        face_id = registry.resolve(track_id, embedding, centroid, frame_no)

        # each lost track:
        registry.on_track_lost(track_id, embedding, centroid)
    """

    def __init__(self, rec_cfg: dict, reid_cfg: dict):
        # Thresholds
        self._match_threshold: float     = rec_cfg["embedding_threshold"]
        self._dup_threshold: float       = rec_cfg.get("duplicate_guard_threshold", 0.75)
        self._avg_frames: int            = rec_cfg.get("embedding_avg_frames", 5)

        # Ghost config
        self._ghost_ttl: float           = reid_cfg.get("ghost_ttl_seconds", 8)
        self._ghost_max_dist: float      = reid_cfg.get("ghost_max_bbox_dist", 200)
        self._ghost_sim_thresh: float    = reid_cfg.get("ghost_sim_threshold", 0.55)

        # ── Stores ──────────────────────────────────────────────────────
        # face_id → averaged embedding (L2-normalised)
        self._registry: Dict[str, np.ndarray] = {}

        # track_id → CacheEntry  (confirmed identity assignments)
        self._track_cache: Dict[int, CacheEntry] = {}

        # face_id → list of ghost entries (usually 1, but handles duplicates)
        self._ghosts: Dict[str, GhostEntry] = {}

        # track_id → PendingEmbedding  (accumulating before commit)
        self._pending: Dict[int, PendingEmbedding] = {}

        logger.info(
            "IdentityRegistry init | match=%.2f dup=%.2f avg_frames=%d ghost_ttl=%.1fs",
            self._match_threshold, self._dup_threshold,
            self._avg_frames, self._ghost_ttl,
        )

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def load_from_db(self, face_docs: List[dict]):
        """Populate registry from MongoDB face documents at startup."""
        for doc in face_docs:
            fid = doc["face_id"]
            emb = np.array(doc["embedding"], dtype=np.float32)
            emb = self._normalise(emb)
            self._registry[fid] = emb
        logger.info("Registry loaded: %d known identities.", len(self._registry))

    # ── Main resolution entry point ───────────────────────────────────────────

    def needs_recognition(self, track_id: int) -> bool:
        """True if this track_id has no confirmed identity yet."""
        return track_id not in self._track_cache

    def accumulate_embedding(
        self, track_id: int, embedding: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Accumulate embedding frames for a track.
        Returns the averaged embedding once enough frames are collected,
        else None (caller should wait for more frames).
        """
        pending = self._pending.setdefault(
            track_id, PendingEmbedding(target_frames=self._avg_frames)
        )
        pending.embeddings.append(self._normalise(embedding))

        if len(pending.embeddings) >= pending.target_frames:
            averaged = self._average_embeddings(pending.embeddings)
            del self._pending[track_id]
            return averaged

        # Not ready yet; but if this is the very first frame, return it
        # immediately so we don't miss fast-moving people.
        if len(pending.embeddings) == 1:
            return self._normalise(embedding)

        return None

    def resolve(
        self,
        track_id: int,
        embedding: np.ndarray,
        centroid: np.ndarray,
        frame_no: int,
        event_logger=None,              # XAI: optional EventLogger for reasoning logs
        match_threshold: float = None,  # XAI: expose for log context
    ) -> Tuple[Optional[str], float, bool]:
        """
        Determine the face_id for a track given a (possibly averaged) embedding.

        Resolution order:
          1. Track cache hit          → return immediately (no scan)
          2. Ghost buffer match       → re-assign same face_id (crossing fix)
          3. Registry scan            → re-use if sim ≥ match_threshold
          4. Duplicate guard          → abort registration if near-dup found
          5. Register new face_id

        Returns:
            (face_id, confidence, is_new_registration)
        """
        # 1. Cache hit — this track_id already has a confirmed identity
        if track_id in self._track_cache:
            ce = self._track_cache[track_id]
            return ce.face_id, ce.confidence, False

        embedding = self._normalise(embedding)
        thresh    = match_threshold or self._match_threshold
        self._expire_ghosts()

        # 2. Ghost match — most important for crossing re-ID
        ghost_id, ghost_sim = self._match_ghosts(embedding, centroid)
        if ghost_id is not None:
            reason = (
                f"ghost_similarity={ghost_sim:.3f} > "
                f"ghost_threshold={self._ghost_sim_thresh:.2f} "
                f"[crossing re-ID]"
            )
            logger.info("Ghost re-ID: track_id=%d → face_id=%s (%s)", track_id, ghost_id, reason)
            if event_logger:
                event_logger.log_xai("Re-ID (ghost)", ghost_id, reason, track_id=track_id)
            self._commit_cache(track_id, ghost_id, ghost_sim, frame_no)
            self._ghosts.pop(ghost_id, None)
            return ghost_id, ghost_sim, False

        # 3. Global registry scan
        reg_id, reg_sim = self._match_registry(embedding, threshold=thresh)
        if reg_id is not None:
            reason = (
                f"similarity={reg_sim:.3f} > threshold={thresh:.2f}"
            )
            logger.info("Registry match: track_id=%d → face_id=%s (%s)", track_id, reg_id, reason)
            if event_logger:
                event_logger.log_xai("Matched", reg_id, reason, track_id=track_id)
            self._commit_cache(track_id, reg_id, reg_sim, frame_no)
            return reg_id, reg_sim, False

        # 4. Duplicate guard — scan at stricter threshold before registering
        dup_id, dup_sim = self._match_registry(embedding, threshold=self._dup_threshold)
        if dup_id is not None:
            reason = (
                f"similarity={dup_sim:.3f} >= dup_threshold={self._dup_threshold:.2f} "
                f"[duplicate guard, reusing existing ID]"
            )
            logger.warning(
                "Duplicate guard triggered: track_id=%d → face_id=%s (%s)",
                track_id, dup_id, reason,
            )
            if event_logger:
                event_logger.log_xai("Duplicate-guard", dup_id, reason, level="warning", track_id=track_id)
            self._commit_cache(track_id, dup_id, dup_sim, frame_no)
            return dup_id, dup_sim, False

        # 5. Genuinely new identity
        new_id = "face_" + uuid.uuid4().hex[:12]
        self._registry[new_id] = embedding
        self._commit_cache(track_id, new_id, 1.0, frame_no)
        # best registry sim was reg_sim (might be negative if registry empty)
        _, best_sim = self._match_registry(embedding, threshold=-1.0)
        reason = (
            f"best_similarity={best_sim:.3f} < threshold={thresh:.2f} "
            f"[no match found — new identity created]"
        )
        logger.info("New identity: track_id=%d → face_id=%s  %s", track_id, new_id, reason)
        if event_logger:
            event_logger.log_xai("Registered", new_id, reason, track_id=track_id)
        return new_id, 1.0, True

    def update_confidence(
        self, track_id: int, embedding: np.ndarray, frame_no: int
    ) -> Optional[str]:
        """
        Called when a better-quality embedding arrives for an already-resolved
        track.  Updates the registry embedding and cache confidence only if the
        new similarity is higher than stored.  Returns face_id if updated.
        """
        if track_id not in self._track_cache:
            return None

        embedding = self._normalise(embedding)
        ce = self._track_cache[track_id]
        stored_emb = self._registry.get(ce.face_id)
        if stored_emb is None:
            return None

        new_sim = float(np.dot(embedding, stored_emb))
        if new_sim > ce.confidence:
            # Update averaged embedding in registry
            averaged = self._average_embeddings([stored_emb, embedding])
            self._registry[ce.face_id] = averaged
            self._track_cache[track_id] = CacheEntry(
                face_id=ce.face_id,
                confidence=new_sim,
                frame_assigned=frame_no,
            )
            logger.debug(
                "Confidence updated: face_id=%s %.3f → %.3f",
                ce.face_id, ce.confidence, new_sim,
            )
            return ce.face_id
        return None

    # ── Track lifecycle ───────────────────────────────────────────────────────

    def on_track_lost(
        self,
        track_id: int,
        embedding: Optional[np.ndarray],
        centroid: np.ndarray,
    ):
        """
        Move a lost track's identity into the ghost buffer.
        Called by pipeline when tracker marks a track as permanently lost.
        """
        face_id = None
        if track_id in self._track_cache:
            face_id = self._track_cache[track_id].face_id
            del self._track_cache[track_id]

        self._pending.pop(track_id, None)

        if face_id and embedding is not None:
            ghost = GhostEntry(
                face_id=face_id,
                embedding=self._normalise(embedding),
                centroid=centroid,
                lost_at=time.monotonic(),
                ttl=self._ghost_ttl,
            )
            self._ghosts[face_id] = ghost
            logger.debug(
                "Ghost added: face_id=%s | TTL=%.1fs", face_id, self._ghost_ttl
            )

    def get_face_id(self, track_id: int) -> Optional[str]:
        """Return confirmed face_id for a track, or None."""
        ce = self._track_cache.get(track_id)
        return ce.face_id if ce else None

    def get_all_face_ids(self) -> List[str]:
        return list(self._registry.keys())

    def registry_size(self) -> int:
        return len(self._registry)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _match_registry(
        self, query: np.ndarray, threshold: float
    ) -> Tuple[Optional[str], float]:
        """Scan full registry, return best match above threshold."""
        best_id, best_sim = None, -1.0
        for fid, stored in self._registry.items():
            sim = float(np.dot(query, stored))
            if sim > best_sim:
                best_sim, best_id = sim, fid
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    def _match_ghosts(
        self, query: np.ndarray, centroid: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Match against active ghosts using combined similarity + spatial proximity.
        Both gates must pass.
        """
        best_id, best_sim = None, -1.0
        for fid, ghost in self._ghosts.items():
            sim = float(np.dot(query, ghost.embedding))
            if sim < self._ghost_sim_thresh:
                continue
            dist = float(np.linalg.norm(centroid - ghost.centroid))
            if dist > self._ghost_max_dist:
                continue
            if sim > best_sim:
                best_sim, best_id = sim, fid
        return best_id, best_sim

    def _expire_ghosts(self):
        now = time.monotonic()
        expired = [
            fid for fid, g in self._ghosts.items()
            if (now - g.lost_at) > g.ttl
        ]
        for fid in expired:
            del self._ghosts[fid]
            logger.debug("Ghost expired: face_id=%s", fid)

    def _commit_cache(
        self, track_id: int, face_id: str, confidence: float, frame_no: int
    ):
        self._track_cache[track_id] = CacheEntry(
            face_id=face_id,
            confidence=confidence,
            frame_assigned=frame_no,
        )

    @staticmethod
    def _normalise(emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-6 else emb

    @staticmethod
    def _average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(embeddings, axis=0)
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-6 else mean