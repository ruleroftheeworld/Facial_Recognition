"""
utils/performance_monitor.py

OPTIMIZATION 8 — FPS calculation and bottleneck profiling.

Tracks per-stage timing so you can see exactly where time is spent:
  detection / recognition / tracking / db_write / annotation

Usage:
    mon = PerformanceMonitor(log_interval=100)
    mon.tick()                                   # call at start of every frame
    with mon.measure("detection"):
        detections = detector.detect(frame)
    with mon.measure("recognition"):
        ...
    mon.maybe_log()                              # prints stats every N frames
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Lightweight per-stage timer + FPS calculator."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self._frame_count  = 0
        self._session_start = time.monotonic()
        self._last_log_time = time.monotonic()
        self._last_log_frame = 0

        # Accumulated stage times (seconds) since last log
        self._stage_totals: Dict[str, float] = defaultdict(float)
        self._stage_counts: Dict[str, int]   = defaultdict(int)
        self._stage_start: Dict[str, float]  = {}

    def tick(self):
        """Call once at the start of each frame."""
        self._frame_count += 1

    @contextmanager
    def measure(self, stage: str):
        """Context manager to time a pipeline stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._stage_totals[stage] += elapsed
            self._stage_counts[stage] += 1

    def maybe_log(self):
        """Log FPS and per-stage breakdown every log_interval frames."""
        if self._frame_count % self.log_interval != 0:
            return

        now     = time.monotonic()
        elapsed = now - self._last_log_time
        frames  = self._frame_count - self._last_log_frame

        fps = frames / elapsed if elapsed > 0 else 0.0
        overall_fps = self._frame_count / (now - self._session_start)

        lines = [
            f"PERF | frame={self._frame_count} | "
            f"fps={fps:.1f} | avg_fps={overall_fps:.1f}"
        ]

        # Per-stage averages
        for stage, total in sorted(self._stage_totals.items()):
            count = self._stage_counts[stage]
            if count == 0:
                continue
            avg_ms = (total / count) * 1000
            lines.append(f"  {stage:<20} avg={avg_ms:6.1f} ms")

        # Reset interval accumulators
        self._last_log_time  = now
        self._last_log_frame = self._frame_count
        self._stage_totals.clear()
        self._stage_counts.clear()

        report = "\n".join(lines)
        logger.info(report)
        print(report)    # also print to stdout for quick visibility

    @property
    def current_fps(self) -> float:
        elapsed = time.monotonic() - self._session_start
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count