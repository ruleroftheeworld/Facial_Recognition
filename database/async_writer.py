"""
database/async_writer.py

OPTIMIZATION 6 — Asynchronous MongoDB writes.

All DB writes (register_face, log_event, update_face_last_seen) are pushed
onto a thread-safe queue and executed in a single background daemon thread.
The main pipeline thread never blocks on I/O.

Design decisions:
  - Single writer thread avoids concurrent write contention on MongoDB.
  - Queue is bounded (max_queue_size) to apply back-pressure if the writer
    falls behind; dropped writes are logged as warnings.
  - On shutdown, drain() flushes the queue with a configurable timeout so
    no events are lost on clean exit.
  - Thread safety: queue.Queue is inherently thread-safe; no extra locks needed.
"""

import logging
import queue
import threading
import time
from typing import Any, Callable, Tuple

logger = logging.getLogger(__name__)

# Internal sentinel to signal the writer thread to stop
_STOP = object()


class AsyncDBWriter:
    """
    Background thread that drains a write queue and executes DB calls.

    Usage:
        writer = AsyncDBWriter(max_queue_size=500)
        writer.start()

        # From pipeline thread (non-blocking):
        writer.submit(db.register_face, face_id=..., embedding=...)

        # On shutdown:
        writer.drain(timeout=5.0)
        writer.stop()
    """

    def __init__(self, max_queue_size: int = 500):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread = threading.Thread(
            target=self._worker, daemon=True, name="db-writer"
        )
        self._running = False
        self._writes_completed = 0
        self._writes_dropped   = 0

    def start(self):
        self._running = True
        self._thread.start()
        logger.info("AsyncDBWriter started (daemon thread).")

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> bool:
        """
        Enqueue a DB write.  Non-blocking; returns False if queue is full.
        OPTIMIZATION 6: main thread returns immediately after this call.
        """
        try:
            self._queue.put_nowait((fn, args, kwargs))
            return True
        except queue.Full:
            self._writes_dropped += 1
            logger.warning(
                "AsyncDBWriter queue full — write dropped. "
                "Consider increasing async_queue_size. "
                "Total dropped: %d", self._writes_dropped
            )
            return False

    def drain(self, timeout: float = 5.0):
        """Block until queue is empty or timeout expires (used on shutdown)."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty():
            if time.monotonic() > deadline:
                remaining = self._queue.qsize()
                logger.warning(
                    "AsyncDBWriter drain timeout — %d writes still pending.", remaining
                )
                return
            time.sleep(0.05)
        logger.info("AsyncDBWriter drained. Total writes: %d", self._writes_completed)

    def stop(self):
        """Signal the worker thread to exit after draining."""
        self._queue.put(_STOP)
        self._thread.join(timeout=3.0)
        self._running = False
        logger.info(
            "AsyncDBWriter stopped. completed=%d dropped=%d",
            self._writes_completed, self._writes_dropped,
        )

    # ------------------------------------------------------------------ #
    #  Worker                                                              #
    # ------------------------------------------------------------------ #

    def _worker(self):
        """Daemon thread: dequeue and execute writes sequentially."""
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
                self._writes_completed += 1
            except Exception as exc:
                logger.error("AsyncDBWriter write failed: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "writes_completed": self._writes_completed,
            "writes_dropped":   self._writes_dropped,
            "queue_depth":      self.queue_depth,
        }