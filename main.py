"""
main.py
Entry point for the Intelligent Face Tracker.

Usage:
    python main.py                        # use config.json defaults
    python main.py --config my_cfg.json   # custom config path
    python main.py --source video.mp4     # override video source
    python main.py --rtsp                 # use RTSP stream from config
"""

import argparse
import logging
import signal
import sys
import time

import cv2

from core.pipeline import FaceTrackerPipeline
from utils.helpers import load_config, setup_logging, open_video_source

# ------------------------------------------------------------------ #
#  Graceful shutdown handler                                           #
# ------------------------------------------------------------------ #

_pipeline: FaceTrackerPipeline = None


def _signal_handler(sig, frame):
    logging.getLogger(__name__).info("Interrupt received – shutting down…")
    if _pipeline:
        _pipeline.shutdown()
    sys.exit(0)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    global _pipeline

    # ---- CLI args ----
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None, help="Override video source path")
    parser.add_argument("--rtsp", action="store_true", help="Use RTSP stream")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    args = parser.parse_args()

    # ---- Config ----
    config = load_config(args.config)
    setup_logging(
        level=config["logging"].get("log_level", "INFO"),
        log_file="logs/app.log",
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Face Tracker starting up")
    logger.info("=" * 60)

    if args.source:
        config["camera"]["source"] = args.source
    if args.rtsp:
        config["camera"]["use_rtsp"] = True
    if args.no_display:
        config["camera"]["display_window"] = False

    # ---- Signals ----
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ---- Pipeline ----
    _pipeline = FaceTrackerPipeline(config)

    # ---- Video source ----
    cap, fps, width, height = open_video_source(config["camera"])

    display = config["camera"].get("display_window", True)
    window_name = config["camera"].get("window_name", "Face Tracker")

    frame_delay = max(1, int(1000 / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    processed = 0
    start_time = time.time()

    logger.info(
        "Processing video: %dx%d @ %.1f FPS | total_frames=%s",
        width, height, fps, total_frames if total_frames > 0 else "∞"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            # ---- Process ----
            annotated = _pipeline.process_frame(frame)
            processed += 1

            # ---- Display ----
            if display:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord("q") or key == 27:   # q or ESC
                    logger.info("User quit.")
                    break
                elif key == ord("s"):
                    # print stats to console
                    print("\n--- Stats ---")
                    for k, v in _pipeline.get_stats().items():
                        print(f"  {k}: {v}")

            # ---- Progress log every 100 frames ----
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Processed %d frames | %.1f FPS | elapsed %ds",
                    processed, current_fps, int(elapsed),
                )

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

        # ---- Final stats ----
        stats = _pipeline.get_stats()
        logger.info("=" * 60)
        logger.info("Session complete. Final stats:")
        for k, v in stats.items():
            logger.info("  %s: %s", k, v)
        logger.info("=" * 60)

        _pipeline.shutdown()


if __name__ == "__main__":
    main()
