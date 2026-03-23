# 🎯 Intelligent Face Tracker with Auto-Registration & Visitor Counting

> **This project is a part of a hackathon run by https://katomaran.com**

---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Configuration (config.json)](#configuration)
8. [Running the Application](#running)
9. [Frontend Dashboard](#dashboard)
10. [Sample Output](#sample-output)
11. [AI Planning Document](#ai-planning)
12. [Compute Load Estimation](#compute-load)
13. [Assumptions](#assumptions)

---

## Overview <a name="overview"></a>

An end-to-end AI-driven face tracking system that:
- Detects faces in a video stream using **YOLOv8**
- Generates facial embeddings via **InsightFace (ArcFace)**
- Tracks faces across frames with a **ByteTrack-style centroid tracker**
- Auto-registers new faces on first detection
- Logs every **entry** and **exit** event with a timestamped image
- Stores all metadata in **MongoDB**
- Maintains an accurate **unique visitor count**

---

## Architecture <a name="architecture"></a>

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Video Source                                │
│              (MP4 file  ──or──  RTSP camera stream)                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ frames
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FaceTrackerPipeline                           │
│                          (core/pipeline.py)                          │
│                                                                      │
│   ┌────────────┐    ┌──────────────┐    ┌─────────────────────┐     │
│   │  Detector  │───▶│   Tracker    │───▶│    Recognizer       │     │
│   │ (YOLOv8)  │    │  (Centroid/  │    │  (InsightFace       │     │
│   │           │    │  ByteTrack)  │    │   ArcFace 512-d)    │     │
│   └────────────┘    └──────┬───────┘    └──────────┬──────────┘     │
│                            │                        │                │
│                    detections + track IDs     embeddings             │
│                            │                        │                │
│                            └───────────┬────────────┘               │
│                                        │                             │
│                              ┌─────────▼──────────┐                 │
│                              │   Match / Register  │                 │
│                              │   (cosine sim)      │                 │
│                              └─────────┬───────────┘                │
│                                        │                             │
│               ┌────────────────────────┼───────────────────┐        │
│               │                        │                   │        │
│        ┌──────▼──────┐        ┌────────▼────────┐  ┌───────▼──────┐│
│        │  EventLogger│        │  MongoManager   │  │  Annotated   ││
│        │ events.log  │        │  (MongoDB)      │  │  Frame Out   ││
│        │ face images │        │  faces/events/  │  │  (display/   ││
│        └─────────────┘        │  visitors cols  │  │   video)     ││
│                               └─────────────────┘  └──────────────┘│
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  Streamlit Dashboard   │
                  │  (frontend/dashboard)  │
                  │  KPIs · Events · Chart │
                  └────────────────────────┘
```

### Data Flow per Frame

```
Frame IN
  │
  ├─[every N frames]──► YOLOv8 Detection ──► BBox list
  │                                               │
  │◄──────────────────────────────────────────────┘
  │
  ├──► Centroid Tracker ──► Active Track list + Lost Track list
  │                               │
  │            [new track_id]     │
  │                 │             │
  │         InsightFace           │
  │         Embedding             │
  │              │                │
  │      Cosine similarity        │
  │      vs. DB cache             │
  │         │        │            │
  │      Match    No Match        │
  │         │        │            │
  │     Update   Register         │
  │     last_seen  + fire         │
  │              entry event      │
  │                               │
  │            [lost track]       │
  │                 │             │
  │            Fire exit event ◄──┘
  │
  ├──► Annotate frame (boxes + labels + visitor count)
  │
Frame OUT
```

---

## Features <a name="features"></a>

| Module | Feature |
|--------|---------|
| Detection | YOLOv8 real-time face detection with configurable confidence + IOU |
| Recognition | InsightFace ArcFace 512-d embeddings, cosine similarity matching |
| Tracking | Centroid tracker with max_disappeared + max_distance params |
| Auto-Register | New faces assigned UUID, stored in MongoDB on first detection |
| Entry/Exit Logging | One entry + one exit event per visit, with timestamp + face crop |
| Image Storage | Structured `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/` |
| Event Log File | `logs/events.log` with rotating handler (10 MB × 5 backups) |
| Unique Visitor Count | Per-day counter in `visitors` MongoDB collection |
| Frame Skip | Configurable detection cadence via `frame_skip` in config.json |
| Dashboard | Streamlit UI: KPIs, recent events table, 7-day visitor bar chart |
| CLI Stats | `scripts/view_stats.py` for quick DB queries |
| RTSP Support | Switch to live camera via `use_rtsp: true` in config |

---

## Tech Stack <a name="tech-stack"></a>

| Layer | Technology |
|-------|-----------|
| Face Detection | YOLOv8 (ultralytics) |
| Face Recognition | InsightFace / ArcFace (buffalo_l model) |
| Tracking | Custom ByteTrack-style centroid tracker (OpenCV + scipy) |
| Backend | Python 3.10+ |
| Database | MongoDB 7.0 (via pymongo) |
| Configuration | `config.json` |
| Logging | Python logging + RotatingFileHandler + local image store |
| Camera | OpenCV VideoCapture (file + RTSP) |
| Frontend | Streamlit (optional) |
| DevOps | Docker Compose (MongoDB + Mongo Express) |

---

## Project Structure <a name="project-structure"></a>

```
face_tracker/
├── main.py                          # Entry point
├── config.json                      # All runtime parameters
├── requirements.txt
├── docker-compose.yml               # MongoDB + Mongo Express
│
├── core/
│   ├── __init__.py
│   ├── detector.py                  # YOLOv8 face detector
│   ├── recognizer.py                # InsightFace ArcFace embeddings
│   ├── tracker.py                   # Centroid / ByteTrack tracker
│   └── pipeline.py                  # Orchestrator
│
├── database/
│   ├── __init__.py
│   └── mongo_manager.py             # MongoDB CRUD + visitor counter
│
├── logging_system/
│   ├── __init__.py
│   └── event_logger.py              # events.log + face image saver
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                   # config loader, logging setup, video open
│
├── frontend/
│   └── dashboard.py                 # Streamlit dashboard
│
├── scripts/
│   ├── mongo_init.js                # MongoDB initialisation script
│   └── view_stats.py                # CLI stats viewer
│
├── logs/                            # Auto-created at runtime
│   ├── events.log
│   ├── app.log
│   ├── entries/
│   │   └── YYYY-MM-DD/
│   │       └── face_<id>_HH-MM-SS-ms.jpg
│   └── exits/
│       └── YYYY-MM-DD/
│           └── face_<id>_HH-MM-SS-ms.jpg
│
└── models/                          # Optional: pre-downloaded model weights
```

---

## Setup Instructions <a name="setup-instructions"></a>

### Prerequisites
- Python 3.10+
- Docker + Docker Compose (for MongoDB)
- CUDA 11.8+ (optional, for GPU acceleration)

### Step 1 — Clone & create virtual environment

```bash
git clone <your-repo-url>
cd face_tracker
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on InsightFace:** On Linux/Mac it installs cleanly.  
> On Windows you may need Visual C++ Build Tools:  
> `pip install insightface onnxruntime`  
> For GPU: `pip install onnxruntime-gpu` instead of `onnxruntime`

### Step 3 — Start MongoDB

```bash
docker-compose up -d
```

This starts MongoDB on `localhost:27017` and Mongo Express UI at `http://localhost:8081`.

### Step 4 — Download sample video

Download the provided video from:  
`https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIfrUBFUJ`

Place it in the project root and update `config.json`:
```json
"camera": {
  "source": "your_video.mp4"
}
```

### Step 5 — Run

```bash
python main.py
```

Press **Q** or **ESC** in the preview window to stop.  
Press **S** to print live stats to console.

---

## Configuration <a name="configuration"></a>

### Sample `config.json`

```json
{
  "detection": {
    "frame_skip": 3,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.4,
    "yolo_model": "yolov8n-face.pt"
  },
  "recognition": {
    "model_name": "buffalo_l",
    "embedding_threshold": 0.45,
    "min_face_size": 40
  },
  "tracking": {
    "max_disappeared": 30,
    "max_distance": 0.5,
    "tracker_type": "bytetrack"
  },
  "database": {
    "uri": "mongodb://localhost:27017",
    "name": "face_tracker_db",
    "collections": {
      "faces": "faces",
      "events": "events",
      "visitors": "visitors"
    }
  },
  "logging": {
    "log_file": "logs/events.log",
    "image_base_dir": "logs",
    "log_level": "INFO"
  },
  "camera": {
    "source": "sample_video.mp4",
    "rtsp_url": "rtsp://username:password@ip:port/stream",
    "use_rtsp": false,
    "display_window": true,
    "window_name": "Face Tracker"
  },
  "performance": {
    "use_gpu": true,
    "num_threads": 4,
    "batch_size": 1
  }
}
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `frame_skip` | Run detection every N+1 frames (0 = every frame). Higher = faster but less accurate. |
| `confidence_threshold` | YOLO minimum detection confidence (0–1) |
| `embedding_threshold` | Cosine similarity threshold for face re-identification (0–1) |
| `max_disappeared` | Frames a track can be missing before it's marked as "exited" |
| `use_rtsp` | `true` to switch from video file to live RTSP stream |

---

## Running the Application <a name="running"></a>

```bash
# Default (uses config.json)
python main.py

# Custom config
python main.py --config my_config.json

# Override video source
python main.py --source /path/to/video.mp4

# Use RTSP stream
python main.py --rtsp

# Headless (no display window)
python main.py --no-display

# View stats after run
python scripts/view_stats.py
python scripts/view_stats.py --date 2024-01-15
python scripts/view_stats.py --face face_abc123def456
python scripts/view_stats.py --json
```

---

## Frontend Dashboard <a name="dashboard"></a>

```bash
streamlit run frontend/dashboard.py
```

Opens at `http://localhost:8501` showing:
- **KPI cards**: unique visitors today, total registered faces, entries, exits
- **Recent events table**: live-updating log
- **Registered faces table**
- **7-day visitor bar chart**
- Auto-refresh slider (1–30 sec)

---

## Sample Output <a name="sample-output"></a>

### events.log (excerpt)

```
2024-01-15 10:23:45 [INFO    ] REGISTER | face_id=face_a1b2c3d4e5f6 | track_id=0
2024-01-15 10:23:45 [INFO    ] IMAGE_SAVED | face_id=face_a1b2c3d4e5f6 | type=entry | path=logs/entries/2024-01-15/face_a1b2c3d4e5f6_10-23-45-123.jpg
2024-01-15 10:23:45 [INFO    ] ENTRY | face_id=face_a1b2c3d4e5f6 | track_id=0 | ts=2024-01-15T10:23:45.123456
2024-01-15 10:23:52 [INFO    ] EXIT  | face_id=face_a1b2c3d4e5f6 | track_id=0 | ts=2024-01-15T10:23:52.456789
2024-01-15 10:24:01 [INFO    ] REGISTER | face_id=face_b2c3d4e5f6a1 | track_id=1
2024-01-15 10:24:01 [INFO    ] ENTRY | face_id=face_b2c3d4e5f6a1 | track_id=1 | ts=2024-01-15T10:24:01.789012
```

### MongoDB — `faces` collection (sample document)

```json
{
  "face_id": "face_a1b2c3d4e5f6",
  "embedding": [0.023, -0.118, 0.204, "...512 values..."],
  "thumbnail_path": "logs/entries/2024-01-15/face_a1b2c3d4e5f6_10-23-45-123.jpg",
  "registered_at": "2024-01-15T10:23:45.123Z",
  "last_seen": "2024-01-15T10:23:52.456Z",
  "visit_count": 1,
  "track_id": 0
}
```

### MongoDB — `events` collection (sample document)

```json
{
  "face_id": "face_a1b2c3d4e5f6",
  "event_type": "entry",
  "image_path": "logs/entries/2024-01-15/face_a1b2c3d4e5f6_10-23-45-123.jpg",
  "timestamp": "2024-01-15T10:23:45.123Z"
}
```

### MongoDB — `visitors` collection (sample document)

```json
{
  "date": "2024-01-15",
  "unique_visitors": 7
}
```

### File system layout after run

```
logs/
├── events.log
├── app.log
├── entries/
│   └── 2024-01-15/
│       ├── face_a1b2c3d4e5f6_10-23-45-123.jpg
│       └── face_b2c3d4e5f6a1_10-24-01-789.jpg
└── exits/
    └── 2024-01-15/
        └── face_a1b2c3d4e5f6_10-23-52-456.jpg
```

---

## AI Planning Document <a name="ai-planning"></a>

### Phase 1 — Planning

**Goal:** Count unique human visitors in a video stream with zero false duplicates.

**Core challenges identified:**
1. Same person re-entering later must not increment count
2. Multiple people in frame simultaneously must all be tracked independently
3. Brief occlusions must not cause false re-registrations
4. System must be resilient to interruptions (crash-safe logging)

**Design decisions:**
- **YOLOv8** over Haar cascades: 10× faster, handles rotations and partial occlusions
- **InsightFace ArcFace** over `face_recognition` library: production-grade 512-d embeddings, 99.7%+ accuracy on LFW benchmark
- **Centroid tracker** with configurable `max_disappeared`: simple, fast, no external deps
- **MongoDB** over SQLite: schema-flexible, easy to query embeddings, scales horizontally
- **frame_skip** parameter: balances CPU load vs. tracking accuracy

### Phase 2 — Feature list

1. YOLO face detection with configurable confidence + IOU
2. InsightFace embedding generation (L2-normalised 512-d vectors)
3. Cosine similarity matching against in-memory face cache
4. Auto-registration of new faces with UUID
5. Centroid-based multi-object tracker
6. Entry event: fired once per unique face on first detection
7. Exit event: fired when track disappears for > max_disappeared frames
8. Cropped face image saved on entry, exit, and registration
9. events.log with rotating handler
10. MongoDB persistence: faces, events, visitors collections
11. Daily unique visitor counter
12. CLI stats viewer
13. Streamlit dashboard (bonus)
14. RTSP stream support

### Phase 3 — Implementation order

```
config.json → MongoManager → EventLogger
    → FaceDetector → FaceRecognizer → CentroidTracker
    → FaceTrackerPipeline → main.py → dashboard.py
```

---

## Compute Load Estimation <a name="compute-load"></a>

### Per-frame breakdown (1080p @ 30 FPS)

| Component | CPU (no GPU) | GPU (CUDA) |
|-----------|-------------|-----------|
| YOLOv8n detection | ~80 ms | ~8 ms |
| InsightFace embedding | ~120 ms | ~12 ms |
| Tracker update | ~2 ms | ~2 ms |
| DB write (async) | ~5 ms | ~5 ms |
| **Total per frame** | **~207 ms (~5 FPS)** | **~27 ms (~37 FPS)** |

### Throughput with `frame_skip = 3`

Detection runs every 4th frame; tracker handles intermediate frames.

| Mode | Effective throughput |
|------|---------------------|
| CPU only | ~15–20 FPS |
| GPU (RTX 3060+) | ~30 FPS (real-time) |

### Memory

| Resource | Estimated Usage |
|----------|----------------|
| RAM | ~800 MB – 1.2 GB |
| VRAM (GPU) | ~1.5 GB |
| MongoDB | ~50 MB per 10,000 faces |
| Disk (images) | ~5–15 KB per event image |

---

## Assumptions <a name="assumptions"></a>

1. **One camera angle** — system designed for a single fixed camera feed.
2. **Front-facing faces** — ArcFace accuracy drops for profiles > 45°; side-view faces may be missed.
3. **Minimum face size** — faces smaller than 40×40 px are skipped (configurable via `min_face_size`).
4. **Unique visitor = unique face** — two different people will never share a face ID.
5. **Re-entry handling** — if the same person leaves and re-enters, the embedding cache matches them to the existing face_id; only one entry/exit event pair fires per "visit" (track lifetime), but visit_count increments in the DB.
6. **MongoDB running locally** — default URI is `mongodb://localhost:27017`; update config for remote/Atlas.
7. **Video file for dev, RTSP for production** — switch via `use_rtsp` flag in config.
8. **No authentication** required on MongoDB for local dev.
9. **Python 3.10+** required for full type hint syntax compatibility.
10. **InsightFace model downloads automatically** on first run (~200 MB, requires internet).

---

## Demo Video

> 🎬 **[Watch Demo on YouTube / Loom]** — *(link to be added after recording)*

---

*This project is a part of a hackathon run by https://katomaran.com*
