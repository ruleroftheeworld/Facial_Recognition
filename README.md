# 🎯 Intelligent Face Tracker with Auto-Registration & Visitor Intelligence

> **This project is a part of a hackathon run by https://katomaran.com**

---

## 🚀 Overview

An end-to-end **AI-powered real-time face tracking and visitor intelligence system** that goes beyond basic detection by providing:

* 🧠 **Visitor behavior analytics**
* ⏱️ **Dwell-time insights**
* 🔁 **Returning visitor tracking**
* 🚨 **Loitering & suspicious activity detection**
* 🔍 **Explainable AI (XAI) decision logging**

The system processes video streams (file or RTSP), detects and tracks faces, automatically registers new identities, and logs structured events into MongoDB — all while maintaining real-time performance.

---

## 🏗️ Architecture

```
Video Source (MP4 / RTSP)
        │
        ▼
FaceTrackerPipeline (core/pipeline.py)
        │
 ┌────────────┬──────────────┬──────────────┐
 │ Detector   │ Tracker      │ Recognizer   │
 │ YOLOv8     │ Centroid     │ InsightFace  │
 └─────┬──────┴──────┬───────┴──────┬──────┘
       │             │              │
       └─────────────▼──────────────┘
               Match / Register
                       │
                       ▼
            🔥 Analytics Layer
     (Dwell | Visitors | Alerts | XAI)
                       │
      ┌───────────────┼────────────────┐
      │               │                │
 EventLogger     MongoDB         Annotated Frame
 (logs/images)   (faces/events/  (display/output)
                 visitors/alerts)
                       │
                       ▼
              Streamlit Dashboard
```

---

## 🔥 Features

### 🎯 Core System

| Module          | Feature                                |
| --------------- | -------------------------------------- |
| Detection       | YOLOv8 real-time face detection        |
| Recognition     | InsightFace ArcFace embeddings (512-d) |
| Tracking        | Centroid / ByteTrack-style tracking    |
| Auto-Register   | New faces assigned unique IDs          |
| Entry/Exit Logs | One entry & exit per visit             |
| Storage         | MongoDB + structured image logs        |
| RTSP Support    | Live camera streaming                  |
| Dashboard       | Streamlit analytics UI                 |

---

### 🧠 Advanced Intelligence (Key Differentiator)

| Feature                   | Description                                        |
| ------------------------- | -------------------------------------------------- |
| ⏱️ Dwell Time Analytics   | Tracks entry/exit time and computes dwell time     |
| 🔁 Returning Visitors     | Maintains visit_count, tags users (new / frequent) |
| 🚨 Loitering Detection    | Flags long-duration presence                       |
| ⚠️ Suspicious Behavior    | Detects abnormal visit patterns                    |
| 🔍 Explainable AI (XAI)   | Logs WHY each decision was made                    |
| 📊 Visitor Categorization | passerby / engaged / highly engaged                |
| 📢 Alerts System          | Stores alerts in DB + logs                         |

---

## 🧱 Tech Stack

| Layer       | Technology                   |
| ----------- | ---------------------------- |
| Detection   | YOLOv8                       |
| Recognition | InsightFace (ArcFace)        |
| Tracking    | OpenCV + custom tracker      |
| Backend     | Python                       |
| Database    | MongoDB                      |
| Dashboard   | Streamlit                    |
| Logging     | Python logging + file system |
| DevOps      | Docker Compose               |

---

## 📂 Project Structure

```
face_tracker/
├── core/
├── database/
├── analytics/            
├── logging_system/
├── frontend/
├── scripts/
├── logs/
├── config.json
├── main.py
```

---

## ⚙️ Setup Instructions

### 1. Clone & Setup

```bash
git clone <repo>
cd face_tracker
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Start MongoDB

```bash
docker-compose up -d
```

👉 UI: http://localhost:8081

---

### 4. Run Application

```bash
python main.py
```

---

### 5. Dashboard

```bash
python -m streamlit run frontend/dashboard.py
```

👉 http://localhost:8501

---

## ⚙️ Configuration

Add analytics config:

```json
{
  "analytics": {
    "dwell_time_threshold": 20,
    "suspicious_visit_threshold": 5
  }
}
```

---

## 📊 MongoDB Schema

### faces

```json
{
  "face_id": "...",
  "visit_count": 3,
  "avg_dwell_time": 15.2,
  "total_dwell_time": 45.6,
  "tag": "frequent"
}
```

---

### events

```json
{
  "event_type": "exit",
  "dwell_time": 12.4,
  "category": "engaged"
}
```

---

### alerts

```json
{
  "face_id": "...",
  "alert_type": "loitering",
  "reason": "dwell_time > threshold",
  "timestamp": "..."
}
```

---

## 🧪 Sample Logs

### XAI Logs

```
[XAI] Matched Face ID abc123
Reason: similarity=0.82 > threshold=0.60

[XAI] Registered Face ID xyz456
Reason: similarity=0.41 < threshold=0.60
```

---

### Alerts

```
[ALERT] Face ID abc123 flagged: loitering (dwell_time=32s)
```

---

## 📈 Compute Performance

| Mode | FPS       |
| ---- | --------- |
| CPU  | 15–20 FPS |
| GPU  | 30+ FPS   |

Optimizations:

* Frame skip
* GPU acceleration
* Async DB writes
* Face filtering

---

## 🧠 AI Planning Summary

* Designed for real-time constraints
* Prevents duplicate registrations
* Handles occlusion via tracking
* Ensures consistent identity mapping

---

## 📌 Assumptions

* Single camera view
* Faces mostly front-facing
* MongoDB running locally
* Internet required for first model download

---
🎥 Demo Video
https://www.youtube.com/watch?v=1TpCZwFBkP4
---

## 🏆 Final Outcome

This project evolves from a simple face tracker into a:

> 🚀 **Real-Time Visitor Intelligence & Surveillance Platform**

with analytics, explainability, and behavioral insights.

---

**This project is a part of a hackathon run by https://katomaran.com**
