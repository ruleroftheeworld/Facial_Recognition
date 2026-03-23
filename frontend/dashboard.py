"""
frontend/dashboard.py
Optional Streamlit dashboard for real-time monitoring.

Run with:
    streamlit run frontend/dashboard.py
"""

import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

from database.mongo_manager import MongoManager
from utils.helpers import load_config


# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Face Tracker Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  DB connection (cached)                                              #
# ------------------------------------------------------------------ #

@st.cache_resource
def get_db() -> MongoManager:
    cfg = load_config()
    db_cfg = cfg["database"]
    return MongoManager(
        uri=db_cfg["uri"],
        db_name=db_cfg["name"],
        collections=db_cfg["collections"],
    )


db = get_db()

# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #

st.sidebar.title("⚙️ Controls")
refresh_rate = st.sidebar.slider("Auto-refresh (sec)", 1, 30, 5)
date_filter = st.sidebar.date_input("Date", datetime.utcnow().date())

if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# ------------------------------------------------------------------ #
#  Title                                                               #
# ------------------------------------------------------------------ #

st.title("👤 Intelligent Face Tracker — Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ------------------------------------------------------------------ #
#  KPI Cards                                                           #
# ------------------------------------------------------------------ #

stats = db.get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Unique Visitors Today", stats["today_unique_visitors"])
col2.metric("Total Registered Faces", stats["total_registered_faces"])
col3.metric("Total Entries", stats["total_entries"])
col4.metric("Total Exits", stats["total_exits"])

st.divider()

# ------------------------------------------------------------------ #
#  Recent Events Table                                                 #
# ------------------------------------------------------------------ #

st.subheader("📋 Recent Events")

events_col = db.db[db.collection_names["events"]]
recent_events = list(
    events_col.find({}, {"_id": 0, "face_id": 1, "event_type": 1,
                         "timestamp": 1, "image_path": 1})
    .sort("timestamp", -1)
    .limit(50)
)

if recent_events:
    df = pd.DataFrame(recent_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df, use_container_width=True, height=300)
else:
    st.info("No events recorded yet.")

st.divider()

# ------------------------------------------------------------------ #
#  Registered Faces                                                    #
# ------------------------------------------------------------------ #

st.subheader("🗂️ Registered Faces")

faces_col = db.db[db.collection_names["faces"]]
faces = list(
    faces_col.find(
        {},
        {"_id": 0, "face_id": 1, "registered_at": 1,
         "last_seen": 1, "visit_count": 1, "thumbnail_path": 1},
    )
    .sort("registered_at", -1)
    .limit(100)
)

if faces:
    df_faces = pd.DataFrame(faces)
    for col in ["registered_at", "last_seen"]:
        if col in df_faces.columns:
            df_faces[col] = pd.to_datetime(df_faces[col]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    st.dataframe(df_faces, use_container_width=True, height=300)
else:
    st.info("No faces registered yet.")

st.divider()

# ------------------------------------------------------------------ #
#  Daily Visitor Chart                                                 #
# ------------------------------------------------------------------ #

st.subheader("📊 Unique Visitors (Last 7 Days)")

visitors_col = db.db[db.collection_names["visitors"]]
last_7 = []
for i in range(6, -1, -1):
    d = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
    doc = visitors_col.find_one({"date": d})
    last_7.append({"date": d, "visitors": doc["unique_visitors"] if doc else 0})

df_visitors = pd.DataFrame(last_7).set_index("date")
st.bar_chart(df_visitors)

# ------------------------------------------------------------------ #
#  Auto-refresh                                                        #
# ------------------------------------------------------------------ #

time.sleep(refresh_rate)
st.rerun()
