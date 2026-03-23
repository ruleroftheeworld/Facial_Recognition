"""
scripts/view_stats.py

CLI analytics dashboard — runs against live MongoDB data.

Usage:
    python scripts/view_stats.py                 # today's summary
    python scripts/view_stats.py --date 2025-06-10
    python scripts/view_stats.py --alerts        # show recent alerts
    python scripts/view_stats.py --top-dwellers  # longest dwell times
    python scripts/view_stats.py --returning     # returning visitor breakdown
"""

import argparse
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
import json


def get_db():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    return client["face_tracker_db"]


def print_banner(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def summary(db, date: str):
    print_banner(f"Daily Summary — {date}")

    # Visitor counts
    vis = db.visitors.find_one({"date": date}) or {}
    print(f"  Unique visitors today : {vis.get('unique_visitors', 0)}")

    total_faces = db.faces.count_documents({})
    print(f"  Total registered faces: {total_faces}")

    # Event breakdown
    pipeline = [{"$group": {"_id": "$event_type", "count": {"$sum": 1}}}]
    event_counts = {d["_id"]: d["count"] for d in db.events.aggregate(pipeline)}
    print(f"  Entry events          : {event_counts.get('entry', 0)}")
    print(f"  Exit events           : {event_counts.get('exit', 0)}")

    # Dwell category breakdown
    print_banner("Dwell Engagement Breakdown")
    cat_pipeline = [
        {"$match": {"event_type": "exit", "dwell_time": {"$exists": True}}},
        {"$group": {"_id": "$category", "count": {"$sum": 1},
                    "avg_dwell": {"$avg": "$dwell_time"}}},
        {"$sort": {"count": -1}},
    ]
    for row in db.events.aggregate(cat_pipeline):
        cat = row["_id"] or "unknown"
        print(f"  {cat:<20}  count={row['count']}  avg_dwell={row['avg_dwell']:.1f}s")

    # Visitor tags
    print_banner("Visitor Tags")
    tag_pipeline = [{"$group": {"_id": "$tag", "count": {"$sum": 1}}}]
    for row in db.faces.aggregate(tag_pipeline):
        print(f"  {str(row['_id']):<15}  {row['count']}")


def show_alerts(db, limit=20):
    print_banner(f"Recent Alerts (last {limit})")
    for alert in db.alerts.find().sort("timestamp", DESCENDING).limit(limit):
        ts  = alert.get("timestamp", "?")
        fid = alert.get("face_id", "?")[-12:]
        atype = alert.get("alert_type", "?")
        reason = alert.get("reason", "")
        print(f"  [{ts}]  {fid}  type={atype}")
        print(f"          {reason}")


def top_dwellers(db, limit=10):
    print_banner(f"Top {limit} Longest Dwellers")
    pipeline = [
        {"$match": {"event_type": "exit", "dwell_time": {"$exists": True}}},
        {"$sort": {"dwell_time": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "face_id": 1, "dwell_time": 1,
                      "category": 1, "timestamp": 1}},
    ]
    for i, row in enumerate(db.events.aggregate(pipeline), 1):
        fid = (row.get("face_id") or "?")[-12:]
        print(
            f"  {i:>2}. face={fid}  dwell={row.get('dwell_time', 0):.1f}s"
            f"  cat={row.get('category', '?')}  @ {row.get('timestamp', '?')}"
        )


def returning_visitors(db, limit=15):
    print_banner(f"Top Returning Visitors (visit_count desc)")
    for doc in (
        db.faces.find(
            {"visit_count": {"$gt": 1}},
            {"face_id": 1, "visit_count": 1, "tag": 1,
             "avg_dwell_time": 1, "total_dwell_time": 1, "first_seen": 1, "_id": 0}
        ).sort("visit_count", DESCENDING).limit(limit)
    ):
        fid   = (doc.get("face_id") or "?")[-12:]
        vc    = doc.get("visit_count", 0)
        tag   = doc.get("tag", "?")
        avg_d = doc.get("avg_dwell_time", 0)
        tot_d = doc.get("total_dwell_time", 0)
        print(
            f"  face={fid}  visits={vc}  tag={tag:<12}"
            f"  avg_dwell={avg_d:.1f}s  total_dwell={tot_d:.1f}s"
        )


def main():
    parser = argparse.ArgumentParser(description="Face Tracker Analytics Dashboard")
    parser.add_argument("--date", default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--alerts",      action="store_true")
    parser.add_argument("--top-dwellers",action="store_true")
    parser.add_argument("--returning",   action="store_true")
    args = parser.parse_args()

    db = get_db()

    if args.alerts:
        show_alerts(db)
    elif args.top_dwellers:
        top_dwellers(db)
    elif args.returning:
        returning_visitors(db)
    else:
        summary(db, args.date)
        show_alerts(db, limit=5)

    print()


if __name__ == "__main__":
    main()
