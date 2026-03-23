"""
scripts/view_stats.py
Quick CLI tool to query and display tracker statistics from MongoDB.

Usage:
    python scripts/view_stats.py
    python scripts/view_stats.py --date 2024-01-15
    python scripts/view_stats.py --face face_abc123def456
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.mongo_manager import MongoManager
from utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser(description="Face Tracker Stats Viewer")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD)")
    parser.add_argument("--face", default=None, help="Query events for a specific face_id")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_cfg = cfg["database"]
    db = MongoManager(
        uri=db_cfg["uri"],
        db_name=db_cfg["name"],
        collections=db_cfg["collections"],
    )

    try:
        if args.face:
            events = db.get_events_for_face(args.face)
            data = {"face_id": args.face, "events": events}
        else:
            stats = db.get_stats()
            if args.date:
                stats["visitors_on_date"] = db.get_unique_visitor_count(args.date)
            data = stats

        if args.json:
            # convert datetime objects for JSON serialisation
            print(json.dumps(data, default=str, indent=2))
        else:
            print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Face Tracker — Statistics")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"  {k}: [{len(v)} records]")
                    for item in v[:10]:   # show first 10
                        print(f"    {item}")
                else:
                    print(f"  {k}: {v}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
