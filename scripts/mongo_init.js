// scripts/mongo_init.js
// Run with: mongosh face_tracker_db scripts/mongo_init.js

db = db.getSiblingDB("face_tracker_db");

// ── faces collection ─────────────────────────────────────────────────────────
// New analytics fields added to schema (backwards-compatible):
//   visit_count, first_seen, last_seen, tag (new/occasional/frequent)
//   total_dwell_time, avg_dwell_time
db.createCollection("faces");
db.faces.createIndex({ face_id: 1 }, { unique: true });
db.faces.createIndex({ registered_at: -1 });
db.faces.createIndex({ tag: 1 });
db.faces.createIndex({ visit_count: -1 });
print("✓ faces collection and indexes ready");

// ── events collection ────────────────────────────────────────────────────────
// New analytics fields on exit events:
//   exit_time, dwell_time (seconds), category (passerby/engaged/highly_engaged)
db.createCollection("events");
db.events.createIndex({ face_id: 1 });
db.events.createIndex({ timestamp: -1 });
db.events.createIndex({ event_type: 1 });
db.events.createIndex({ category: 1 });          // NEW – analytics query index
db.events.createIndex({ dwell_time: -1 });       // NEW – sort by dwell
print("✓ events collection and indexes ready");

// ── visitors collection ──────────────────────────────────────────────────────
// Daily counter (existing schema unchanged)
db.createCollection("visitors");
db.visitors.createIndex({ date: 1 }, { unique: true });
print("✓ visitors collection ready");

// ── alerts collection ────────────────────────────────────────────────────────
// NEW – loitering and suspicious behaviour alerts
db.createCollection("alerts");
db.alerts.createIndex({ face_id: 1 });
db.alerts.createIndex({ timestamp: -1 });
db.alerts.createIndex({ alert_type: 1 });
print("✓ alerts collection and indexes ready");

print("\n=== Schema upgrade complete ===");
print("New fields on `faces`:  visit_count, first_seen, last_seen, tag,");
print("                        total_dwell_time, avg_dwell_time");
print("New fields on `events`: exit_time, dwell_time, category  (exit events only)");
print("New collection `alerts`: loitering and suspicious detections");
