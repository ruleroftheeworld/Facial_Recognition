// scripts/mongo_init.js
// Runs on first container start to create collections and indexes.

db = db.getSiblingDB("face_tracker_db");

db.createCollection("faces");
db.createCollection("events");
db.createCollection("visitors");

db.faces.createIndex({ face_id: 1 }, { unique: true });
db.faces.createIndex({ registered_at: -1 });

db.events.createIndex({ face_id: 1 });
db.events.createIndex({ timestamp: -1 });
db.events.createIndex({ event_type: 1 });

db.visitors.createIndex({ date: 1 }, { unique: true });

print("face_tracker_db initialised.");
