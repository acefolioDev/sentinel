import sqlite3
from contextlib import contextmanager
from .config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS cameras (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, location TEXT NOT NULL,
 timezone TEXT NOT NULL DEFAULT 'UTC', clock_offset_ms INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
 id TEXT PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(id), filename TEXT NOT NULL,
 storage_key TEXT NOT NULL, sha256 TEXT NOT NULL, duration_ms INTEGER NOT NULL DEFAULT 0,
 source_started_at_utc TEXT, width INTEGER, height INTEGER, status TEXT NOT NULL,
 frame_count INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id), stage TEXT NOT NULL,
 status TEXT NOT NULL, processed INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
 elapsed_ms INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
 id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id), pts_ms INTEGER NOT NULL,
 thumbnail_key TEXT NOT NULL, embedding TEXT NOT NULL, model_id TEXT NOT NULL,
 width INTEGER NOT NULL, height INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_frames_video_pts ON frames(video_id, pts_ms);
CREATE TABLE IF NOT EXISTS reference_images (
 id TEXT PRIMARY KEY, filename TEXT NOT NULL, storage_key TEXT NOT NULL, crop_key TEXT NOT NULL,
 subject_type TEXT NOT NULL, owner_actor TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS searches (
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, query TEXT, reference_id TEXT,
 owner_actor TEXT NOT NULL,
 status TEXT NOT NULL, summary TEXT NOT NULL, limitations TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
 id TEXT PRIMARY KEY, search_id TEXT NOT NULL REFERENCES searches(id), frame_id TEXT NOT NULL REFERENCES frames(id),
 video_id TEXT NOT NULL REFERENCES videos(id), camera_id TEXT NOT NULL REFERENCES cameras(id),
 start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, similarity_raw REAL NOT NULL,
 review_status TEXT NOT NULL DEFAULT 'unreviewed', review_note TEXT NOT NULL DEFAULT '',
 reviewed_at TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_matches_search ON matches(search_id);
CREATE TABLE IF NOT EXISTS reports (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, notes TEXT NOT NULL, storage_key TEXT NOT NULL,
 evidence_ids TEXT NOT NULL, owner_actor TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_logs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, resource_id TEXT,
 actor TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL
);
"""

def connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    with connect() as db:
        db.executescript(SCHEMA)

@contextmanager
def transaction():
    db = connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def rows(sql, params=()):
    with connect() as db:
        return [dict(r) for r in db.execute(sql, params).fetchall()]

def row(sql, params=()):
    with connect() as db:
        value = db.execute(sql, params).fetchone()
        return dict(value) if value else None

def audit(action, resource_id=None, details="", actor="local-admin"):
    from datetime import datetime, timezone
    with transaction() as db:
        db.execute("INSERT INTO audit_logs(action,resource_id,actor,details,created_at) VALUES(?,?,?,?,?)",
                   (action,resource_id,actor,details,datetime.now(timezone.utc).isoformat()))
