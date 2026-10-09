"""PostgreSQL metadata and pgvector storage. Search ranking happens in SQL."""
from contextlib import contextmanager
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from .config import DATABASE_URL, EMBEDDING_DIMENSIONS

SCHEMA = [
"""CREATE TABLE IF NOT EXISTS cameras (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, location TEXT NOT NULL,
 timezone TEXT NOT NULL DEFAULT 'UTC', clock_offset_ms INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL)""",
"""CREATE TABLE IF NOT EXISTS videos (
 id TEXT PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(id), filename TEXT NOT NULL,
 storage_key TEXT NOT NULL, sha256 TEXT NOT NULL, duration_ms INTEGER NOT NULL DEFAULT 0,
 source_started_at_utc TEXT, width INTEGER, height INTEGER, status TEXT NOT NULL,
 frame_count INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL)""",
"""CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id), stage TEXT NOT NULL,
 status TEXT NOT NULL, processed INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
 elapsed_ms INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
f"""CREATE TABLE IF NOT EXISTS frames (
 id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id), pts_ms INTEGER NOT NULL,
 thumbnail_key TEXT NOT NULL, embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL, model_id TEXT NOT NULL,
 width INTEGER NOT NULL, height INTEGER NOT NULL)""",
"CREATE INDEX IF NOT EXISTS ix_frames_video_pts ON frames(video_id, pts_ms)",
"CREATE INDEX IF NOT EXISTS ix_frames_embedding_hnsw ON frames USING hnsw (embedding vector_cosine_ops)",
"""CREATE TABLE IF NOT EXISTS reference_images (
 id TEXT PRIMARY KEY, filename TEXT NOT NULL, storage_key TEXT NOT NULL, crop_key TEXT NOT NULL,
 subject_type TEXT NOT NULL, owner_actor TEXT NOT NULL, created_at TEXT NOT NULL)""",
"""CREATE TABLE IF NOT EXISTS searches (
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, query TEXT, reference_id TEXT,
 owner_actor TEXT NOT NULL, video_id TEXT, status TEXT NOT NULL, summary TEXT NOT NULL,
 limitations TEXT NOT NULL, answer_count INTEGER, answer_model TEXT, created_at TEXT NOT NULL)""",
"ALTER TABLE searches ADD COLUMN IF NOT EXISTS answer_count INTEGER",
"ALTER TABLE searches ADD COLUMN IF NOT EXISTS answer_model TEXT",
"ALTER TABLE searches ADD COLUMN IF NOT EXISTS video_id TEXT",
"""CREATE TABLE IF NOT EXISTS matches (
 id TEXT PRIMARY KEY, search_id TEXT NOT NULL REFERENCES searches(id), frame_id TEXT NOT NULL REFERENCES frames(id),
 video_id TEXT NOT NULL REFERENCES videos(id), camera_id TEXT NOT NULL REFERENCES cameras(id),
 start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, similarity_raw DOUBLE PRECISION NOT NULL,
 observed_start_ms INTEGER, observed_end_ms INTEGER,
 review_status TEXT NOT NULL DEFAULT 'unreviewed', review_note TEXT NOT NULL DEFAULT '',
 reviewed_at TEXT, observation TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)""",
"ALTER TABLE matches ADD COLUMN IF NOT EXISTS observation TEXT NOT NULL DEFAULT ''",
"ALTER TABLE matches ADD COLUMN IF NOT EXISTS observed_start_ms INTEGER",
"ALTER TABLE matches ADD COLUMN IF NOT EXISTS observed_end_ms INTEGER",
"CREATE INDEX IF NOT EXISTS ix_matches_search ON matches(search_id)",
"""CREATE TABLE IF NOT EXISTS match_detections (
 match_id TEXT PRIMARY KEY REFERENCES matches(id) ON DELETE CASCADE,
 payload JSONB NOT NULL, created_at TEXT NOT NULL)""",
"UPDATE searches s SET video_id=(SELECT m.video_id FROM matches m WHERE m.search_id=s.id LIMIT 1) WHERE s.kind='qa' AND s.video_id IS NULL",
"""CREATE TABLE IF NOT EXISTS reports (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, notes TEXT NOT NULL, storage_key TEXT NOT NULL,
 evidence_ids TEXT NOT NULL, owner_actor TEXT NOT NULL, created_at TEXT NOT NULL)""",
"""CREATE TABLE IF NOT EXISTS audit_logs (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, action TEXT NOT NULL,
 resource_id TEXT, actor TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL)""",
]

class Database:
    def __init__(self, register=True):
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        if register:
            register_vector(self.conn)
    def execute(self, sql, params=()):
        # Existing route SQL uses qmark placeholders; psycopg uses %s.
        return self.conn.execute(sql.replace('?', '%s'), params)
    def commit(self):
        self.conn.commit()
    def rollback(self):
        self.conn.rollback()
    def close(self):
        self.conn.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type: self.rollback()
        else: self.commit()
        self.close()

def connect(register=True):
    return Database(register=register)

def init_db():
    with connect(register=False) as db:
        db.execute('CREATE EXTENSION IF NOT EXISTS vector')
    with connect() as db:
        for statement in SCHEMA:
            db.execute(statement)

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
        return list(db.execute(sql, params).fetchall())

def row(sql, params=()):
    with connect() as db:
        return db.execute(sql, params).fetchone()

def audit(action, resource_id=None, details='', actor='local-admin'):
    with transaction() as db:
        db.execute('INSERT INTO audit_logs(action,resource_id,actor,details,created_at) VALUES(?,?,?,?,?)',
                   (action,resource_id,actor,details,datetime.now(timezone.utc).isoformat()))
