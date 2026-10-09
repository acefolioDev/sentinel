import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import av
from PIL import Image
from .config import DATA_DIR, FRAME_INTERVAL_SEC, MAX_VIDEO_SECONDS, MODEL_ID
from .db import row, transaction
from .vision import encode_images

_running = set()
_running_lock = threading.Lock()
_worker_slot = threading.Semaphore(1)

def now():
    return datetime.now(timezone.utc).isoformat()

def probe(path: Path):
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == 'video'), None)
        if stream is None:
            raise ValueError('No video stream found')
        duration = (float(stream.duration * stream.time_base) if stream.duration is not None else
                    float(container.duration / av.time_base) if container.duration else 0)
        if duration <= 0:
            raise ValueError('Video duration is unavailable')
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(f'Video exceeds the {MAX_VIDEO_SECONDS // 60}-minute local limit')
        return {'duration_ms': int(duration * 1000), 'width': stream.width, 'height': stream.height}

def start_job(job_id: str):
    with _running_lock:
        if job_id in _running:
            return
        _running.add(job_id)
    threading.Thread(target=_queued_run, args=(job_id,), daemon=True, name=f'index-{job_id[:8]}').start()

def _queued_run(job_id):
    with _worker_slot:
        _run(job_id)

def _run(job_id):
    started = time.monotonic()
    job = row('SELECT * FROM jobs WHERE id=?', (job_id,))
    if not job:
        return
    video = row('SELECT * FROM videos WHERE id=?', (job['video_id'],))
    try:
        with transaction() as db:
            db.execute("UPDATE jobs SET status='processing',stage='decoding',error=NULL,updated_at=? WHERE id=?", (now(),job_id))
            db.execute("UPDATE videos SET status='processing',error=NULL WHERE id=?", (video['id'],))
            db.execute('DELETE FROM frames WHERE video_id=?', (video['id'],))
        source = DATA_DIR / video['storage_key']
        batch = []
        processed = 0
        next_sample = 0.0
        total = max(1, int(video['duration_ms'] / (FRAME_INTERVAL_SEC * 1000)) + 1)
        with transaction() as db:
            db.execute('UPDATE jobs SET total=?,updated_at=? WHERE id=?', (total,now(),job_id))
        with av.open(str(source)) as container:
            stream = next(s for s in container.streams if s.type == 'video')
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                pts = float(frame.pts * frame.time_base)
                if pts + 0.01 < next_sample:
                    continue
                image = frame.to_image().convert('RGB')
                batch.append((int(pts * 1000), image))
                next_sample = pts + FRAME_INTERVAL_SEC
                if len(batch) >= 12:
                    processed += _save_batch(batch, video['id'])
                    batch.clear()
                    _progress(job_id, processed, total, started)
            if batch:
                processed += _save_batch(batch, video['id'])
                _progress(job_id, processed, total, started)
        if processed == 0:
            raise ValueError('No decodable video frames')
        with transaction() as db:
            db.execute("UPDATE videos SET status='ready',frame_count=? WHERE id=?", (processed,video['id']))
            db.execute("UPDATE jobs SET status='completed',stage='ready',processed=?,total=?,elapsed_ms=?,updated_at=? WHERE id=?",
                       (processed,processed,int((time.monotonic()-started)*1000),now(),job_id))
    except Exception as exc:
        message = str(exc)[:500]
        with transaction() as db:
            db.execute("UPDATE videos SET status='failed',error=? WHERE id=?", (message,video['id']))
            db.execute("UPDATE jobs SET status='failed',stage='failed',error=?,elapsed_ms=?,updated_at=? WHERE id=?",
                       (message,int((time.monotonic()-started)*1000),now(),job_id))
    finally:
        with _running_lock:
            _running.discard(job_id)

def _save_batch(batch, video_id):
    vectors = encode_images([im for _, im in batch])
    with transaction() as db:
        for (pts, image), vector in zip(batch, vectors):
            frame_id = uuid.uuid4().hex
            thumb_rel = f'frames/{frame_id}.jpg'
            preview = image.copy()
            preview.thumbnail((640, 360), Image.Resampling.LANCZOS)
            preview.save(DATA_DIR / thumb_rel, 'JPEG', quality=82)
            db.execute('INSERT INTO frames(id,video_id,pts_ms,thumbnail_key,embedding,model_id,width,height) VALUES(?,?,?,?,?,?,?,?)',
                       (frame_id,video_id,pts,thumb_rel,json.dumps(vector),MODEL_ID,image.width,image.height))
    return len(batch)

def _progress(job_id, count, total, started):
    with transaction() as db:
        db.execute("UPDATE jobs SET stage='embedding',processed=?,total=?,elapsed_ms=?,updated_at=? WHERE id=?",
                   (count,total,int((time.monotonic()-started)*1000),now(),job_id))
