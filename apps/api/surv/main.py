import asyncio
import hashlib
import io
import json
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import ADMIN_TOKEN, DATA_DIR, EMBEDDING_DIMENSIONS, MAX_UPLOAD_BYTES, MODEL_ID, QA_MODEL_ID
from .db import audit, connect, init_db, row, rows, transaction
from .worker import probe, start_job, now
from .vision import configured, encode_images, encode_text, encode_hybrid
from .qa import answer_video, validated_observations
from .reports import build_report

Image.MAX_IMAGE_PIXELS = 30_000_000

@asynccontextmanager
async def lifespan(app):
    init_db()
    for job in rows("SELECT id FROM jobs WHERE status IN ('queued','processing')"):
        start_job(job['id'])
    yield

app = FastAPI(title='Surveillance Q&A API', version='0.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],
                   allow_methods=['*'], allow_headers=['*'])

def auth(authorization: Annotated[str | None, Header()] = None):
    token = (authorization or '').removeprefix('Bearer ').strip()
    if token and token == ADMIN_TOKEN:
        return {'actor': 'local-admin', 'role': 'admin', 'camera_ids': None}
    try:
        configured = json.loads(os.getenv('SURV_TOKENS_JSON', '{}'))
        identity = configured.get(token)
        if token and identity and isinstance(identity.get('camera_ids'), list):
            return {'actor': identity.get('actor','scoped-user'), 'role': identity.get('role','investigator'),
                    'camera_ids': set(identity['camera_ids'])}
    except (ValueError, AttributeError):
        pass
    raise HTTPException(401, 'Valid bearer token required')

def allowed(user, camera_id):
    if user['camera_ids'] is not None and camera_id not in user['camera_ids']:
        raise HTTPException(404, 'Resource not found')

def _camera_filter(user, requested=None):
    if requested:
        for camera_id in requested:
            allowed(user, camera_id)
        return set(requested)
    return user['camera_ids']

def _utc_for(video, pts_ms):
    if not video['source_started_at_utc']:
        return None
    try:
        base = datetime.fromisoformat(video['source_started_at_utc'].replace('Z','+00:00'))
        camera = row('SELECT clock_offset_ms FROM cameras WHERE id=?', (video['camera_id'],))
        return (base + timedelta(milliseconds=pts_ms + camera['clock_offset_ms'])).astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None

class CameraInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=150)
    timezone: str = 'UTC'
    clock_offset_ms: int = 0

class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    camera_ids: list[str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int = Field(default=12, ge=1, le=50)

class ReviewInput(BaseModel):
    status: str
    note: str = Field(default='', max_length=2000)

class ReportInput(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    notes: str = Field(default='', max_length=5000)
    evidence_ids: list[str] = Field(min_length=1, max_length=40)

@app.get('/api/v1/health')
def health():
    return {'ok': True, 'model_id': MODEL_ID, 'qa_model_id': QA_MODEL_ID,
            'ai_provider': 'Gemini', 'provider_configured': configured()}

@app.get('/api/v1/me')
def me(user=Depends(auth)):
    return {'actor': user['actor'], 'role': user['role'],
            'camera_ids': sorted(user['camera_ids']) if user['camera_ids'] is not None else None}

@app.get('/api/v1/overview')
def overview(user=Depends(auth)):
    visible = _camera_filter(user)
    videos = [v for v in rows('SELECT * FROM videos ORDER BY created_at DESC') if visible is None or v['camera_id'] in visible]
    jobs = [j for j in rows('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 30') if any(v['id']==j['video_id'] for v in videos)]
    return {'camera_count': len([c for c in rows('SELECT id FROM cameras') if visible is None or c['id'] in visible]),
            'video_count': len(videos), 'ready_count': sum(v['status']=='ready' for v in videos),
            'frame_count': sum(v['frame_count'] for v in videos), 'recent_videos': videos[:6], 'jobs': jobs[:8],
            'provider_configured': configured()}

@app.get('/api/v1/cameras')
def cameras(user=Depends(auth)):
    return [c for c in rows('SELECT * FROM cameras ORDER BY created_at DESC') if user['camera_ids'] is None or c['id'] in user['camera_ids']]

@app.post('/api/v1/cameras')
def create_camera(body: CameraInput, user=Depends(auth)):
    if user['role'] != 'admin':
        raise HTTPException(403, 'Administrator required')
    camera_id = uuid.uuid4().hex
    with transaction() as db:
        db.execute('INSERT INTO cameras(id,name,location,timezone,clock_offset_ms,created_at) VALUES(?,?,?,?,?,?)',
                   (camera_id,body.name,body.location,body.timezone,body.clock_offset_ms,now()))
    audit('camera.create',camera_id, actor=user['actor'])
    return row('SELECT * FROM cameras WHERE id=?',(camera_id,))

@app.get('/api/v1/videos')
def videos(user=Depends(auth)):
    return [v for v in rows('SELECT v.*, c.name AS camera_name,c.location FROM videos v JOIN cameras c ON c.id=v.camera_id ORDER BY v.created_at DESC')
            if user['camera_ids'] is None or v['camera_id'] in user['camera_ids']]

@app.post('/api/v1/videos')
async def upload_video(file: UploadFile = File(...), camera_id: str = Form(...), source_started_at_utc: str | None = Form(None), user=Depends(auth)):
    _require_provider()
    allowed(user,camera_id)
    if not row('SELECT id FROM cameras WHERE id=?',(camera_id,)):
        raise HTTPException(404,'Camera not found')
    if source_started_at_utc:
        try:
            parsed = datetime.fromisoformat(source_started_at_utc.replace('Z','+00:00'))
            if parsed.tzinfo is None: raise ValueError()
            source_started_at_utc = parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            raise HTTPException(422,'Source time must include a timezone')
    video_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    rel = f'videos/{video_id}.mp4'
    path = DATA_DIR / rel
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open('wb') as output:
            while chunk := await file.read(1024*1024):
                count += len(chunk)
                if count > MAX_UPLOAD_BYTES:
                    raise HTTPException(413,'Video exceeds upload limit')
                output.write(chunk)
                digest.update(chunk)
        with path.open('rb') as source:
            header = source.read(16)
        if len(header) < 12 or header[4:8] != b'ftyp':
            raise HTTPException(415,'An MP4 container is required')
        try:
            media = probe(path)
        except Exception as exc:
            raise HTTPException(415,str(exc))
        with transaction() as db:
            db.execute('INSERT INTO videos(id,camera_id,filename,storage_key,sha256,duration_ms,source_started_at_utc,width,height,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (video_id,camera_id,Path(file.filename or 'upload.mp4').name,rel,digest.hexdigest(),media['duration_ms'],source_started_at_utc,media['width'],media['height'],'queued',now()))
            db.execute('INSERT INTO jobs(id,video_id,stage,status,created_at,updated_at) VALUES(?,?,?,?,?,?)',
                       (job_id,video_id,'queued','queued',now(),now()))
        audit('video.upload',video_id,actor=user['actor'])
        start_job(job_id)
        return {'video_id':video_id,'job_id':job_id,'status':'queued'}
    except Exception:
        path.unlink(missing_ok=True)
        raise

@app.get('/api/v1/videos/{video_id}')
def video_detail(video_id: str, user=Depends(auth)):
    video = row('SELECT * FROM videos WHERE id=?',(video_id,))
    if not video: raise HTTPException(404,'Video not found')
    allowed(user,video['camera_id'])
    return video

@app.get('/api/v1/jobs/{job_id}')
def job_detail(job_id: str, user=Depends(auth)):
    job = row('SELECT j.*,v.camera_id FROM jobs j JOIN videos v ON v.id=j.video_id WHERE j.id=?',(job_id,))
    if not job: raise HTTPException(404,'Job not found')
    allowed(user,job['camera_id'])
    return job

@app.post('/api/v1/jobs/{job_id}/retry')
def retry_job(job_id: str, user=Depends(auth)):
    job = job_detail(job_id,user)
    if job['status'] != 'failed': raise HTTPException(409,'Only failed jobs can be retried')
    with transaction() as db:
        db.execute("UPDATE jobs SET status='queued',stage='queued',processed=0,error=NULL,updated_at=? WHERE id=?",(now(),job_id))
    start_job(job_id)
    audit('job.retry',job_id,actor=user['actor'])
    return {'status':'queued'}

def _require_provider():
    if not configured():
        raise HTTPException(503, 'Set GEMINI_API_KEY in .env before indexing or searching')

def _ranked_frames(vector, user, camera_ids, start_time, end_time, limit):
    if (start_time and start_time.tzinfo is None) or (end_time and end_time.tzinfo is None):
        raise HTTPException(422, 'Time filters must include a timezone')
    if start_time and end_time and start_time > end_time:
        raise HTTPException(422, 'Start time must be before end time')
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise HTTPException(500, 'Embedding dimension does not match the pgvector index')
    scope = _camera_filter(user,camera_ids)
    import numpy as np
    query_vector = np.asarray(vector, dtype=np.float32)
    conditions = ["v.status='ready'", 'f.model_id=?']
    filters = [MODEL_ID]
    if scope is not None:
        if not scope:
            return []
        conditions.append('v.camera_id = ANY(?)')
        filters.append(sorted(scope))
    stamp = "(v.source_started_at_utc::timestamptz + (f.pts_ms + c.clock_offset_ms) * interval '1 millisecond')"
    if start_time or end_time:
        conditions.append('v.source_started_at_utc IS NOT NULL')
    if start_time:
        conditions.append(stamp+' >= ?')
        filters.append(start_time)
    if end_time:
        conditions.append(stamp+' <= ?')
        filters.append(end_time)
    sql = f'''SELECT f.id, f.video_id, f.pts_ms, v.camera_id, v.duration_ms,
                     1 - (f.embedding <=> ?) AS similarity_raw
              FROM frames f JOIN videos v ON v.id=f.video_id
              JOIN cameras c ON c.id=v.camera_id
              WHERE {' AND '.join(conditions)}
              ORDER BY f.embedding <=> ? LIMIT ?'''
    with connect() as db:
        # Continue an HNSW scan past filtered-out camera/time rows when needed.
        db.execute('SET LOCAL hnsw.iterative_scan = strict_order')
        return list(db.execute(sql, [query_vector, *filters, query_vector, limit]).fetchall())

def _create_search(kind, query, reference_id, vector, user, camera_ids, start_time, end_time, top_k):
    ranked = _ranked_frames(vector,user,camera_ids,start_time,end_time,top_k*8)
    selected=[]
    for frame in ranked:
        if any(item['video_id']==frame['video_id'] and abs(item['pts_ms']-frame['pts_ms']) < 6000 for _,item in selected):
            continue
        selected.append((float(frame['similarity_raw']),frame))
        if len(selected)>=top_k: break
    search_id=uuid.uuid4().hex
    status='possible' if selected else 'not_found'
    summary=(f'{len(selected)} candidate window(s) found. Inspect source footage before drawing conclusions.' if selected else
             'No candidate was found in the selected indexed footage.')
    limitations=('Raw Gemini embedding cosine similarity is uncalibrated. Sampled frames may miss short events; visual resemblance does not establish identity or action.' if selected else
                 'No result may reflect visibility, sampling, selected cameras, time filters, or incomplete indexing.')
    with transaction() as db:
        db.execute('INSERT INTO searches(id,kind,query,reference_id,owner_actor,status,summary,limitations,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                   (search_id,kind,query,reference_id,user['actor'],status,summary,limitations,now()))
        for score,frame in selected:
            start=max(0,frame['pts_ms']-3000)
            end=min(frame['duration_ms'],frame['pts_ms']+5000)
            db.execute('INSERT INTO matches(id,search_id,frame_id,video_id,camera_id,start_ms,end_ms,similarity_raw,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                       (uuid.uuid4().hex,search_id,frame['id'],frame['video_id'],frame['camera_id'],start,end,score,now()))
    audit('search.'+kind,search_id,details=json.dumps({'camera_ids':camera_ids,'results':len(selected)}),actor=user['actor'])
    return search_detail(search_id,user)

@app.post('/api/v1/search/text')
def search_text(body: SearchInput, user=Depends(auth)):
    if _is_temporal_question(body.query):
        return {'search_id':None,'answer_status':'inconclusive','summary':'This question needs ordered video observations. Frame similarity cannot establish the event or sequence.',
                'limitations':'Review the source video manually. Temporal event models are not enabled in this MVP.','matches':[]}
    _require_provider()
    vector=encode_text(body.query)
    return _create_search('text',body.query,None,vector,user,body.camera_ids,body.start_time,body.end_time,body.top_k)

def _is_temporal_question(query):
    import re
    return bool(re.search(r'\b(fell|falling|abandoned|left behind|before|after|how long|remained|dwell|opened|closed)\b',query.lower()))

async def _read_reference(file, subject_type, crop):
    if subject_type not in {'person','vehicle','object'}:
        raise HTTPException(422,'Choose person, vehicle, or object')
    data=await file.read(15*1024*1024+1)
    if len(data)>15*1024*1024: raise HTTPException(413,'Image exceeds 15 MB')
    try:
        image=Image.open(io.BytesIO(data))
        image.verify()
        image=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    except (UnidentifiedImageError,ValueError,OSError):
        raise HTTPException(415,'Unsupported image')
    try:
        bounds=json.loads(crop) if crop else [0,0,1,1]
        x1,y1,x2,y2=[float(x) for x in bounds]
        if not (0<=x1<x2<=1 and 0<=y1<y2<=1 and (x2-x1)*(y2-y1)>=0.002): raise ValueError()
    except (ValueError,TypeError):
        raise HTTPException(422,'Crop must be normalized [x1,y1,x2,y2]')
    cropped=image.crop((int(x1*image.width),int(y1*image.height),int(x2*image.width),int(y2*image.height)))
    if min(cropped.size)<16: raise HTTPException(422,'Selected subject is too small')
    return image, cropped

def _store_reference(file, subject_type, image, cropped, user):
    ref_id=uuid.uuid4().hex
    original_rel=f'references/{ref_id}.jpg'
    crop_rel=f'references/{ref_id}-crop.jpg'
    try:
        image.save(DATA_DIR/original_rel,'JPEG',quality=90)
        cropped.save(DATA_DIR/crop_rel,'JPEG',quality=90)
        with transaction() as db:
            db.execute('INSERT INTO reference_images(id,filename,storage_key,crop_key,subject_type,owner_actor,created_at) VALUES(?,?,?,?,?,?,?)',
                       (ref_id,Path(file.filename or 'reference.jpg').name,original_rel,crop_rel,subject_type,user['actor'],now()))
    except Exception:
        (DATA_DIR/original_rel).unlink(missing_ok=True)
        (DATA_DIR/crop_rel).unlink(missing_ok=True)
        raise
    return ref_id

async def _image_search(file, subject_type, crop, query, camera_ids, start_time, end_time, top_k, user):
    _require_provider()
    image, cropped = await _read_reference(file, subject_type, crop)
    ref_id = None
    try:
        vector=encode_images([cropped])[0]
        kind='image'
        if query:
            vector=encode_hybrid(cropped,query)
            kind='hybrid'
        ref_id=_store_reference(file,subject_type,image,cropped,user)
        result=_create_search(kind,query,ref_id,vector,user,camera_ids,start_time,end_time,top_k)
        result['reference_id']=ref_id
        return result
    except Exception:
        if ref_id:
            with transaction() as db:
                db.execute('DELETE FROM reference_images WHERE id=?',(ref_id,))
            (DATA_DIR/f'references/{ref_id}.jpg').unlink(missing_ok=True)
            (DATA_DIR/f'references/{ref_id}-crop.jpg').unlink(missing_ok=True)
        raise

@app.post('/api/v1/qa')
async def ask_video(video_id: str = Form(...), question: str = Form(...),
                    file: UploadFile | None = File(None), subject_type: str = Form('object'),
                    crop: str | None = Form(None), user=Depends(auth)):
    """Answer a question about one authorized original video and link valid intervals."""
    _require_provider()
    question=question.strip()
    if not 2 <= len(question) <= 500:
        raise HTTPException(422,'Question must be 2–500 characters')
    video=row('SELECT * FROM videos WHERE id=?',(video_id,))
    if not video:
        raise HTTPException(404,'Video not found')
    allowed(user,video['camera_id'])
    if video['status'] != 'ready':
        raise HTTPException(409,'Video must finish indexing before Q&A')
    image=cropped=None
    if file is not None:
        image,cropped=await _read_reference(file,subject_type,crop)
    try:
        answer=await asyncio.to_thread(answer_video,DATA_DIR/video['storage_key'],question,video['duration_ms'],cropped,subject_type if cropped else None)
    except Exception as exc:
        raise HTTPException(502,f'Gemini video analysis failed: {type(exc).__name__}') from exc
    intervals=validated_observations(answer,video['duration_ms'])
    status=answer.answer_status
    summary=answer.answer
    count=answer.count
    limitations=('Gemini observations and timestamps are provisional; review each source clip. '+answer.limitations).strip()
    import re
    counting_question=bool(re.search(r'\b(how many|number of|count|kitne|kitni|ginti)\b|कितने|कितनी',question,re.IGNORECASE))
    if counting_question:
        if count is None or (count > 0 and count != len(intervals)) or (count == 0 and intervals) or any(end-start > 12000 for _,_,_,start,end in intervals):
            status='inconclusive'
            count=None
            summary='A reliable count could not be linked to one valid source interval per observed passage. Review the listed clips and original video.'
            limitations=(limitations+' Automated counting has not been independently verified.').strip()
        elif count > 0:
            status='possible'
            summary=f'Gemini estimates {count} distinct passage(s). Each listed time is a model observation; inspect every linked clip before relying on this count.'
            limitations=(limitations+' This is a model estimate, not a tracker-verified count.').strip()
        else:
            status='not_found'
            summary='Gemini did not observe a completed passage matching the question in this video.'
    else:
        count=None
    if cropped is not None and status in {'supported','possible'}:
        status='possible'
        times=', '.join(f'{start//60000:02d}:{start//1000%60:02d}–{end//60000:02d}:{end//1000%60:02d}'
                        for _,_,_,start,end in intervals)
        summary=(f'A visually similar {subject_type} may appear at {times} (video-relative). '
                 'The reference image does not establish exact identity.') if times else summary
        limitations=(limitations+' Visual similarity alone cannot verify an exact identity.').strip()
    elif cropped is not None and status=='not_found':
        summary=f'Gemini did not observe a visually similar {subject_type} in this video. This does not prove absence.'
    if status in {'supported','possible'} and not intervals:
        status='inconclusive'
        summary='Gemini returned an answer, but no valid source-video interval could be linked to it.'
        limitations=(limitations+' Review the original video manually.').strip()
        count=None
    ref_id=None
    search_id=uuid.uuid4().hex
    try:
        if cropped is not None:
            ref_id=_store_reference(file,subject_type,image,cropped,user)
        with transaction() as db:
            db.execute('INSERT INTO searches(id,kind,query,reference_id,owner_actor,video_id,status,summary,limitations,answer_count,answer_model,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (search_id,'qa',question,ref_id,user['actor'],video_id,status,summary,limitations,count,QA_MODEL_ID,now()))
            for start_ms,end_ms,description,observed_start_ms,observed_end_ms in intervals:
                frame=db.execute('SELECT id FROM frames WHERE video_id=? ORDER BY ABS(pts_ms-?) LIMIT 1',
                                 (video_id,(start_ms+end_ms)//2)).fetchone()
                if not frame:
                    continue
                db.execute('INSERT INTO matches(id,search_id,frame_id,video_id,camera_id,start_ms,end_ms,similarity_raw,observation,observed_start_ms,observed_end_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                           (uuid.uuid4().hex,search_id,frame['id'],video_id,video['camera_id'],start_ms,end_ms,0.0,description,observed_start_ms,observed_end_ms,now()))
    except Exception:
        if ref_id:
            with transaction() as db:
                db.execute('DELETE FROM reference_images WHERE id=?',(ref_id,))
            (DATA_DIR/f'references/{ref_id}.jpg').unlink(missing_ok=True)
            (DATA_DIR/f'references/{ref_id}-crop.jpg').unlink(missing_ok=True)
        raise
    audit('search.qa',search_id,details=json.dumps({'video_id':video_id,'evidence_intervals':len(intervals)}),actor=user['actor'])
    return search_detail(search_id,user)

@app.post('/api/v1/search/image')
async def search_image(file: UploadFile = File(...), subject_type: str = Form('object'), crop: str | None = Form(None),
                       camera_ids: str | None = Form(None), start_time: str | None = Form(None), end_time: str | None = Form(None),
                       top_k: int = Form(12), user=Depends(auth)):
    return await _image_search(file,subject_type,crop,None,_parse_ids(camera_ids),_parse_time(start_time),_parse_time(end_time),min(max(top_k,1),50),user)

@app.post('/api/v1/search/hybrid')
async def search_hybrid(file: UploadFile = File(...), query: str = Form(...), subject_type: str = Form('object'), crop: str | None = Form(None),
                        camera_ids: str | None = Form(None), start_time: str | None = Form(None), end_time: str | None = Form(None),
                        top_k: int = Form(12), user=Depends(auth)):
    if len(query)<2 or len(query)>500: raise HTTPException(422,'Query must be 2–500 characters')
    if _is_temporal_question(query): raise HTTPException(422,'Temporal claims require video review and are unavailable in hybrid retrieval')
    return await _image_search(file,subject_type,crop,query,_parse_ids(camera_ids),_parse_time(start_time),_parse_time(end_time),min(max(top_k,1),50),user)

def _parse_ids(value):
    if not value: return None
    try:
        ids=json.loads(value)
        if not isinstance(ids,list) or not all(isinstance(x,str) for x in ids): raise ValueError()
        return ids
    except ValueError: raise HTTPException(422,'camera_ids must be a JSON array')

def _parse_time(value):
    if not value: return None
    try:
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None: raise ValueError()
        return parsed
    except ValueError: raise HTTPException(422,'Time filter must include a timezone')

@app.get('/api/v1/search/{search_id}')
def search_detail(search_id: str, user=Depends(auth)):
    search=row('SELECT * FROM searches WHERE id=?',(search_id,))
    if not search: raise HTTPException(404,'Search not found')
    if search['owner_actor'] != user['actor'] and user['role'] != 'admin':
        raise HTTPException(404,'Search not found')
    if search['kind']=='qa':
        video=row('SELECT camera_id FROM videos WHERE id=?',(search['video_id'],))
        if not video:
            raise HTTPException(404,'Search not found')
        allowed(user,video['camera_id'])
    matches=[]
    for item in rows('''SELECT m.*,f.pts_ms,f.thumbnail_key,v.filename,v.sha256,v.source_started_at_utc,
                               c.name AS camera_name,c.location
                        FROM matches m JOIN frames f ON f.id=m.frame_id JOIN videos v ON v.id=m.video_id
                        JOIN cameras c ON c.id=m.camera_id WHERE m.search_id=? ORDER BY m.similarity_raw DESC''',(search_id,)):
        if user['camera_ids'] is not None and item['camera_id'] not in user['camera_ids']: continue
        item['source_utc']=_utc_for(item,item['observed_start_ms'] if item['observed_start_ms'] is not None else item['start_ms'])
        item['thumbnail_url']=f'/api/v1/evidence/{item["id"]}/keyframe'
        item['clip_url']=f'/api/v1/evidence/{item["id"]}/clip'
        matches.append(item)
    return {'search_id':search_id,'kind':search['kind'],'query':search['query'],'reference_id':search['reference_id'],
            'answer_count':search['answer_count'],'answer_model':search['answer_model'],
            'answer_status':search['status'] if matches or search['kind']=='qa' else 'not_found',
            'summary':search['summary'] if matches or search['kind']=='qa' else 'No authorized candidates available.',
            'limitations':search['limitations'],'matches':matches}

def _match(match_id,user):
    match=row('''SELECT m.*,s.owner_actor,s.kind,s.query,s.status AS answer_status,s.answer_model,f.thumbnail_key,f.model_id,v.storage_key,v.duration_ms,v.sha256,v.source_started_at_utc,
                       c.name AS camera_name,c.location FROM matches m JOIN frames f ON f.id=m.frame_id
                       JOIN searches s ON s.id=m.search_id JOIN videos v ON v.id=m.video_id
                       JOIN cameras c ON c.id=m.camera_id WHERE m.id=?''',(match_id,))
    if not match: raise HTTPException(404,'Evidence not found')
    allowed(user,match['camera_id'])
    if match['owner_actor'] != user['actor'] and user['role'] != 'admin':
        raise HTTPException(404,'Evidence not found')
    return match

@app.get('/api/v1/evidence/{match_id}/keyframe')
def keyframe(match_id: str, user=Depends(auth)):
    item=_match(match_id,user)
    audit('evidence.preview',match_id,actor=user['actor'])
    return FileResponse(DATA_DIR/item['thumbnail_key'],media_type='image/jpeg',headers={'Cache-Control':'private, max-age=60'})

@app.get('/api/v1/evidence/{match_id}/clip')
def clip(match_id: str, user=Depends(auth)):
    item=_match(match_id,user)
    clip_path=DATA_DIR/'clips'/f'{match_id}.mp4'
    if not clip_path.exists():
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        temp=clip_path.with_suffix('.tmp.mp4')
        duration=max(0.1,(item['end_ms']-item['start_ms'])/1000)
        command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str(item['start_ms']/1000),'-i',str(DATA_DIR/item['storage_key']),
                 '-t',str(duration),'-an','-c:v','libx264','-preset','ultrafast','-movflags','+faststart',str(temp)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if result.returncode != 0:
            temp.unlink(missing_ok=True)
            raise HTTPException(500,'Clip generation failed: '+result.stderr[-200:])
        temp.replace(clip_path)
    audit('evidence.clip',match_id,actor=user['actor'])
    return FileResponse(clip_path,media_type='video/mp4',headers={'Accept-Ranges':'bytes','Cache-Control':'private, max-age=60'})

@app.post('/api/v1/evidence/{match_id}/review')
def review(match_id: str, body: ReviewInput, user=Depends(auth)):
    _match(match_id,user)
    if body.status not in {'possible','rejected','verified'}: raise HTTPException(422,'Invalid review status')
    if body.status == 'verified' and user['role'] not in {'reviewer','admin'}:
        raise HTTPException(403,'Reviewer role required for verification')
    with transaction() as db:
        db.execute('UPDATE matches SET review_status=?,review_note=?,reviewed_at=? WHERE id=?',(body.status,body.note,now(),match_id))
    audit('evidence.review',match_id,details=body.status,actor=user['actor'])
    return {'id':match_id,'review_status':body.status,'review_note':body.note}

@app.get('/api/v1/references/{ref_id}/image')
def reference_image(ref_id: str, crop: bool = True, user=Depends(auth)):
    ref=row('SELECT * FROM reference_images WHERE id=?',(ref_id,))
    if not ref: raise HTTPException(404,'Reference not found')
    if ref['owner_actor'] != user['actor'] and user['role'] != 'admin':
        raise HTTPException(404,'Reference not found')
    linked=rows('SELECT DISTINCT camera_id FROM matches m JOIN searches s ON s.id=m.search_id WHERE s.reference_id=?',(ref_id,))
    if user['camera_ids'] is not None and any(x['camera_id'] not in user['camera_ids'] for x in linked):
        raise HTTPException(404,'Reference not found')
    return FileResponse(DATA_DIR/(ref['crop_key'] if crop else ref['storage_key']),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=60'})

@app.delete('/api/v1/references/{ref_id}')
def delete_reference(ref_id: str, user=Depends(auth)):
    ref=row('SELECT * FROM reference_images WHERE id=?',(ref_id,))
    if not ref: raise HTTPException(404,'Reference not found')
    if ref['owner_actor'] != user['actor'] and user['role'] != 'admin':
        raise HTTPException(404,'Reference not found')
    with transaction() as db:
        db.execute('UPDATE searches SET reference_id=NULL WHERE reference_id=?',(ref_id,))
        db.execute('DELETE FROM reference_images WHERE id=?',(ref_id,))
    (DATA_DIR/ref['storage_key']).unlink(missing_ok=True)
    (DATA_DIR/ref['crop_key']).unlink(missing_ok=True)
    audit('reference.delete',ref_id,actor=user['actor'])
    return {'deleted':True}

@app.post('/api/v1/reports')
def create_report(body: ReportInput, user=Depends(auth)):
    unique=list(dict.fromkeys(body.evidence_ids))
    evidence=[_match(evidence_id,user) for evidence_id in unique]
    for item in evidence:
        item['source_utc']=_utc_for(item,item['observed_start_ms'] if item['observed_start_ms'] is not None else item['start_ms'])
    report_id=uuid.uuid4().hex
    rel=f'reports/{report_id}.pdf'
    build_report(DATA_DIR/rel,body.title,body.notes,evidence)
    with transaction() as db:
        db.execute('INSERT INTO reports(id,title,notes,storage_key,evidence_ids,owner_actor,created_at) VALUES(?,?,?,?,?,?,?)',
                   (report_id,body.title,body.notes,rel,json.dumps(unique),user['actor'],now()))
    audit('report.create',report_id,details=json.dumps(unique),actor=user['actor'])
    return {'report_id':report_id,'download_url':f'/api/v1/reports/{report_id}/download'}

@app.get('/api/v1/reports/{report_id}/download')
def download_report(report_id: str, user=Depends(auth)):
    report=row('SELECT * FROM reports WHERE id=?',(report_id,))
    if not report: raise HTTPException(404,'Report not found')
    if report['owner_actor'] != user['actor'] and user['role'] != 'admin':
        raise HTTPException(404,'Report not found')
    for evidence_id in json.loads(report['evidence_ids']): _match(evidence_id,user)
    audit('report.download',report_id,actor=user['actor'])
    return FileResponse(DATA_DIR/report['storage_key'],media_type='application/pdf',filename=f'evidence-report-{report_id[:8]}.pdf')

@app.get('/api/v1/metrics/pipeline')
def pipeline_metrics(user=Depends(auth)):
    if user['role']!='admin': raise HTTPException(403,'Administrator required')
    jobs=rows('SELECT status,stage,elapsed_ms,processed,total,updated_at FROM jobs ORDER BY created_at DESC LIMIT 50')
    return {'jobs':jobs,'completed':sum(j['status']=='completed' for j in jobs),'failed':sum(j['status']=='failed' for j in jobs)}
