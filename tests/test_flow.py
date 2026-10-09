"""End-to-end API checks use PostgreSQL/pgvector and mock only Gemini calls."""
import atexit
import io
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import psycopg
from psycopg import sql

TEST_DATABASE = 'surv_test_' + uuid.uuid4().hex[:12]
with psycopg.connect('postgresql:///postgres', autocommit=True) as admin:
    admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(TEST_DATABASE)))

def drop_test_database():
    with psycopg.connect('postgresql:///postgres', autocommit=True) as admin:
        admin.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(TEST_DATABASE)))

atexit.register(drop_test_database)
os.environ['SURV_DATABASE_URL'] = f'postgresql:///{TEST_DATABASE}'
os.environ['SURV_DATA_DIR'] = tempfile.mkdtemp(prefix='surv-test-')
os.environ['SURV_ADMIN_TOKEN'] = 'test-admin'
os.environ['SURV_FRAME_INTERVAL_SEC'] = '1'
os.environ['GEMINI_API_KEY'] = 'test-only-key'

import imageio_ffmpeg
from PIL import Image
from fastapi.testclient import TestClient
from surv.db import row
from surv.main import app
from surv.qa import VideoAnswer, Observation
import surv.main as api_module
import surv.worker as worker_module

HEADERS={'Authorization':'Bearer test-admin'}
VECTOR=[1.0]+[0.0]*767

def make_mp4(path:Path):
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
             'color=c=red:size=160x96:rate=12:duration=3','-c:v','libx264','-pix_fmt','yuv420p',str(path)]
    subprocess.run(command,check=True)

def test_full_investigation_flow(tmp_path,monkeypatch):
    monkeypatch.setattr(worker_module,'encode_images',lambda images:[VECTOR for _ in images])
    monkeypatch.setattr(api_module,'encode_images',lambda images:[VECTOR for _ in images])
    monkeypatch.setattr(api_module,'encode_text',lambda query:VECTOR)
    monkeypatch.setattr(api_module,'encode_hybrid',lambda image,query:VECTOR)
    source=tmp_path/'red.mp4'
    make_mp4(source)
    with TestClient(app) as client:
        assert row("SELECT extversion FROM pg_extension WHERE extname='vector'")
        assert row("SELECT udt_name FROM information_schema.columns WHERE table_name='frames' AND column_name='embedding'")['udt_name']=='vector'
        assert client.get('/api/v1/cameras').status_code==401
        camera=client.post('/api/v1/cameras',headers=HEADERS,json={'name':'Lobby','location':'North wing'}).json()
        upload=client.post('/api/v1/videos',headers=HEADERS,data={'camera_id':camera['id'],'source_started_at_utc':'2026-10-09T10:00:00Z'},
                           files={'file':('red.mp4',source.read_bytes(),'video/mp4')})
        assert upload.status_code==200,upload.text
        job_id=upload.json()['job_id']
        for _ in range(100):
            job=client.get(f'/api/v1/jobs/{job_id}',headers=HEADERS).json()
            if job['status'] in {'completed','failed'}: break
            time.sleep(.1)
        assert job['status']=='completed',job
        result=client.post('/api/v1/search/text',headers=HEADERS,json={'query':'red object'}).json()
        assert result['answer_status']=='possible' and result['matches']
        match=result['matches'][0]
        assert match['source_utc'] and abs(match['similarity_raw']-1.0)<1e-6
        assert client.get(match['thumbnail_url'],headers=HEADERS).status_code==200
        clip=client.get(match['clip_url'],headers=HEADERS)
        assert clip.status_code==200 and len(clip.content)>100
        monkeypatch.setattr(api_module,'answer_video',lambda *args,**kwargs:VideoAnswer(
            answer_status='supported',answer='Two people crossed the view near 00:01.',count=2,
            limitations='Count requires human review.',evidence=[
                Observation(start_seconds=.4,end_seconds=1.6,description='People crossing'),
                Observation(start_seconds=900,end_seconds=901,description='Invalid timestamp')]))
        qa=client.post('/api/v1/qa',headers=HEADERS,data={'video_id':upload.json()['video_id'],'question':'How many people crossed?'})
        assert qa.status_code==200,qa.text
        qa_result=qa.json()
        assert qa_result['answer_status']=='inconclusive' and qa_result['answer_count'] is None
        assert len(qa_result['matches'])==1 and qa_result['matches'][0]['observation']=='People crossing'
        assert qa_result['matches'][0]['observed_start_ms']==400
        assert client.get(qa_result['matches'][0]['clip_url'],headers=HEADERS).status_code==200
        monkeypatch.setattr(api_module,'answer_video',lambda *args,**kwargs:VideoAnswer(
            answer_status='supported',answer='One person crossed.',count=1,
            limitations='Count requires human review.',evidence=[
                Observation(start_seconds=.4,end_seconds=1.6,description='Person crossing')]))
        count_qa=client.post('/api/v1/qa',headers=HEADERS,data={'video_id':upload.json()['video_id'],'question':'How many people crossed?'})
        assert count_qa.json()['answer_status']=='possible' and count_qa.json()['answer_count']==1
        image=Image.new('RGB',(50,50),(255,0,0));stream=io.BytesIO();image.save(stream,'PNG')
        image_qa=client.post('/api/v1/qa',headers=HEADERS,
                             data={'video_id':upload.json()['video_id'],'question':'When does this person appear?',
                                   'subject_type':'person','crop':'[0,0,1,1]'},
                             files={'file':('reference.png',stream.getvalue(),'image/png')})
        assert image_qa.status_code==200,image_qa.text
        assert image_qa.json()['answer_status']=='possible' and image_qa.json()['reference_id']
        for kind in ('image','hybrid'):
            payload={'subject_type':'object','crop':'[0,0,1,1]'}
            if kind=='hybrid':payload['query']='red object'
            image_result=client.post(f'/api/v1/search/{kind}',headers=HEADERS,data=payload,
                                     files={'file':('reference.png',stream.getvalue(),'image/png')}).json()
            assert image_result['matches'] and image_result['reference_id']
            ref_id=image_result['reference_id']
            assert client.delete(f'/api/v1/references/{ref_id}',headers=HEADERS).json()['deleted']
            assert client.get(f'/api/v1/references/{ref_id}/image',headers=HEADERS).status_code==404
        review=client.post(f'/api/v1/evidence/{match["id"]}/review',headers=HEADERS,json={'status':'verified','note':'Visible in source clip'})
        assert review.status_code==200
        report=client.post('/api/v1/reports',headers=HEADERS,json={'title':'Lobby incident','notes':'Reviewed source','evidence_ids':[match['id']]})
        assert report.status_code==200,report.text
        pdf=client.get(report.json()['download_url'],headers=HEADERS)
        assert pdf.status_code==200 and pdf.content.startswith(b'%PDF')
        os.environ['SURV_TOKENS_JSON']='{"outsider":{"actor":"outsider","camera_ids":["other-camera"]}}'
        outsider={'Authorization':'Bearer outsider'}
        assert client.get(f'/api/v1/videos/{upload.json()["video_id"]}',headers=outsider).status_code==404
        assert client.get(match['thumbnail_url'],headers=outsider).status_code==404
        assert client.post('/api/v1/qa',headers=outsider,
                           data={'video_id':upload.json()['video_id'],'question':'What happened?'}).status_code==404
        assert client.post('/api/v1/search/text',headers=outsider,json={'query':'red object','camera_ids':[camera['id']]}).status_code==404
        os.environ['SURV_TOKENS_JSON']='{"restricted":{"actor":"local-admin","camera_ids":["other-camera"]}}'
        assert client.get(f'/api/v1/search/{qa_result["search_id"]}',
                          headers={'Authorization':'Bearer restricted'}).status_code==404
        os.environ['SURV_TOKENS_JSON']=f'{{"investigator":{{"actor":"investigator","role":"investigator","camera_ids":["{camera["id"]}"]}}}}'
        investigator={'Authorization':'Bearer investigator'}
        own_match=client.post('/api/v1/search/text',headers=investigator,json={'query':'red object'}).json()['matches'][0]
        assert client.post(f'/api/v1/evidence/{own_match["id"]}/review',headers=investigator,
                           json={'status':'verified'}).status_code==403
        assert client.post(f'/api/v1/evidence/{own_match["id"]}/review',headers=investigator,
                           json={'status':'possible'}).status_code==200
        os.environ.pop('SURV_TOKENS_JSON',None)

def test_temporal_question_abstains():
    with TestClient(app) as client:
        response=client.post('/api/v1/search/text',headers=HEADERS,json={'query':'Did someone fall before the alarm?'}).json()
        assert response['answer_status']=='inconclusive' and response['matches']==[]
