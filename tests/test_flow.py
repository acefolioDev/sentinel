"""End-to-end API checks use a deterministic encoder to avoid a model download in CI."""
import io
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ['SURV_DATA_DIR'] = tempfile.mkdtemp(prefix='surv-test-')
os.environ['SURV_ADMIN_TOKEN'] = 'test-admin'
os.environ['SURV_FRAME_INTERVAL_SEC'] = '1'

import imageio_ffmpeg
from PIL import Image
from fastapi.testclient import TestClient
from surv.main import app
import surv.main as api_module
import surv.worker as worker_module

HEADERS={'Authorization':'Bearer test-admin'}

def make_mp4(path:Path):
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
             'color=c=red:size=160x96:rate=12:duration=3','-c:v','libx264','-pix_fmt','yuv420p',str(path)]
    subprocess.run(command,check=True)

def test_full_investigation_flow(tmp_path,monkeypatch):
    monkeypatch.setattr(worker_module,'encode_images',lambda images:[[1.0,0.0] for _ in images])
    monkeypatch.setattr(api_module,'encode_images',lambda images:[[1.0,0.0] for _ in images])
    monkeypatch.setattr(api_module,'encode_text',lambda query:[1.0,0.0])
    source=tmp_path/'red.mp4'
    make_mp4(source)
    with TestClient(app) as client:
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
        assert match['source_utc'] and match['similarity_raw']==1.0
        assert client.get(match['thumbnail_url'],headers=HEADERS).status_code==200
        clip=client.get(match['clip_url'],headers=HEADERS)
        assert clip.status_code==200 and len(clip.content)>100
        image=Image.new('RGB',(50,50),(255,0,0));stream=io.BytesIO();image.save(stream,'PNG')
        image_result=client.post('/api/v1/search/image',headers=HEADERS,data={'subject_type':'object','crop':'[0,0,1,1]'},
                                 files={'file':('reference.png',stream.getvalue(),'image/png')}).json()
        assert image_result['matches'] and image_result['reference_id']
        review=client.post(f'/api/v1/evidence/{match["id"]}/review',headers=HEADERS,json={'status':'verified','note':'Visible in source clip'})
        assert review.status_code==200
        report=client.post('/api/v1/reports',headers=HEADERS,json={'title':'Lobby incident','notes':'Reviewed source','evidence_ids':[match['id']]})
        assert report.status_code==200,report.text
        pdf=client.get(report.json()['download_url'],headers=HEADERS)
        assert pdf.status_code==200 and pdf.content.startswith(b'%PDF')
        ref_id=image_result['reference_id']
        assert client.delete(f'/api/v1/references/{ref_id}',headers=HEADERS).json()['deleted']
        assert client.get(f'/api/v1/references/{ref_id}/image',headers=HEADERS).status_code==404
        os.environ['SURV_TOKENS_JSON']='{"outsider":{"actor":"outsider","camera_ids":["other-camera"]}}'
        outsider={'Authorization':'Bearer outsider'}
        assert client.get(f'/api/v1/videos/{upload.json()["video_id"]}',headers=outsider).status_code==404
        assert client.get(match['thumbnail_url'],headers=outsider).status_code==404
        assert client.post('/api/v1/search/text',headers=outsider,json={'query':'red object','camera_ids':[camera['id']]}).status_code==404
        os.environ['SURV_TOKENS_JSON']=f'{{"investigator":{{"actor":"investigator","role":"investigator","camera_ids":["{camera["id"]}"]}}}}'
        investigator={'Authorization':'Bearer investigator'}
        own_match=client.post('/api/v1/search/text',headers=investigator,json={'query':'red object'}).json()['matches'][0]
        assert client.post(f'/api/v1/evidence/{own_match["id"]}/review',headers=investigator,
                           json={'status':'verified'}).status_code==403
        assert client.post(f'/api/v1/evidence/{own_match["id"]}/review',headers=investigator,
                           json={'status':'possible'}).status_code==200
        os.environ.pop('SURV_TOKENS_JSON',None)

def test_temporal_question_abstains(monkeypatch):
    with TestClient(app) as client:
        response=client.post('/api/v1/search/text',headers=HEADERS,json={'query':'Did someone fall before the alarm?'}).json()
        assert response['answer_status']=='inconclusive' and response['matches']==[]
