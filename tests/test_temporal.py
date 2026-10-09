"""Temporal count ownership and bounding box validation without Gemini calls."""
import json
from types import SimpleNamespace

from PIL import Image

from surv import qa, localization


def test_count_question_uses_local_tracks_without_gemini(monkeypatch, tmp_path):
    monkeypatch.setattr(qa,'_client_instance',lambda: (_ for _ in ()).throw(AssertionError('Gemini must not count')))
    monkeypatch.setattr(qa,'count_passages',lambda *args,**kwargs:{
        'count':2,'events':[{'start_seconds':3,'end_seconds':4,'description':'Person #1 crossed'},
                            {'start_seconds':10,'end_seconds':11,'description':'Person #2 crossed'}],
        'frames':[],'reason':'Local tracks'})
    answer=qa.answer_video(tmp_path/'unused.mp4','How many people crossed?',30000)
    assert answer.count==2 and answer._count_method=='local-tracker'
    assert [item.start_seconds for item in answer.evidence]==[3,10]


def test_unsupported_count_is_inconclusive_without_gemini(monkeypatch, tmp_path):
    monkeypatch.setattr(qa,'_client_instance',lambda: (_ for _ in ()).throw(AssertionError('Gemini must not count')))
    answer=qa.answer_video(tmp_path/'unused.mp4','How many umbrellas passed?',30000)
    assert answer.answer_status=='inconclusive'
    assert answer.count is None


def test_localization_validates_boxes_and_reuses_track_ids(monkeypatch, tmp_path):
    images=[(1000,Image.new('RGB',(80,60))),(2000,Image.new('RGB',(80,60)))]
    monkeypatch.setattr(localization,'_sample_frames',lambda *args:images)
    payload={'frames':[{'frame_index':0,'boxes':[{'box_2d':[100,100,500,500],'label':'person'},
                                                      {'box_2d':[0,-1,100,100],'label':'invalid'}]},
                       {'frame_index':1,'boxes':[{'box_2d':[120,120,520,520],'label':'person'}]}]}
    fake=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:SimpleNamespace(text=json.dumps(payload))))
    monkeypatch.setattr(localization,'_client_instance',lambda:fake)
    frames=localization._detect_sampled(images,'person',None)
    localization._assign_tracks(frames)
    assert len(frames[0]['boxes'])==1
    assert frames[0]['boxes'][0]['track_id']==frames[1]['boxes'][0]['track_id']
    assert frames[0]['boxes'][0]['x1']==.1


def test_tracks_do_not_switch_object_class():
    frames=[{'pts_ms':0,'boxes':[{'x1':.2,'y1':.2,'x2':.4,'y2':.4,'label':'car'}]},
            {'pts_ms':500,'boxes':[{'x1':.21,'y1':.21,'x2':.41,'y2':.41,'label':'truck'}]}]
    localization._assign_tracks(frames)
    assert frames[0]['boxes'][0]['track_id']!=frames[1]['boxes'][0]['track_id']


def test_count_uses_track_crossing_instead_of_frame_count(monkeypatch, tmp_path):
    sampled=[(i*500,Image.new('RGB',(40,30))) for i in range(5)]
    monkeypatch.setattr(localization,'_sample_frames',lambda *args,**kwargs:sampled)
    def box(cx):
        return {'x1':cx-.05,'y1':.3,'x2':cx+.05,'y2':.6,'label':'person'}
    positions=[.2,.34,.49,.65,.8]
    frames=[{'pts_ms':i*500,'boxes':[box(cx),box(.9)]} for i,cx in enumerate(positions)]
    monkeypatch.setattr(localization,'detect_images',lambda *args:frames)
    result=localization.count_passages(tmp_path/'unused.mp4',2500,'How many people crossed?')
    assert result['count']==1
    assert len(result['events'])==1
    assert result['events'][0]['start_seconds']>0
