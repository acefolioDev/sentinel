"""Query-specific boxes for sampled frames of one evidence clip."""
import math
from pathlib import Path

import av
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from .config import QA_MODEL_ID
from .detector import MODEL_ID as LOCAL_MODEL_ID, detect_images, target_classes
from .vision import _client_instance, _image_part


class BoxResponse(BaseModel):
    box_2d: list[int]
    label: str


class FrameResponse(BaseModel):
    frame_index: int
    boxes: list[BoxResponse] = Field(default_factory=list)


class DetectionResponse(BaseModel):
    frames: list[FrameResponse] = Field(default_factory=list)


def _sample_frames(path: Path, start_ms: int, end_ms: int, keyframe_ms: int,
                   interval_ms: int = 1000, max_frames: int = 17):
    targets = list(range(start_ms, end_ms, interval_ms))
    if keyframe_ms >= start_ms and keyframe_ms < end_ms:
        targets.append(keyframe_ms)
    targets = sorted(set(targets))
    if len(targets) > max_frames:
        step = math.ceil(len(targets) / max_frames)
        targets = targets[::step]
    picked = []
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == 'video')
        container.seek(max(0, int(start_ms / 1000 / stream.time_base) - 1), stream=stream, backward=True)
        target_index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_ms = int(float(frame.pts * frame.time_base) * 1000)
            if pts_ms > end_ms + 500 or target_index >= len(targets):
                break
            if pts_ms + 100 < targets[target_index]:
                continue
            image = frame.to_image().convert('RGB')
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            picked.append((pts_ms, image))
            while target_index < len(targets) and targets[target_index] <= pts_ms + 100:
                target_index += 1
    return picked


def _iou(a, b):
    x1, y1, x2, y2 = max(a['x1'], b['x1']), max(a['y1'], b['y1']), min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a['x2']-a['x1']) * (a['y2']-a['y1'])
    area_b = (b['x2']-b['x1']) * (b['y2']-b['y1'])
    return intersection / max(1e-9, area_a+area_b-intersection)


def _assign_tracks(frames):
    """Short-gap, one-to-one associations; IDs are visual aids, not identity claims."""
    tracks = {}
    next_id = 1
    for frame in frames:
        claimed = set()
        for box in frame['boxes']:
            best_id, best_score = None, -1.0
            cx, cy = (box['x1']+box['x2'])/2, (box['y1']+box['y2'])/2
            for track_id, (last_ms, last_box) in tracks.items():
                if track_id in claimed or frame['pts_ms']-last_ms > 2200:
                    continue
                if box['label'] != last_box['label']:
                    continue
                distance = math.hypot(cx-(last_box['x1']+last_box['x2'])/2,
                                      cy-(last_box['y1']+last_box['y2'])/2)
                overlap = _iou(box,last_box)
                if overlap < 0.12 and distance > 0.18:
                    continue
                score = overlap - distance * 0.3
                if score > best_score:
                    best_id,best_score = track_id,score
            if best_id is None:
                best_id = next_id
                next_id += 1
            box['track_id'] = best_id
            claimed.add(best_id)
            tracks[best_id] = (frame['pts_ms'],box)


def _detect_sampled(sampled, query: str, reference: Image.Image | None):
    if not sampled:
        return []
    prompt = (
        'These are numbered frames sampled in time order from one CCTV evidence clip. '
        f'Question or visual target: {query}. Detect and draw boxes ONLY around objects relevant to that target. '
        'If the question asks for people or vehicles crossing, box each visible person or vehicle. '
        'If a reference image is supplied, box only visually similar candidates; similarity does not prove identity. '
        'Return one frames entry for every numbered frame, including empty boxes when nothing matches. '
        'Use box_2d = [ymin, xmin, ymax, xmax] with integers normalized to 0..1000. '
        'Do not invent objects or boxes outside the frame. Keep labels short. '
        'These frames are sampled, so do not infer a continuous track from missing frames.'
    )
    contents = [prompt]
    if reference is not None:
        contents.extend(['Reference image:', _image_part(reference)])
    for index, (pts_ms, image) in enumerate(sampled):
        contents.extend([f'Frame {index} at {pts_ms / 1000:.3f} seconds:', _image_part(image)])
    response = _client_instance().models.generate_content(
        model=QA_MODEL_ID, contents=contents,
        config=types.GenerateContentConfig(response_mime_type='application/json',
                                           response_schema=DetectionResponse, temperature=0))
    if not response.text:
        raise RuntimeError('Gemini returned no object boxes')
    parsed = DetectionResponse.model_validate_json(response.text)
    by_index = {item.frame_index: item for item in parsed.frames if 0 <= item.frame_index < len(sampled)}
    frames = []
    for index, (pts_ms, _) in enumerate(sampled):
        boxes = []
        for item in (by_index[index].boxes[:30] if index in by_index else []):
            if len(item.box_2d) != 4:
                continue
            y1,x1,y2,x2 = item.box_2d
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                continue
            if (x2-x1)*(y2-y1) < 25:
                continue
            boxes.append({'x1': x1/1000, 'y1': y1/1000, 'x2': x2/1000, 'y2': y2/1000,
                          'label': item.label[:80]})
        frames.append({'pts_ms': pts_ms, 'boxes': boxes})
    return frames


def localize_clip(path: Path, start_ms: int, end_ms: int, keyframe_ms: int,
                  query: str, reference: Image.Image | None = None) -> dict:
    sampled = _sample_frames(path,start_ms,end_ms,keyframe_ms)
    local = reference is None and bool(target_classes(query))
    frames = detect_images(sampled,query) if local else _detect_sampled(sampled,query,reference)
    _assign_tracks(frames)
    return {'frames': frames, 'sample_interval_ms': 1000,
            'model_id': LOCAL_MODEL_ID if local else QA_MODEL_ID,
            'note': ('Local boxes identify the object class, not a particular reference-image identity.'
                     if local and reference is not None else
                     'Local object boxes and short-gap tracks are approximate.' if local else
                     'Gemini boxes are sampled observations, not continuous detections.')}


def count_passages(path: Path, duration_ms: int, query: str,
                   reference: Image.Image | None = None, subject_type: str | None = None,
                   passage: bool = True) -> dict:
    """Count passages or visible tracks from actual ordered frame detections."""
    interval_ms = 500 if duration_ms <= 120000 else 1000
    if duration_ms > 300000:
        return {'count': None, 'events': [], 'frames': [],
                'reason': 'Continuous frame tracking is limited to five-minute recordings.'}
    sampled = _sample_frames(path,0,duration_ms,-1,interval_ms=interval_ms,max_frames=650)
    if len(sampled) < 3:
        return {'count': None, 'events': [], 'frames': [], 'reason': 'Too few decodable frames for continuity.'}
    frames = detect_images(sampled,query,subject_type)
    _assign_tracks(frames)
    tracks = {}
    for frame in frames:
        for box in frame['boxes']:
            tracks.setdefault(box['track_id'],[]).append((frame['pts_ms'],box))
    if not passage:
        events=[]
        for track_id,points in tracks.items():
            if len(points)<2:
                continue
            first=points[0][0]
            label=points[0][1]['label'] or 'Object'
            events.append({'start_seconds':max(0,(first-interval_ms/2)/1000),
                           'end_seconds':min(duration_ms/1000,(first+interval_ms*1.5)/1000),
                           'description':f'{label.capitalize()} track #{track_id} was visible.'})
        events.sort(key=lambda event:event['start_seconds'])
        if len(events)>100:
            return {'count':None,'events':events[:100],'frames':frames,'reason':'More than 100 tracks need review.'}
        return {'count':len(events),'events':events,'frames':frames,
                'reason':f'Counted {len(events)} visible detector tracks at {interval_ms/1000:g}s sampling. A subject may create more than one track after occlusion or re-entry.'}
    viable = []
    for track_id,points in tracks.items():
        if len(points) < 3 or points[-1][0]-points[0][0] < 1000:
            continue
        first,last = points[0][1],points[-1][1]
        dx = (last['x1']+last['x2']-first['x1']-first['x2'])/2
        dy = (last['y1']+last['y2']-first['y1']-first['y2'])/2
        if max(abs(dx),abs(dy)) >= 0.1:
            viable.append((track_id,points,dx,dy))
    if not viable:
        return {'count': None, 'events': [], 'frames': frames,
                'reason': 'No object could be followed across enough consecutive frames.'}
    x_motion = sum(abs(item[2]) for item in viable)
    y_motion = sum(abs(item[3]) for item in viable)
    axis = 'x' if x_motion >= y_motion else 'y'
    events = []
    for track_id,points,_,_ in viable:
        def coordinate(box):
            return (box[f'{axis}1']+box[f'{axis}2'])/2
        first,last=coordinate(points[0][1]),coordinate(points[-1][1])
        if not ((first <= .45 and last >= .55) or (first >= .55 and last <= .45)):
            continue
        crossing_index = next((i for i in range(1,len(points))
                               if (coordinate(points[i-1][1])-.5)*(coordinate(points[i][1])-.5) <= 0),None)
        if crossing_index is None:
            continue
        before,after = points[crossing_index-1][0],points[crossing_index][0]
        direction = ('left to right' if first < last else 'right to left') if axis=='x' else ('top to bottom' if first < last else 'bottom to top')
        label = points[crossing_index][1]['label'] or 'Object'
        events.append({'start_seconds':max(0,(before-interval_ms/2)/1000),
                       'end_seconds':min(duration_ms/1000,(after+interval_ms/2)/1000),
                       'description':f'{label.capitalize()} track #{track_id} crossed the video centre line {direction}.'})
    events.sort(key=lambda event:event['start_seconds'])
    if len(events)>100:
        return {'count':None,'events':events[:100],'frames':frames,'reason':'More than 100 passages need review.'}
    note=(' Reference-image similarity is not checked by the local class detector.' if reference is not None else '')
    return {'count':len(events),'events':events,'frames':frames,'axis':axis,
            'reason':f'Counted distinct tracked boxes crossing the video centre {"vertical" if axis=="x" else "horizontal"} line. Sampling is {interval_ms/1000:g} seconds; occlusions or track switches may affect the result.{note}'}
