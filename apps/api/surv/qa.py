"""Ground questions in original video; count supported objects with local tracks."""
import math
import re
import time
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr
from PIL import Image

from .config import QA_MODEL_ID
from .vision import _client_instance, _image_part
from .localization import count_passages
from .detector import target_classes


class Observation(BaseModel):
    start_seconds: float
    end_seconds: float
    description: str = Field(max_length=500)


class VideoAnswer(BaseModel):
    answer_status: Literal['supported', 'possible', 'inconclusive', 'not_found']
    answer: str = Field(max_length=3000)
    count: int | None = Field(default=None, ge=0)
    limitations: str = Field(max_length=1500)
    evidence: list[Observation] = Field(default_factory=list)
    _tracking_frames: list[dict] = PrivateAttr(default_factory=list)
    _tracking_interval_ms: int = PrivateAttr(default=1000)
    _count_method: str = PrivateAttr(default='gemini')
    _count_kind: str = PrivateAttr(default='passage')


MAX_CONTINUOUS_COUNT_SECONDS = 300


def is_count_question(question: str) -> bool:
    return bool(re.search(r'\b(how many|number of|count|kitne|kitni|ginti)\b|कितने|कितनी', question, re.IGNORECASE))


def _passage_question(question: str) -> bool:
    return bool(re.search(r'\b(cross(?:ed|ing)?|pass(?:ed|ing)?|enter(?:ed|ing)?|exit(?:ed|ing)?|went through|go through|guza?re?|guzri)\b|गुजरे|गुजरी|पार', question, re.IGNORECASE))


def _part_with_video_metadata(video_part: types.Part, *, fps: float,
                              start: float | None = None, end: float | None = None) -> types.Part:
    metadata = types.VideoMetadata(fps=fps,
                                   start_offset=f'{start:.3f}s' if start is not None else None,
                                   end_offset=f'{end:.3f}s' if end is not None else None)
    return types.Part(inline_data=video_part.inline_data, file_data=video_part.file_data,
                      video_metadata=metadata)


def _generate(client, video_part: types.Part, prompt: str, reference: Image.Image | None) -> VideoAnswer:
    contents = [video_part]
    if reference is not None:
        contents.append(_image_part(reference))
    contents.append(prompt)
    response = client.models.generate_content(
        model=QA_MODEL_ID, contents=contents,
        config=types.GenerateContentConfig(response_mime_type='application/json', response_schema=VideoAnswer))
    if not response.text:
        raise RuntimeError('Gemini returned no answer')
    return VideoAnswer.model_validate_json(response.text)


def _ready_file(client, uploaded):
    deadline = time.monotonic() + 180
    while not uploaded.state or uploaded.state.name == 'PROCESSING':
        if time.monotonic() >= deadline:
            raise TimeoutError('Gemini video processing timed out')
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name != 'ACTIVE':
        raise RuntimeError('Gemini could not process this video')
    return uploaded


def answer_video(path: Path, question: str, duration_ms: int,
                 reference: Image.Image | None = None, subject_type: str | None = None) -> VideoAnswer:
    if is_count_question(question):
        passage=_passage_question(question)
        if duration_ms / 1000 > MAX_CONTINUOUS_COUNT_SECONDS:
            answer=VideoAnswer(answer_status='inconclusive', count=None, evidence=[],
                               answer='A continuous count needs a shorter recording (up to five minutes).',
                               limitations='Split this recording into shorter clips and review each boundary.')
            answer._count_method='local-tracker'
            return answer
        if reference is not None:
            answer=VideoAnswer(answer_status='inconclusive',count=None,evidence=[],
                               answer='A count for this specific reference subject cannot be verified by the local tracker.',
                               limitations='The local detector tracks object classes but cannot establish that a person or vehicle matches the reference image.')
            answer._count_method='local-tracker'
            return answer
        if not target_classes(question, subject_type):
            answer=VideoAnswer(answer_status='inconclusive',count=None,evidence=[],
                               answer='A continuous count is unavailable for the requested object type.',
                               limitations='Local tracking currently supports people, vehicles, bicycles, bags, cats, and dogs. Specify one of these object types in the question.')
            answer._count_method='local-tracker'
            return answer
        tracked = count_passages(path,duration_ms,question,reference,subject_type,passage=passage)
        count = tracked['count']
        answer = VideoAnswer(answer_status='inconclusive' if count is None else ('possible' if count else 'not_found'),
                             count=count,
                             answer=(f'{count} tracked object(s) crossed the inferred video centre line.' if count is not None and passage
                                     else f'{count} relevant object track(s) were visible.' if count is not None
                                     else 'A reliable count could not be established from tracked frames.'),
                             limitations=tracked['reason'],
                             evidence=[Observation.model_validate(item) for item in tracked['events']])
        answer._tracking_frames=tracked['frames']
        answer._tracking_interval_ms=500 if duration_ms <= 120000 else 1000
        answer._count_method='local-tracker'
        answer._count_kind='passage' if passage else 'visible'
        return answer
    client = _client_instance()
    prompt = (
        'You are examining one original CCTV recording for a human investigator. '
        f'The recording lasts {duration_ms / 1000:.3f} seconds. '
        'Answer the user question using only what is visible or audible in this recording. '
        'An optional reference image is a visual example of the subject; compare it to the video, '
        'but do not claim a person or vehicle is the exact same one from appearance alone. '
        'For every positive finding, give source-video start_seconds and end_seconds that bracket '
        'the visible observation. Use video-relative seconds starting at zero. '
        'Only populate count when the user explicitly asks for a count. '
        'For questions about when a reference subject appears, list each distinct possible sighting. '
        'Use supported only when directly visible evidence supports the answer; use possible for '
        'uncertain visual similarity; not_found if no relevant observation is visible; '
        'inconclusive if the video cannot establish the answer. '
        'Never invent timestamps, counts, identities, or events. Keep the answer in the language '
        'of the question. Keep each evidence interval as short as possible, ideally at most 10 seconds. '
        'Give at most 100 evidence intervals and explain any coverage limits. '
        f'Reference subject type: {subject_type or "none"}.\nQuestion: {question}'
    )
    uploaded = None
    try:
        if path.stat().st_size < 15 * 1024 * 1024:
            video_part = types.Part.from_bytes(data=path.read_bytes(), mime_type='video/mp4')
        else:
            uploaded = client.files.upload(file=str(path), config={'mime_type': 'video/mp4'})
            uploaded = _ready_file(client, uploaded)
            video_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type or 'video/mp4')
        return _generate(client,_part_with_video_metadata(video_part,fps=2),prompt,reference)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def validated_observations(answer: VideoAnswer, duration_ms: int) -> list[tuple[int, int, str, int, int]]:
    """Discard impossible model timestamps before they can become evidence links."""
    result = []
    for item in answer.evidence:
        start, end = item.start_seconds, item.end_seconds
        if not all(math.isfinite(v) for v in (start, end)):
            continue
        if start < 0 or start >= duration_ms / 1000 or end <= start or end > duration_ms / 1000 + 1:
            continue
        observed_start_ms = int(start * 1000)
        observed_end_ms = min(duration_ms, int(end * 1000))
        start_ms = max(0, observed_start_ms - 1500)
        end_ms = min(duration_ms, max(observed_end_ms + 1500, start_ms + 1000))
        if any(observed_start_ms == old_observed_start and observed_end_ms == old_observed_end and item.description == old_description
               for _, _, old_description, old_observed_start, old_observed_end in result):
            continue
        result.append((start_ms, end_ms, item.description, observed_start_ms, observed_end_ms))
    return result
