"""Ground a question in the original video with Gemini video understanding."""
import math
import time
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image

from .config import QA_MODEL_ID
from .vision import _client_instance, _image_part


class Observation(BaseModel):
    start_seconds: float
    end_seconds: float
    description: str = Field(max_length=500)


class VideoAnswer(BaseModel):
    answer_status: Literal['supported', 'possible', 'inconclusive', 'not_found']
    answer: str = Field(max_length=3000)
    count: int | None = Field(default=None, ge=0)
    limitations: str = Field(max_length=1500)
    evidence: list[Observation] = Field(default_factory=list, max_length=20)


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
    client = _client_instance()
    prompt = (
        'You are examining one original CCTV recording for a human investigator. '
        f'The recording lasts {duration_ms / 1000:.3f} seconds. '
        'Answer the user question using only what is visible or audible in this recording. '
        'An optional reference image is a visual example of the subject; compare it to the video, '
        'but do not claim a person or vehicle is the exact same one from appearance alone. '
        'For every positive finding, give source-video start_seconds and end_seconds that bracket '
        'the visible observation. Use video-relative seconds starting at zero. '
        'For counting crossings or passing vehicles, count distinct completed passages across a '
        'single consistent virtual line. Do not count the same subject again just because it remains '
        'visible in adjacent frames. If the line or distinct passages cannot be determined, set '
        'count to null and answer_status to inconclusive. Only populate count when the user '
        'explicitly asks for a count. Every count must have one evidence interval per distinct passage. '
        'For questions about when a reference subject appears, list each distinct possible sighting. '
        'Use supported only when directly visible evidence supports the answer; use possible for '
        'uncertain visual similarity; not_found if no relevant observation is visible; '
        'inconclusive if the video cannot establish the answer. '
        'Never invent timestamps, counts, identities, or events. Keep the answer in the language '
        'of the question. Keep each evidence interval as short as possible, ideally at most 10 seconds. '
        'Give at most 20 evidence intervals and explain any coverage limits. '
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
        contents = [video_part]
        if reference is not None:
            contents.append(_image_part(reference))
        contents.append(prompt)
        response = client.models.generate_content(
            model=QA_MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type='application/json', response_schema=VideoAnswer))
        if not response.text:
            raise RuntimeError('Gemini returned no answer')
        return VideoAnswer.model_validate_json(response.text)
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
