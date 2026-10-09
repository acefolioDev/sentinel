"""Gemini Embedding 2: text, images, and combined queries share one vector space."""
import io
import math
from PIL import Image
from google import genai
from google.genai import types
from .config import EMBEDDING_DIMENSIONS, GEMINI_API_KEY, MODEL_ID

_client = None

class ProviderNotConfigured(RuntimeError):
    pass

def configured():
    return bool(GEMINI_API_KEY)

def _client_instance():
    global _client
    if not GEMINI_API_KEY:
        raise ProviderNotConfigured('Set GEMINI_API_KEY in .env to enable Gemini indexing and search')
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

def _image_part(image: Image.Image):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=85)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type='image/jpeg')

def _vectors(response, expected):
    embeddings = response.embeddings or []
    if len(embeddings) != expected:
        raise RuntimeError(f'Gemini returned {len(embeddings)} embeddings; expected {expected}')
    result = []
    for item in embeddings:
        values = item.values or []
        if len(values) != EMBEDDING_DIMENSIONS or not all(math.isfinite(x) for x in values):
            raise RuntimeError('Gemini returned an invalid embedding')
        result.append([float(x) for x in values])
    return result

def encode_images(images: list[Image.Image]) -> list[list[float]]:
    if not images:
        return []
    client = _client_instance()
    vectors = []
    # Gemini Embedding 2 allows up to six images per request. Content wrappers
    # request separate embeddings rather than one aggregated embedding.
    for start in range(0, len(images), 6):
        chunk = images[start:start+6]
        contents = [types.Content(parts=[_image_part(image)]) for image in chunk]
        response = client.models.embed_content(
            model=MODEL_ID, contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS))
        vectors.extend(_vectors(response, len(chunk)))
    return vectors

def encode_text(query: str) -> list[float]:
    response = _client_instance().models.embed_content(
        model=MODEL_ID, contents=f'task: search result | query: {query}',
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS))
    return _vectors(response, 1)[0]

def encode_hybrid(image: Image.Image, query: str) -> list[float]:
    # The SDK aggregates text and image parts in one multimodal embedding.
    response = _client_instance().models.embed_content(
        model=MODEL_ID, contents=[query, _image_part(image)],
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS))
    return _vectors(response, 1)[0]
