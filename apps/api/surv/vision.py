"""Shared CLIP image/text space. Raw cosine scores are retrieval ranks, not confidence."""
import threading
import os
import numpy as np
from PIL import Image
from .config import MODEL_ID

_lock = threading.Lock()
_model = None
_processor = None
_torch = None

def _load():
    global _model, _processor, _torch
    if _model is None:
        with _lock:
            if _model is None:
                os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
                import torch
                from transformers import CLIPModel, CLIPProcessor
                _processor = CLIPProcessor.from_pretrained(MODEL_ID)
                _model = CLIPModel.from_pretrained(MODEL_ID).eval()
                _torch = torch
    return _model, _processor, _torch

def encode_images(images: list[Image.Image]) -> list[list[float]]:
    model, processor, torch = _load()
    values = processor(images=[im.convert('RGB') for im in images], return_tensors='pt')
    with torch.no_grad():
        vectors = model.get_image_features(pixel_values=values['pixel_values'])
        if not isinstance(vectors, torch.Tensor):
            vectors = vectors.pooler_output
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return vectors.cpu().numpy().astype('float32').tolist()

def encode_text(text: str) -> list[float]:
    model, processor, torch = _load()
    values = processor(text=[text], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        vectors = model.get_text_features(input_ids=values['input_ids'], attention_mask=values['attention_mask'])
        if not isinstance(vectors, torch.Tensor):
            vectors = vectors.pooler_output
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return vectors[0].cpu().numpy().astype('float32').tolist()

def rank(query: list[float], candidates: list[dict], limit: int) -> list[tuple[float,dict]]:
    if not candidates:
        return []
    matrix = np.asarray([c['embedding'] for c in candidates], dtype=np.float32)
    scores = matrix @ np.asarray(query, dtype=np.float32)
    order = np.argsort(-scores)[:limit]
    return [(float(scores[i]), candidates[int(i)]) for i in order]
