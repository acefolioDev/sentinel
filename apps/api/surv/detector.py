"""Local COCO object detection for frame-linked boxes and continuity counts."""
import re
import threading

from .config import DATA_DIR

MODEL_ID = 'hustvl/yolos-tiny'
MODEL_PATH = DATA_DIR / 'models' / 'yolos-tiny'
_lock = threading.Lock()
_processor = None
_model = None


def target_classes(query: str, subject_type: str | None = None) -> set[str]:
    text = query.lower()
    if subject_type == 'person' or re.search(r'\b(people|person|persons|pedestrian|man|men|woman|women|log|banda|banday)\b',text):
        return {'person'}
    if subject_type == 'vehicle' or re.search(r'\b(vehicles?|traffic|gaari|gari|cars?)\b',text):
        return {'car','truck','bus','motorcycle'} if 'car' not in text else {'car'}
    aliases = {'truck':{'truck'},'bus':{'bus'},'motorcycle':{'motorcycle'},
               'bicycle':{'bicycle'},'bike':{'bicycle','motorcycle'},
               'dog':{'dog'},'cat':{'cat'},'bag':{'handbag','backpack','suitcase'},
               'suitcase':{'suitcase'}}
    for word, classes in aliases.items():
        if re.search(rf'\b{word}s?\b',text):
            return classes
    return set()


def _load_model():
    global _processor, _model
    with _lock:
        if _model is None:
            if not (MODEL_PATH / 'model.safetensors').exists():
                raise RuntimeError('Local detector model is missing. Run ./scripts/setup.sh to download it.')
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            torch.set_num_threads(min(4,torch.get_num_threads()))
            _processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH),local_files_only=True,use_fast=False)
            _model = AutoModelForObjectDetection.from_pretrained(str(MODEL_PATH),
                                                                 local_files_only=True,use_safetensors=True).eval()
    return _processor,_model


def detect_images(sampled, query: str, subject_type: str | None = None):
    """Return normalized boxes only for locally supported query classes."""
    classes = target_classes(query,subject_type)
    if not classes:
        raise ValueError('The local detector needs a recognizable object class such as person, car, vehicle, or bag.')
    import torch
    processor,model = _load_model()
    frames = []
    for pts_ms,image in sampled:
        inputs=processor(images=image,return_tensors='pt')
        with torch.inference_mode():
            outputs=model(**inputs)
        predictions=processor.post_process_object_detection(
            outputs,threshold=.45,target_sizes=torch.tensor([image.size[::-1]]))[0]
        boxes=[]
        for score,label,coordinates in zip(predictions['scores'],predictions['labels'],predictions['boxes']):
            name=model.config.id2label[int(label)].lower()
            if name not in classes:
                continue
            x1,y1,x2,y2=(float(value) for value in coordinates)
            x1,y1,x2,y2=max(0,x1/image.width),max(0,y1/image.height),min(1,x2/image.width),min(1,y2/image.height)
            if x2-x1 < .005 or y2-y1 < .005:
                continue
            boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'label':name,'score':round(float(score),3)})
        frames.append({'pts_ms':pts_ms,'boxes':boxes})
    return frames
