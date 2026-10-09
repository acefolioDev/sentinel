"""Cache the pinned Apache-2.0 YOLOS model for offline local inference."""
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
target = root / 'data' / 'models' / 'yolos-tiny'
snapshot_download(
    repo_id='hustvl/yolos-tiny',
    revision='da86128da961944dd8e33bb7c1baea46ed0a4753',
    allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'],
    local_dir=str(target),
)
print(f'Local detector ready at {target}')
