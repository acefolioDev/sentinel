from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("SURV_DATA_DIR", ROOT / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
for folder in ("videos", "frames", "references", "clips", "reports"):
    (DATA_DIR / folder).mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "surv.sqlite3"
ADMIN_TOKEN = os.getenv("SURV_ADMIN_TOKEN", "local-demo-token")
MAX_UPLOAD_BYTES = int(os.getenv("SURV_MAX_UPLOAD_BYTES", str(1024**3)))
FRAME_INTERVAL_SEC = float(os.getenv("SURV_FRAME_INTERVAL_SEC", "3"))
MAX_VIDEO_SECONDS = int(os.getenv("SURV_MAX_VIDEO_SECONDS", "1800"))
MODEL_ID = os.getenv("SURV_MODEL_ID", "openai/clip-vit-base-patch32")
