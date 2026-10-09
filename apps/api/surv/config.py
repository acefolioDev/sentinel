from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / '.env', override=False)
DATA_DIR = Path(os.getenv('SURV_DATA_DIR', ROOT / 'data')).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
for folder in ('videos', 'frames', 'references', 'clips', 'reports'):
    (DATA_DIR / folder).mkdir(exist_ok=True)
DATABASE_URL = os.getenv('SURV_DATABASE_URL', 'postgresql:///surveillance_qa')
ADMIN_TOKEN = os.getenv('SURV_ADMIN_TOKEN', 'local-demo-token')
MAX_UPLOAD_BYTES = int(os.getenv('SURV_MAX_UPLOAD_BYTES', str(1024**3)))
FRAME_INTERVAL_SEC = float(os.getenv('SURV_FRAME_INTERVAL_SEC', '3'))
MAX_VIDEO_SECONDS = int(os.getenv('SURV_MAX_VIDEO_SECONDS', '1800'))
MODEL_ID = os.getenv('SURV_MODEL_ID', 'gemini-embedding-2')
QA_MODEL_ID = os.getenv('SURV_QA_MODEL_ID', 'gemini-2.5-flash')
EMBEDDING_DIMENSIONS = 768
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
