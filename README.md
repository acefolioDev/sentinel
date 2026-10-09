# Sentinel: video surveillance Q&A

An evidence-first investigation app built from the supplied development blueprint. Upload recorded MP4 footage, ask a natural-language question about one video, optionally add a cropped person/vehicle/object reference image, and inspect the answer's timestamped source clips. Cross-video text and image retrieval, human review, and PDF evidence reports are also available. Metadata and 768-dimensional **Gemini Embedding 2** vectors live in **PostgreSQL + pgvector**; vector ranking runs in the database. **Gemini video understanding** answers video questions.

## Run locally

Requires PostgreSQL with the `vector` extension, Python 3.12, and Node 20+. This workspace already has PostgreSQL 18 and pgvector 0.8.7 installed. Setup also downloads the Apache-2.0 YOLOS-tiny local object detector (about 26 MB of weights) into ignored `data/models/`. The Gemini API key belongs in `.env` (ignored by Git):

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
./scripts/setup.sh
./scripts/dev.sh
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The local development bearer token defaults to `local-demo-token`; change `SURV_ADMIN_TOKEN` in `.env` before exposing the API. The app binds to loopback by default. The API reads `.env` on startup, so restart it after changing the key.

`SURV_DATABASE_URL` defaults to `postgresql:///surveillance_qa`, using the local PostgreSQL Unix socket and your operating system user. `setup.sh` creates that database if absent and enables pgvector. If you use a separate PostgreSQL instance, create the database, enable the `vector` extension, and set the URL in `.env` before starting the API.

## Workflow

1. Add a camera in **Video library**, then upload a short MP4. Source start time is optional; source-relative time is always retained.
2. Watch real decoding and embedding progress in **Operations**. The worker sends sampled frames to Gemini Embedding 2 and writes the returned vectors into pgvector.
3. In **Ask video**, select an indexed recording and ask a question such as “When does the person in this photo appear?” or “How many people crossed the path?” Optionally upload and crop a reference photo. For person and vehicle passage counts, a local detector examines ordered frames, associates nearby boxes into short tracks, and counts tracks crossing an inferred centre line. Each crossing links to a source interval. Counts remain estimates. The API validates intervals against the media duration and extracts local MP4 clips.
4. In the other tabs, search across indexed footage by description, image, or both. These retrieval modes use pgvector to rank authorized frames. Open a clip to see query-class object boxes at sampled moments, compare the keyframe with a reference, then mark evidence possible, rejected, or verified with a note. Verification requires reviewer or admin role.
5. Select evidence and export a PDF with camera, interval, source SHA-256, raw score, and review state.

**Data handling:** Original videos and derived clips stay in `data/`. Sampled CCTV frames, selected reference crops, and text queries are sent to Google's Gemini API for embeddings. Non-counting video questions send the selected video and any optional reference crop to Gemini; larger videos use Google's temporary Files API, which the app deletes after the answer. Common person/vehicle/object class boxes and passage counts run locally with YOLOS-tiny. Unsupported box queries may use Gemini. Vectors and local evidence metadata are stored in PostgreSQL. Use footage and images you are authorized to process with this provider.

## API

OpenAPI documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). All routes except health require `Authorization: Bearer <token>`.

- `POST /api/v1/cameras`, `GET /api/v1/cameras`
- `POST /api/v1/videos`, `GET /api/v1/videos`, `GET /api/v1/jobs/{id}`, `POST /api/v1/jobs/{id}/retry`
- `POST /api/v1/search/text`, `/search/image`, `/search/hybrid`; `GET /api/v1/search/{id}`
- `POST /api/v1/qa` (multipart `video_id`, `question`, optional `file`, `subject_type`, `crop`) answers one video and links evidence clips
- `GET /api/v1/evidence/{id}/keyframe`, `/clip`, `/detections`; `POST /api/v1/evidence/{id}/review`
- `GET /api/v1/references/{id}/image`, `DELETE /api/v1/references/{id}`
- `POST /api/v1/reports`, `GET /api/v1/reports/{id}/download`
- `GET /api/v1/metrics/pipeline`

For camera-scoped tokens, set `SURV_TOKENS_JSON` to a JSON object mapping token to `{ "actor": "name", "role": "investigator", "camera_ids": ["camera-id"] }`. The admin token sees all cameras and can create them. Camera and time filters are applied in PostgreSQL before vector ranking. Media, references, searches, evidence, and reports are access checked.

## Data, model, and limits

`data/` holds private original media, keyframes, reference images, derived clips, and reports. It is excluded from Git. `SURV_DATA_DIR` changes the storage root. `SURV_MAX_UPLOAD_BYTES` defaults to 1 GiB, `SURV_MAX_VIDEO_SECONDS` to 1800, and `SURV_FRAME_INTERVAL_SEC` to 3. The decoder uses presentation timestamps (PTS), including for variable frame rate material. Original media remains immutable. Failed indexing jobs can be retried.

Embeddings use `gemini-embedding-2` at 768 dimensions. The `frames.embedding` column is `vector(768)` with an HNSW cosine index. `model_id` is stored per frame, and searches exclude vectors from another model. Changing the embedding model or dimensions requires reindexing existing footage. The previous SQLite database was empty and is no longer read by the application.

Video Q&A defaults to `SURV_QA_MODEL_ID=gemini-2.5-flash` (configurable in `.env`). Counting uses the local COCO detector at 2 FPS up to two minutes, then 1 FPS up to five minutes. Passage questions count tracked boxes crossing the inferred horizontal or vertical video centre line; that line may differ from a real doorway or road boundary. Other count questions estimate visible object tracks, where occlusion or re-entry can split one subject into multiple tracks. A count for one reference-image identity is marked inconclusive because exact identity continuity cannot be verified. Counts and intervals are estimates. Query-class boxes are sampled about once per second for opened clips, while counted clips reuse the denser boxes from analysis. Their positions and track IDs are approximate, and an absent box does not prove an object was absent. Generic COCO class detection cannot confirm a red car or the exact person in a reference image. The app does not implement calibrated re-identification, cross-camera tracking, RTSP, badge-log fusion, or validated ground-truth temporal detection. This is a local development build, not a production deployment.

## Verify

```bash
PYTHONPATH=apps/api .venv/bin/pytest -q tests
npm --prefix apps/web run build
```

The API test creates a disposable PostgreSQL database and synthetic MP4, mocks Gemini calls, and checks upload → index → pgvector search and video Q&A → clip → review → PDF → reference deletion, including unauthorized camera access. Evaluate retrieval, count, and timestamp quality with labeled, permitted footage before using results operationally.
