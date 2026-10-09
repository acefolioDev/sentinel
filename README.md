# Sentinel: video surveillance Q&A

A local, evidence-first investigation MVP built from the supplied development blueprint. Upload recorded MP4 footage, index sampled frames with CLIP, search by text or a cropped reference image, inspect source clips, record a review decision, and export a PDF evidence report.

## Run locally

Requires macOS or Linux, Python 3.12, Node 20+, and internet for the first CLIP model download. On this workspace, setup can use the bundled Python 3.12 runtime.

```bash
./scripts/setup.sh
./scripts/dev.sh
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The local development bearer token defaults to `local-demo-token`. Set `SURV_ADMIN_TOKEN` before starting the API to change it, then enter that token through the profile control in the UI. Keep the API bound to loopback for this local build.

The first indexing or search operation downloads `openai/clip-vit-base-patch32` to the Hugging Face cache. It is several hundred MB and can take time. Subsequent runs reuse the cache. There is no external inference API call after download.

## Workflow

1. Add a camera in **Video library**, then upload a short MP4. Source start time is optional; source-relative time is always retained.
2. Watch the real decoding and embedding progress in **Operations**.
3. Search by description or upload a reference image, drag a subject crop, and choose a subject type. Combined search averages compatible text and image embeddings.
4. Open a candidate clip, compare the keyframe with the reference, then mark it possible, rejected, or reviewed with a note.
5. Select evidence and export a PDF with camera, interval, source SHA-256, raw score, and review state.

## API

OpenAPI documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). All routes except health require `Authorization: Bearer <token>`.

- `POST /api/v1/cameras`, `GET /api/v1/cameras`
- `POST /api/v1/videos`, `GET /api/v1/videos`, `GET /api/v1/jobs/{id}`, `POST /api/v1/jobs/{id}/retry`
- `POST /api/v1/search/text`, `/search/image`, `/search/hybrid`; `GET /api/v1/search/{id}`
- `GET /api/v1/evidence/{id}/keyframe`, `/clip`; `POST /api/v1/evidence/{id}/review`
- `GET /api/v1/references/{id}/image`, `DELETE /api/v1/references/{id}`
- `POST /api/v1/reports`, `GET /api/v1/reports/{id}/download`
- `GET /api/v1/metrics/pipeline`

For camera-scoped investigator tokens, set `SURV_TOKENS_JSON` to a JSON object mapping token to `{ "actor": "name", "role": "investigator", "camera_ids": ["camera-id"] }`. The admin token sees all cameras and can create them. Search candidates are filtered by camera before ranking. Media, references, searches, evidence, and reports are access checked.

## Data and limits

`data/` holds the SQLite database and private original media, keyframes, reference images, derived clips, and reports. It is excluded from Git. `SURV_DATA_DIR` changes the storage root. `SURV_MAX_UPLOAD_BYTES` defaults to 1 GiB, `SURV_MAX_VIDEO_SECONDS` to 1800, and `SURV_FRAME_INTERVAL_SEC` to 3. The video decoder uses presentation timestamps (PTS), including for variable frame rate material. The first frame near each sampling interval is indexed. Original media remains immutable. A failed indexing job can be retried.

This build implements the blueprint's MVP path. It does not implement detector-produced boxes, person/vehicle re-identification, cross-camera tracking, streaming RTSP, badge-log fusion, or validated temporal event detection. Reference crop is user selected. Generic CLIP similarity is an uncalibrated retrieval signal, and no identity or event claim follows from it. Temporal action and sequence questions explicitly return **inconclusive**. Sampling can miss short events. It is intended for local development and evaluation with footage you are allowed to use, not production deployment.

## Verify

```bash
PYTHONPATH=apps/api .venv/bin/pytest -q tests
npm --prefix apps/web run build
```

The API test creates a synthetic MP4 and uses a deterministic encoder to check upload → index → text/image search → clip → review → PDF → reference deletion, including unauthorized camera access. Evaluate real retrieval quality with labeled, permitted footage before using results operationally.
