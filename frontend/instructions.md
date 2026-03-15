# Anti-Gravity OCR — Frontend Setup & Deployment Instructions

## Prerequisites

```
Python 3.10+  •  pip  •  (optional) Docker 20+  •  (optional) Streamlit
```

---

## 1. Local Run (Recommended)

```bash
# From repo root
cd d:\GPT_instinct

# Activate virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

# Start the FastAPI backend (serves API + static UI)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open in your browser:
- **Static UI:** http://localhost:8000/ui
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## 2. Streamlit Demo (Optional — Quick Stakeholder Demo)

Open a **second** terminal while the FastAPI server is running:

```bash
cd d:\GPT_instinct
.venv\Scripts\activate
pip install streamlit requests    # only needed once
streamlit run frontend/streamlit_demo.py
```

Open http://localhost:8501  
The demo talks to the FastAPI backend at `http://localhost:8000` (configurable in the sidebar).

---

## 3. Run Tests

```bash
cd d:\GPT_instinct
.venv\Scripts\activate
pytest tests/test_ui_proxy.py -v
```

Expected output:
```
tests/test_ui_proxy.py::TestUIInferProxy::test_valid_png_returns_200              PASSED
tests/test_ui_proxy.py::TestUIInferProxy::test_valid_png_response_has_required_keys PASSED
tests/test_ui_proxy.py::TestUIInferProxy::test_valid_png_kwh_field_structure       PASSED
tests/test_ui_proxy.py::TestUIInferProxy::test_invalid_file_type_is_handled        PASSED
tests/test_ui_proxy.py::TestUIInferProxy::test_empty_file_is_handled               PASSED
tests/test_ui_proxy.py::TestUIInferProxy::test_processing_latency_ms_is_positive   PASSED
tests/test_ui_proxy.py::TestStaticMount::test_ui_root_returns_html                 PASSED
tests/test_ui_proxy.py::TestStaticMount::test_health_endpoint_still_works          PASSED
```

---

## 4. Docker — Single Image (API + UI)

```bash
# Build
docker build -t agm-ocr:latest .

# Run (API + static UI inside same container)
docker run -p 8000:8000 agm-ocr:latest
```

Then open http://localhost:8000/ui

> The `Dockerfile` already copies the entire project including `frontend/`. The `app.py` auto-detects and mounts `frontend/` at `/ui` on startup.

---

## 5. Docker Compose (Full Stack)

```bash
docker-compose up -d
```

Services started:
- `api` — FastAPI OCR backend on port 8000
- `redis` — Task queue broker
- `minio` — Artifact object storage
- `worker` — Celery worker for background QC tasks

---

## 6. Free Internet Deployment

### Option A — Render.com (FastAPI backend + static UI)

1. Push your repo to GitHub.
2. Go to https://render.com → **New Web Service**.
3. Connect your GitHub repo.
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
5. Add env var: `PORT=10000` (Render assigns automatically).
6. Deploy. Access `/ui` at your Render URL.

> ⚠️ Free tier has 512 MB RAM — sufficient for EasyOCR alone; TrOCR + PaddleOCR may require the $7/month Starter plan.

### Option B — Streamlit Community Cloud (Demo only)

1. Push `frontend/streamlit_demo.py` to GitHub.
2. Go to https://share.streamlit.io → **New app**.
3. Point to `frontend/streamlit_demo.py`.
4. Add a `requirements_streamlit.txt` with just:
   ```
   streamlit
   requests
   pandas
   ```
5. In the Streamlit app sidebar, set **Backend URL** to your Render API URL.

### Option C — Hugging Face Spaces (Docker)

1. Create a new Space with **Docker** runtime.
2. Upload your `Dockerfile` and source.
3. In `Dockerfile`, change the CMD port to `7860` (HF default):
   ```dockerfile
   CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

---

## 7. File Structure

```
frontend/
├── index.html          ← Static UI (drag-drop, result cards, Tailwind CDN)
├── streamlit_demo.py   ← Streamlit demo app
└── instructions.md     ← This file

api/
└── app.py              ← FastAPI (now includes /ui/infer proxy + static mount)

tests/
└── test_ui_proxy.py    ← pytest tests for /ui/infer and /ui static route
```

---

## 8. API Quick Reference

| Endpoint         | Method | Description |
|------------------|--------|-------------|
| `/infer`         | POST   | Main OCR endpoint (multipart image) |
| `/ui/infer`      | POST   | Proxy (same as /infer + structured logging) |
| `/ui`            | GET    | Static demo UI (index.html) |
| `/health`        | GET    | Health check + version |
| `/docs`          | GET    | Interactive Swagger UI |
