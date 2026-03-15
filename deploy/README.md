# Anti-Gravity OCR — VPC Deployment Guide

## Recommended Infrastructure

| Component | Spec | Notes |
|---|---|---|
| API server (CPU) | 4 vCPU, 8 GB RAM | Serves ~2–4 req/s; p50 latency ~2s |
| API server (GPU) | 1× NVIDIA V100/A10 | p50 latency < 300ms; preferred |
| Object storage | MinIO or S3-compatible | Stores artifact crops, debug images |
| Redis (optional) | 1 GB | Task queue for async QC push |

---

## Quick Start (Docker Compose)

```bash
git clone https://github.com/your-org/Insitinct_GPT_OCr.git
cd Insitinct_GPT_OCr

# CPU stack
docker compose --profile cpu up -d

# GPU stack (requires nvidia-container-toolkit)
docker compose --profile gpu up -d
```

Access the API at `http://<host>:8000`  
Access the UI  at `http://<host>:8000/ui`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TROCR_MODEL` | `microsoft/trocr-base-printed` | HuggingFace model id or local path |
| `YOLO_MODEL_PATH` | `models/yolov8_detector.pt` | YOLOv8 display detector checkpoint |
| `DECIMAL_MODEL_PATH` | `models/weights/decimal_cnn_best.pt` | Decimal CNN checkpoint |
| `CALIBRATION_DIR` | `data/calibration` | Directory for .pkl calibrators |
| `LOG_DIR` | `logs` | JSON-line log directory |
| `FAILED_CASES_DIR` | `failed_cases` | Failed image persistence |

---

## Network & Security

1. **No external API calls** — all inference is local. To verify:
   ```bash
   docker run --network none agm-ocr-cpu:latest curl https://example.com
   # → curl: (6) Could not resolve host
   ```

2. **Firewall** — expose port `8000` only to your internal VPC CIDR.
   ```
   aws ec2 authorize-security-group-ingress \
     --group-id sg-XXXX \
     --protocol tcp --port 8000 \
     --cidr 10.0.0.0/8
   ```

3. **Secrets** — no API keys required. If a local vLLM Ollama endpoint is used,
   set `OLLAMA_URL=http://ollama-service:11434`.

4. **Data Privacy** — images are processed in memory only; crops written to
   `debug_artifacts/` and `failed_cases/` on the same host. Mount these as
   encrypted EBS volumes in AWS.

---

## GPU Setup (AWS)

```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run GPU image
docker run --gpus all -p 8000:8000 agm-ocr-gpu:latest
```

---

## Model Persistence Across Restarts

Mount model directories as persistent volumes:

```yaml
# docker-compose.yml (added to existing)
volumes:
  - ./models:/app/models
  - ./data/calibration:/app/data/calibration
  - ./failed_cases:/app/failed_cases
  - ./logs:/app/logs
```

---

## Reproducing Training Inside VPC

```bash
# 1. Copy labelled data to the server
scp -r data/images user@host:/app/data/images
scp data/labels.csv user@host:/app/data/labels.csv

# 2. Run full pipeline (trains models, runs benchmark, writes report)
docker exec -it agm-ocr bash -c \
    "./ci/run_full_pipeline.sh --data data/images --labels data/labels.csv --out reports/"
```

---

## Audit Checklist

| Item | Command | Expected |
|---|---|---|
| API healthy | `curl http://host:8000/health` | `{"status":"healthy","version":"agm_ocr_v2.0"}` |
| No external calls | `tcpdump -i eth0 port 443` during inference | No traffic |
| kWh is numeric | `curl -X POST .../infer -F file=@meter.png \| python -c "import sys,json; j=json.load(sys.stdin); print(j['kwh']['value'])"` | Digits, not 'Int' |
| All tests pass | `pytest tests/ -v` | All green |
| Benchmark pass | `./ci/run_benchmark.sh` | Exit 0 |
