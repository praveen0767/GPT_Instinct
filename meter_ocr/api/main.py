"""
meter_ocr/api/main.py
FastAPI inference endpoint.

POST /infer
  Input : multipart image upload
  Output: structured OCR JSON

Run:
  uvicorn meter_ocr.api.main:app --host 0.0.0.0 --port 8000 --reload

CRITICAL: The pipeline ALWAYS crops the display region before OCR.
          It NEVER runs OCR on the full image.
"""
import os
import time
import uuid
import json
import logging
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ── Pipeline modules ──────────────────────────────────────────────────────────
from meter_ocr.detectors.meter_detector    import MeterDetector
from meter_ocr.detectors.lcd_detector      import LCDDetector
from meter_ocr.preprocessing.perspective   import correct_perspective
from meter_ocr.preprocessing.glare_removal import remove_glare
from meter_ocr.ocr.ensemble                import OCREnsemble
from meter_ocr.validators.decimal_validator import DecimalValidator
from meter_ocr.validators.domain_rules     import apply_domain_rules
from meter_ocr.calibration.probability_calibrator import FieldCalibrators
from meter_ocr.utils.image_utils           import (
    analyze_image_quality, apply_clahe, upscale_small
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs",         exist_ok=True)
os.makedirs("failed_cases", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/meter_ocr.jsonl"),
    ],
)
_log = logging.getLogger("meter_ocr")

# ── Schemas ───────────────────────────────────────────────────────────────────

class FieldResult(BaseModel):
    value:       str
    probability: float
    candidates:  Optional[List[dict]] = None


class InferResponse(BaseModel):
    image_id:              str
    meter_serial:          FieldResult
    kwh:                   FieldResult
    kvah:                  FieldResult
    md_kw:                 FieldResult
    demand_kva:            FieldResult
    image_quality:         dict
    reason_codes:          List[str]
    qc_flag:               bool
    processing_latency_ms: int


# ── App & model init ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Anti-Gravity Meter OCR API",
    description="""
Production-grade OCR for smart electricity meters.

## Pipeline (always crops display first)
1. **Meter detection** → YOLOv8 + edge fallback
2. **Display/LCD detection** → YOLOv8 + HSV green segmentation
3. **Perspective correction** → four-point transform
4. **Glare removal** → HSV mask + TELEA inpaint
5. **OCR ensemble** → TrOCR (0.5) + PaddleOCR (0.3) + EasyOCR (0.2)
6. **Decimal validator** → regex \\d{3,7}\\.\\d + CV dot detection
7. **Domain rules** → energy meter range constraints
8. **Confidence calibration** → per-field isotonic regression
""",
    version="1.0.0",
)

# Shared model registry
_M: dict = {}


@app.on_event("startup")
def _startup():
    YOLO_METER   = os.environ.get("YOLO_METER_PATH",   None)
    YOLO_DISPLAY = os.environ.get("YOLO_DISPLAY_PATH", None)

    _M["meter_detector"]  = MeterDetector(yolo_path=YOLO_METER)
    _M["lcd_detector"]    = LCDDetector(yolo_path=YOLO_DISPLAY)
    _M["ocr_ensemble"]    = OCREnsemble()
    _M["dec_validator"]   = DecimalValidator()
    _M["calibrators"]     = FieldCalibrators(save_dir="data/calibration")
    _log.info("All models loaded.")


# ── Field layout ─────────────────────────────────────────────────────────────
# For kWh (main reading): use the FULL display crop — single-phase meters
# show one primary value at a time.
# For sub-registers (kvah, md_kw, demand_kva): try the sub-region. If the
# crop is too small/narrow, fall back to the full display.
# Serial: always taken from the bottom 25% of the full meter body image.

_FIELD_LAYOUT = {
    # (y_start%, y_end%, x_start%, x_end%)  — fractions of display_crop size
    # kWh = full display (best accuracy for single-phase meters)
    "kwh":        (0.00, 1.00, 0.00, 1.00),
    # Sub-register rows — present only on multi-tariff / LCD-cycling meters
    "kvah":       (0.55, 1.00, 0.00, 0.55),
    "md_kw":      (0.55, 1.00, 0.45, 1.00),
    "demand_kva": (0.75, 1.00, 0.00, 0.55),
}
# Serial from the meter body (not the display)
_SERIAL_LAYOUT = (0.78, 1.00, 0.05, 0.95)


def _crop_field(image: np.ndarray, y0f, y1f, x0f, x1f) -> np.ndarray:
    h, w = image.shape[:2]
    r = image[max(0,int(h*y0f)):min(h,int(h*y1f)),
              max(0,int(w*x0f)):min(w,int(w*x1f))]
    return r if r.size > 0 else image


def _image_quality_score(iq: dict) -> float:
    score = 1.0
    if iq.get("blur"):  score -= 0.3
    if iq.get("glare"): score -= 0.1
    tilt = abs(iq.get("tilt_deg", 0.0))
    if tilt > 5:  score -= 0.1
    if tilt > 20: score -= 0.2
    return max(0.0, score)


# ── POST /infer ───────────────────────────────────────────────────────────────

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    req_id = str(uuid.uuid4())[:8]
    t0     = time.time()

    # ── 1. Decode ────────────────────────────────────────────────────────────
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")

    # ── 2. Image quality (on full image) ────────────────────────────────────
    iq    = analyze_image_quality(img)
    iq_s  = _image_quality_score(iq)

    reason_codes: List[str] = []
    if iq.get("blur"):          reason_codes.append("BLUR_DETECTED")
    if iq.get("glare"):         reason_codes.append("GLARE_DETECTED")
    if abs(iq.get("tilt_deg",0)) > 10: reason_codes.append("HIGH_TILT")

    if iq.get("not_legible"):
        lat = int((time.time()-t0)*1000)
        _log.warning(json.dumps({"req":req_id,"event":"not_legible","lat_ms":lat}))
        empty = FieldResult(value="N/A", probability=0.0)
        return InferResponse(
            image_id=file.filename, meter_serial=empty, kwh=empty,
            kvah=empty, md_kw=empty, demand_kva=empty,
            image_quality=iq, reason_codes=["NOT_LEGIBLE"],
            qc_flag=True, processing_latency_ms=lat,
        )

    # ── 3. Meter detection → crop meter body ─────────────────────────────────
    meter_crop, _mbbox = _M["meter_detector"].detect(img)

    # ── 4. LCD/display detection → crop display region ───────────────────────
    # CRITICAL: OCR is NEVER run on the full image — always on this crop.
    display_crop, _dbbox = _M["lcd_detector"].detect(meter_crop)

    # ── 5. Preprocess display ─────────────────────────────────────────────────
    display_crop = remove_glare(display_crop)
    display_crop = correct_perspective(display_crop)
    display_crop = apply_clahe(display_crop)
    display_crop = upscale_small(display_crop, min_w=240)

    # ── 6. Per-field OCR ──────────────────────────────────────────────────────
    field_results: dict = {}
    domain_raw:    dict = {}

    # Define all target fields
    target_fields = ["kwh", "kvah", "md_kw", "demand_kva", "meter_serial"]

    for field in target_fields:
        # Determine source and coordinates
        if field == "meter_serial":
            src = meter_crop
            y0f, y1f, x0f, x1f = _SERIAL_LAYOUT
        else:
            src = display_crop
            y0f, y1f, x0f, x1f = _FIELD_LAYOUT.get(field, (0, 1, 0, 1))

        crop = _crop_field(src, y0f, y1f, x0f, x1f)
        crop = upscale_small(crop, min_w=160)

        # ── 6a. OCR ensemble ──────────────────────────────────────────────────
        ocr_res = _M["ocr_ensemble"].run(crop)
        raw_text = ocr_res.get("text", "")
        raw_conf = ocr_res.get("confidence", 0.0)
        
        # Calculate agreement score
        engine_results = ocr_res.get("candidates", [])
        valid_texts = [c["text"] for c in engine_results if c.get("text")]
        agree = 1.0 if ocr_res.get("all_agree") else (
                       0.67 if len(set(valid_texts)) <= 1 and len(valid_texts) >= 2
                       else 0.33)

        # ── 6b. Decimal validator ─────────────────────────────────────────────
        dec_res  = _M["dec_validator"].validate(raw_text, field=field,
                                                display_crop=crop)
        val_text = dec_res["value"]
        dec_conf = dec_res["decimal_conf"]

        # ── 6c. Calibrate ─────────────────────────────────────────────────────
        cal_prob = _M["calibrators"].calibrate(
            field,
            raw_conf=raw_conf,
            engine_agreement=agree,
            decimal_conf=dec_conf,
            image_quality_score=iq_s,
        )

        field_results[field] = {
            "value":       val_text,
            "probability": cal_prob,
            "candidates":  engine_results,
        }
        domain_raw[field] = val_text


    # ── 7. Domain rules ───────────────────────────────────────────────────────
    domain_result = apply_domain_rules(domain_raw)
    for field, validated_val in domain_result["validated"].items():
        if validated_val == "—" and field_results[field]["value"] not in ("—",""):
            reason_codes.append(domain_result["reason_codes"][0]
                                if domain_result["reason_codes"] else f"DOMAIN_REJECT_{field.upper()}")
        field_results[field]["value"] = validated_val

    # ── 8. QC flag ────────────────────────────────────────────────────────────
    low_conf_fields = [f for f, r in field_results.items() if r["probability"] < 0.98]
    if low_conf_fields:
        reason_codes.append("LOW_CONFIDENCE")
    qc_flag = len(reason_codes) > 0

    if qc_flag:
        try:
            cv2.imwrite(f"failed_cases/{req_id}_{file.filename}", img)
        except Exception:
            pass

    # ── 9. Response ───────────────────────────────────────────────────────────
    lat = int((time.time() - t0) * 1000)
    _log.info(json.dumps({
        "req": req_id, "event": "infer_done", "lat_ms": lat,
        "qc": qc_flag, "reason_codes": reason_codes,
        "kwh": field_results.get("kwh",{}).get("value"),
    }))

    def _fr(f: str) -> FieldResult:
        r = field_results.get(f, {})
        return FieldResult(
            value=r.get("value","—"),
            probability=r.get("probability",0.0),
            candidates=r.get("candidates"),
        )

    return InferResponse(
        image_id              = file.filename,
        meter_serial          = _fr("meter_serial"),
        kwh                   = _fr("kwh"),
        kvah                  = _fr("kvah"),
        md_kw                 = _fr("md_kw"),
        demand_kva            = _fr("demand_kva"),
        image_quality         = iq,
        reason_codes          = list(set(reason_codes)),
        qc_flag               = qc_flag,
        processing_latency_ms = lat,
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "version": "meter_ocr_v1.0"}


# ── Redirect / → /ui ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


# ── Serve static frontend at /ui ──────────────────────────────────────────────
# Mount AFTER all routes so the router takes priority over static files.
_frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend"
)
if os.path.isdir(_frontend_dir):
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="ui")
    print(f"Static UI mounted at /ui  ({_frontend_dir})")
else:
    print(f"Warning: frontend/ not found at {_frontend_dir}. /ui will return 404.")

