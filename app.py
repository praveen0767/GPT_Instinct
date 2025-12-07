"""
app.py - Flask backend for MeterVision QC Portal (single-file).
Place your 'index.html' at the same path as when you uploaded it (this script expects it at /mnt/data/index.html by default).
It will serve the frontend, inject JS to wire buttons to webhooks, and provide endpoints for list/upload/process.

Author: Generated assistant (adapt for your environment)
"""

import os
import io
import base64
import json
import glob
import uuid
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, abort, Response
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np

# Optional imports (ultralytics). If not installed or model not present, fallback to Tesseract crop-based OCR is used.
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

import pytesseract

# ----------------- CONFIG -----------------
BASE_DIR = Path.cwd()
# Path to the uploaded index.html (you provided this file). Adjust if different.
FRONTEND_HTML = Path("/mnt/data/index.html")  # uses your uploaded index.html
# Folders used by the backend
SAMPLES_DIR = BASE_DIR / "dataset_100"       # sample images (you already made)
UPLOADS_DIR = BASE_DIR / "uploads"           # incoming uploads
VIS_DIR = BASE_DIR / "dataset_100_vis"       # optional visuals
LABELS_DIR = BASE_DIR / "dataset_100_labels" # optional labels

# YOLO model (optional) - set path to your trained detector here if available
YOLO_WEIGHTS = BASE_DIR / "runs/detect/meter_field_detector/weights/best.pt"

# Tesseract bin (Windows users change path). If tesseract is on PATH, no change needed.
# Example Windows path: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Example Linux: omit or set to '/usr/bin/tesseract'
TESSERACT_CMD = None  # set to None to use default; or e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Region templates (fractions); used by fallback detector to crop fields.
# Tune if your tile layout differs.
REGIONS = {
    0: ("DISPLAY_REGION", (0.05, 0.25, 0.95, 0.90)),  # display
    1: ("SERIAL_REGION",  (0.10, 0.02, 0.90, 0.18)),  # serial nameplate
    2: ("KWH_REGION",     (0.18, 0.30, 0.80, 0.54)),  # kWh digits
    3: ("MD_KW_REGION",   (0.58, 0.62, 0.92, 0.82)),  # md_kW digits (right area)
}
# decimal detection offset (relative inside parent region)
DECIMAL_PARAMS = {
    2: (0.85, 0.52, 0.035, 0.12),  # kWh decimal (x_rel,y_rel,w_rel,h_rel)
    3: (0.82, 0.50, 0.06, 0.18),   # MD decimal
}
# thresholds
BLUR_VAR_THRESH = 80.0     # below -> blurry
GLARE_PIXEL_RATIO = 0.02   # percent of pixels > 250 -> glare
# Confidence baseline
CONF_THRESH = 0.5

# Make sure folders exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Load YOLO model if available
yolo_model = None
if HAS_ULTRALYTICS and YOLO_WEIGHTS.exists():
    try:
        yolo_model = YOLO(str(YOLO_WEIGHTS))
        print("Loaded YOLO model:", YOLO_WEIGHTS)
    except Exception as e:
        print("Could not load YOLO model, will fallback to template OCR. Error:", e)
        yolo_model = None
else:
    if HAS_ULTRALYTICS:
        print("Ultralytics installed but weights not found at", YOLO_WEIGHTS)
    else:
        print("Ultralytics not installed; using fallback OCR (Tesseract)")

# Helper utilities
def save_image_pil(img: Image.Image, dest_path: Path, quality=90):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest_path), quality=quality)

def variance_of_laplacian_gray(cv2_gray_image):
    return cv2.Laplacian(cv2_gray_image, cv2.CV_64F).var()

def check_glare(cv2_img_gray):
    # proportion of very bright pixels
    h, w = cv2_img_gray.shape[:2]
    bright = (cv2_img_gray >= 250).sum()
    return (bright / (h * w)) > GLARE_PIXEL_RATIO

def ocr_numeric_crop(img_bgr):
    """Run Tesseract OCR on a numeric crop and return (text, confidence)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # thresholding
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # use image_to_data to extract confidences
    custom_oem_psm = "--psm 7 -c tessedit_char_whitelist=0123456789."
    try:
        data = pytesseract.image_to_data(th, config=custom_oem_psm, output_type=pytesseract.Output.DICT)
        texts = [t for t in data['text'] if t.strip() != ""]
        if texts:
            # choose concatenated text
            text = "".join(texts).strip()
            # confidence: average of numeric token confidences (filter -1)
            confs = [int(c) for c in data['conf'] if c != '-1' and c != '']
            conf = float(np.mean(confs))/100.0 if confs else 0.0
        else:
            text = ""
            conf = 0.0
    except Exception as e:
        text = ""
        conf = 0.0
    return text, conf

def ocr_serial_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    conf_str = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/_"
    try:
        data = pytesseract.image_to_data(th, config=conf_str, output_type=pytesseract.Output.DICT)
        texts = [t for t in data['text'] if t.strip() != ""]
        text = " ".join(texts).strip() if texts else ""
        confs = [int(c) for c in data['conf'] if c != '-1' and c != '']
        conf = float(np.mean(confs))/100.0 if confs else 0.0
    except Exception as e:
        text = ""
        conf = 0.0
    return text, conf

def rect_from_frac(w, h, frac):
    fx1, fy1, fx2, fy2 = frac
    x1 = int(fx1 * w); y1 = int(fy1 * h); x2 = int(frac[2] * w); y2 = int(frac[3] * h)
    return x1, y1, x2, y2

def crop_from_frac(img_bgr, frac):
    h, w = img_bgr.shape[:2]
    x1 = int(frac[0]*w); y1 = int(frac[1]*h); x2 = int(frac[2]*w); y2 = int(frac[3]*h)
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
    return img_bgr[y1:y2, x1:x2]

def decimal_box_from_parent(parent_rect, params, img_shape):
    # parent_rect = (px1,py1,px2,py2); params = (rel_xc,rel_yc,rel_w_rel,rel_h_rel)
    px1,py1,px2,py2 = parent_rect
    p_w = px2 - px1; p_h = py2 - py1
    rel_xc, rel_yc, rel_w_rel, rel_h_rel = params
    dec_cx = int(px1 + rel_xc * p_w)
    dec_cy = int(py1 + rel_yc * p_h)
    dec_w = max(2, int(rel_w_rel * p_w))
    dec_h = max(2, int(rel_h_rel * p_h))
    x1 = max(0, dec_cx - dec_w//2); y1 = max(0, dec_cy - dec_h//2)
    x2 = min(img_shape[1]-1, dec_cx + dec_w//2); y2 = min(img_shape[0]-1, dec_cy + dec_h//2)
    return x1, y1, x2, y2

# Primary processing function (uses YOLO if available else fallback template crop + Tesseract)
def process_image(path_or_np):
    """
    Accepts either a filesystem path (str/Path) or an OpenCV numpy image (BGR).
    Returns result dict:
    {
      "serial": {"text":..., "conf":...},
      "kWh": {"text":..., "conf":...},
      "kVAh": {...},
      "md_kW": {...},
      "demand_kVA": {...},
      "image_quality": {"blur":True/False,"glare":True/False},
      "pass": True/False,
      "reasons": [...]
    }
    """
    # load image to numpy BGR
    if isinstance(path_or_np, (str, Path)):
        img = cv2.imread(str(path_or_np))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_np}")
    else:
        img = path_or_np.copy()

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_var = variance_of_laplacian_gray(gray)
    glare = check_glare(gray)

    result = {
        "serial": {"text": None, "conf": 0.0},
        "kWh": {"text": None, "conf": 0.0},
        "kVAh": {"text": None, "conf": 0.0},
        "md_kW": {"text": None, "conf": 0.0},
        "demand_kVA": {"text": None, "conf": 0.0},
        "image_quality": {"blur_variance": float(blur_var), "blur": blur_var < BLUR_VAR_THRESH, "glare": bool(glare)},
        "pass": False,
        "reasons": []
    }

    # If YOLO model is available, use it
    if yolo_model is not None:
        try:
            preds = yolo_model.predict(source=img, imgsz=640, conf=0.3, verbose=False)[0]
            # iterate boxes
            for box in preds.boxes:
                cls_id = int(box.cls[0].item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                crop = img[y1:y2, x1:x2]
                if cls_id == 1:  # SERIAL_REGION
                    txt, conf = ocr_serial_crop(crop)
                    result["serial"] = {"text": txt, "conf": conf}
                elif cls_id == 2:  # KWH_REGION
                    txt, conf = ocr_numeric_crop(crop)
                    result["kWh"] = {"text": txt, "conf": conf}
                elif cls_id == 3:  # KVAH_REGION
                    txt, conf = ocr_numeric_crop(crop)
                    result["kVAh"] = {"text": txt, "conf": conf}
                elif cls_id == 4 or cls_id == 5:  # MD_KW or DEMAND_KVA depending on labeling
                    # you might want to map cls ids - adapt if your model uses different mapping
                    # Here we assume 4 -> MD_KW, 5 -> DEMAND_KVA
                    # fallback: use class name mapping if available
                    try:
                        cname = yolo_model.names.get(cls_id, "")
                    except Exception:
                        cname = ""
                    if "MD" in cname or "md" in cname or cls_id == 4:
                        txt, conf = ocr_numeric_crop(crop)
                        result["md_kW"] = {"text": txt, "conf": conf}
                    else:
                        txt, conf = ocr_numeric_crop(crop)
                        result["demand_kVA"] = {"text": txt, "conf": conf}
            # finish
        except Exception as e:
            print("YOLO inference error; falling back to template cropping. Error:", e)
            yolo_model = None

    # If no YOLO model or fallback, use region templates + Tesseract
    if yolo_model is None:
        for cls_id, (name, frac) in REGIONS.items():
            crop = crop_from_frac(img, frac)
            if crop.size == 0:
                continue
            if cls_id == 1:
                txt, conf = ocr_serial_crop(crop)
                result["serial"] = {"text": txt, "conf": conf}
            elif cls_id == 2:
                txt, conf = ocr_numeric_crop(crop)
                result["kWh"] = {"text": txt, "conf": conf}
            elif cls_id == 3:
                txt, conf = ocr_numeric_crop(crop)
                result["md_kW"] = {"text": txt, "conf": conf}
            elif cls_id == 0:
                # display region (not directly OCR'd as a unique field)
                pass

        # Optionally produce decimal boxes and check decimal OCR
        for parent_cls, params in DECIMAL_PARAMS.items():
            if parent_cls not in REGIONS: continue
            # parent rect in pixels
            frac = REGIONS[parent_cls][1]
            px1 = int(frac[0]*w); py1 = int(frac[1]*h); px2 = int(frac[2]*w); py2 = int(frac[3]*h)
            dec_x1,dec_y1,dec_x2,dec_y2 = decimal_box_from_parent((px1,py1,px2,py2), params, img.shape)
            dec_crop = img[dec_y1:dec_y2, dec_x1:dec_x2]
            if dec_crop.size == 0: 
                continue
            dec_txt, dec_conf = ocr_numeric_crop(dec_crop)
            # Attach decimal info to corresponding field if present
            if parent_cls == 2:
                if result.get("kWh") and result["kWh"]["text"]:
                    # optionally try to ensure decimal present; if not, combine heuristics
                    result["kWh"]["decimal_detected"] = bool(dec_txt.strip())
                    result["kWh"]["decimal_conf"] = dec_conf
            if parent_cls == 3:
                if result.get("md_kW") and result["md_kW"]["text"]:
                    result["md_kW"]["decimal_detected"] = bool(dec_txt.strip())
                    result["md_kW"]["decimal_conf"] = dec_conf

    # Basic pass logic
    ok_fields = 0
    need = 2  # at least serial + kwh + md ideally
    for f in ["serial", "kWh", "md_kW"]:
        if result[f]["text"] and result[f]["conf"] > CONF_THRESH:
            ok_fields += 1
    if ok_fields >= 2 and not result["image_quality"]["blur"]:
        result["pass"] = True
    else:
        result["pass"] = False
        if result["image_quality"]["blur"]:
            result["reasons"].append("Image too blurry")
        if ok_fields < 2:
            result["reasons"].append("Low OCR confidence or missing fields")

    return result

# ----------------- FLASK APP -----------------
app = Flask(__name__)
CORS(app)  # allow cross-origin for UI (if served from different host)

# Serve sample images
@app.route("/samples/<path:filename>")
def samples_static(filename):
    path = SAMPLES_DIR / filename
    if not path.exists():
        abort(404)
    return send_file(str(path))

# Serve uploads
@app.route("/uploads/<path:filename>")
def uploads_static(filename):
    path = UPLOADS_DIR / filename
    if not path.exists():
        abort(404)
    return send_file(str(path))

# List sample images
@app.route("/api/list_samples", methods=["GET"])
def api_list_samples():
    files = sorted([p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    return jsonify({"samples": files})

# Health check
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "yolo_loaded": bool(yolo_model is not None)})

# Upload image (multipart file)
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error":"missing file field (multipart/form-data 'file')"}), 400
    f = request.files['file']
    if f.filename == "":
        return jsonify({"error":"empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = UPLOADS_DIR / fname
    f.save(str(fpath))
    # process immediately
    try:
        res = process_image(str(fpath))
    except Exception as e:
        return jsonify({"error":"processing failed","detail": str(e)}), 500
    # add returned image url
    res["image_url"] = f"/uploads/{fname}"
    return jsonify(res)

# Upload base64 image (camera capture)
@app.route("/api/upload_base64", methods=["POST"])
def api_upload_base64():
    body = request.get_json(force=True)
    if not body or 'b64' not in body:
        return jsonify({"error":"missing b64 in JSON body"}), 400
    b64 = body['b64']
    try:
        header, data = (b64.split(",",1) if "," in b64 else (None,b64))
        img_bytes = base64.b64decode(data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("decoded image is None")
        # save
        fname = f"{uuid.uuid4().hex}.jpg"
        fpath = UPLOADS_DIR / fname
        cv2.imwrite(str(fpath), img)
        res = process_image(img)  # can process from numpy image
        res["image_url"] = f"/uploads/{fname}"
        return jsonify(res)
    except Exception as e:
        return jsonify({"error":"invalid base64 or decode failed", "detail": str(e)}), 400

# Process existing server sample image by name
@app.route("/api/process", methods=["POST"])
def api_process():
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error":"missing json body"}), 400
    sample = body.get("sample")
    if not sample:
        return jsonify({"error":"missing sample name"}), 400
    path = SAMPLES_DIR / sample
    if not path.exists():
        return jsonify({"error":"sample not found"}), 404
    try:
        res = process_image(str(path))
        res["image_url"] = f"/samples/{sample}"
        return jsonify(res)
    except Exception as e:
        return jsonify({"error":"processing failed","detail": str(e)}), 500

# Serve the index.html and inject small integration script so front-end buttons call our endpoints
@app.route("/")
def index():
    if not FRONTEND_HTML.exists():
        return Response("<h3>index.html not found on server - place it at /mnt/data/index.html</h3>", mimetype="text/html")
    html = FRONTEND_HTML.read_text(encoding="utf-8")
    # integration JS to wire frontend buttons to backend webhooks (keeps it simple)
    integration_js = r"""
<script>
document.addEventListener('DOMContentLoaded', function(){
  // helper to update field boxes
  function updateFields(json){
    document.getElementById('serial-value').innerText = json.serial && json.serial.text ? json.serial.text : '----';
    document.getElementById('serial-conf').innerText = 'Confidence: ' + (json.serial && json.serial.conf ? (json.serial.conf.toFixed(2)) : '-');
    document.getElementById('kwh-value').innerText = json.kWh && json.kWh.text ? json.kWh.text : '----';
    document.getElementById('kwh-conf').innerText = 'Confidence: ' + (json.kWh && json.kWh.conf ? (json.kWh.conf.toFixed(2)) : '-');
    document.getElementById('kvah-value').innerText = json.kVAh && json.kVAh.text ? json.kVAh.text : '----';
    document.getElementById('kvah-conf').innerText = 'Confidence: ' + (json.kVAh && json.kVAh.conf ? (json.kVAh.conf.toFixed(2)) : '-');
    document.getElementById('mdkw-value').innerText = json.md_kW && json.md_kW.text ? json.md_kW.text : '----';
    document.getElementById('mdkw-conf').innerText = 'Confidence: ' + (json.md_kW && json.md_kW.conf ? (json.md_kW.conf.toFixed(2)) : '-');
    document.getElementById('demandkva-value').innerText = json.demand_kVA && json.demand_kVA.text ? json.demand_kVA.text : '----';
    document.getElementById('demandkva-conf').innerText = 'Confidence: ' + (json.demand_kVA && json.demand_kVA.conf ? (json.demand_kVA.conf.toFixed(2)) : '-');
  }

  // Load sample images and process first result
  document.getElementById('btn-load-sample').addEventListener('click', async function(){
    try {
      const r = await fetch('/api/list_samples');
      const j = await r.json();
      if (j.samples && j.samples.length>0) {
        // process the first sample
        const body = JSON.stringify({sample: j.samples[0]});
        const resp = await fetch('/api/process', {method:'POST', headers:{'Content-Type':'application/json'}, body: body});
        const resjson = await resp.json();
        updateFields(resjson);
        console.log('Processed sample', j.samples[0], resjson);
      } else {
        alert('No samples found on server (place images in dataset_100/)');
      }
    } catch(e){ alert('Failed to fetch samples: '+e); }
  });

  // Upload field images (file picker)
  document.getElementById('btn-input-sample').addEventListener('click', function(){
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = 'image/*';
    inp.onchange = async function(){
      const file = inp.files[0];
      if (!file) return;
      const form = new FormData();
      form.append('file', file);
      const resp = await fetch('/api/upload', {method:'POST', body: form});
      const j = await resp.json();
      if (j.error) { alert('Upload error: '+j.error); return; }
      updateFields(j);
    };
    inp.click();
  });

  // Live camera capture -> capture one frame and upload
  document.getElementById('btn-live-camera').addEventListener('click', async function(){
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video:true});
      const v = document.createElement('video');
      v.autoplay = true; v.srcObject = stream;
      // wait for frame
      await new Promise(resolve => setTimeout(resolve, 800));
      const c = document.createElement('canvas');
      c.width = v.videoWidth || 800;
      c.height = v.videoHeight || 600;
      const ctx = c.getContext('2d');
      ctx.drawImage(v, 0, 0, c.width, c.height);
      const dataUrl = c.toDataURL('image/jpeg', 0.9);
      // stop stream
      stream.getTracks().forEach(t=>t.stop());
      const resp = await fetch('/api/upload_base64', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({b64: dataUrl})});
      const j = await resp.json();
      if (j.error) { alert('Capture error: '+j.error); return; }
      updateFields(j);
    } catch(e){
      alert('Camera capture failed: ' + e);
    }
  });

});
</script>
"""
    # inject before </body>
    if "</body>" in html:
        html = html.replace("</body>", integration_js + "\n</body>")
    else:
        html += integration_js
    return Response(html, mimetype="text/html")

# Run app
if __name__ == "__main__":
    # easy-to-change host/port
    app.run(host="0.0.0.0", port=7860, debug=True)
