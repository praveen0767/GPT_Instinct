"""
app_streamlit.py - MeterVision QC (Streamlit single-file app)

Save as app_streamlit.py and run:
    pip install streamlit opencv-python-headless pillow numpy pandas plotly pytesseract
    # optional for detector:
    pip install ultralytics

    streamlit run app_streamlit.py

Folders:
  - dataset_100/    (put sample images here)
  - uploads/        (will be created, saved uploads)
  - outputs/        (batch outputs)

Behavior:
  - If ultralytics & YOLO weights provided, uses detector -> OCR on crops.
  - Otherwise uses fixed region template + pytesseract for OCR (if installed).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import uuid
import io
import time

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import plotly.express as px

# optional libs
try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# ---------- Configuration ----------
BASE_DIR = Path.cwd()
SAMPLES_DIR = BASE_DIR / "dataset_100"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

SAMPLES_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Template region fractions (same as earlier)
REGIONS = {
    0: ("DISPLAY_REGION", (0.05, 0.25, 0.95, 0.90)),  # display
    1: ("SERIAL_REGION",  (0.10, 0.02, 0.90, 0.18)),  # serial
    2: ("KWH_REGION",     (0.18, 0.30, 0.80, 0.54)),  # kWh
    3: ("MD_KW_REGION",   (0.58, 0.62, 0.92, 0.82)),  # md_kW
}
DECIMAL_PARAMS = {
    2: (0.85, 0.52, 0.035, 0.12),
    3: (0.82, 0.50, 0.06, 0.18),
}

BLUR_VAR_THRESH = 80.0
GLARE_PIXEL_RATIO = 0.02
CONF_THRESH = 0.5

# Class name mapping for UI
FIELD_ORDER = ["serial", "kWh", "kVAh", "md_kW", "demand_kVA"]

# ---------- Utility functions ----------
def load_cv2(path: Path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def cv2_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def pil_to_cv2(pil_img: Image.Image):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def variance_of_laplacian_gray(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def check_glare(gray_img):
    h, w = gray_img.shape[:2]
    bright = (gray_img >= 250).sum()
    return (bright / (h * w)) > GLARE_PIXEL_RATIO

def rect_from_frac_w_h(frac, w, h):
    fx1, fy1, fx2, fy2 = frac
    x1 = int(fx1 * w); y1 = int(fy1 * h)
    x2 = int(fx2 * w); y2 = int(fy2 * h)
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2

def decimal_box_from_parent(parent_rect, params, img_shape):
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

def ocr_numeric_crop(img_bgr):
    """Tesseract numeric OCR with confidences if available."""
    if not HAS_TESSERACT:
        return "", 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    conf_cmd = "--psm 7 -c tessedit_char_whitelist=0123456789."
    try:
        data = pytesseract.image_to_data(th, config=conf_cmd, output_type=pytesseract.Output.DICT)
        texts = [t for t in data['text'] if t.strip() != ""]
        if texts:
            text = "".join(texts).strip()
            confs = [int(c) for c in data['conf'] if c != '-1' and c != '']
            conf = float(np.mean(confs))/100.0 if confs else 0.0
        else:
            text = ""
            conf = 0.0
    except Exception:
        text, conf = "", 0.0
    return text, conf

def ocr_serial_crop(img_bgr):
    if not HAS_TESSERACT:
        return "", 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    conf_cmd = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/_"
    try:
        data = pytesseract.image_to_data(th, config=conf_cmd, output_type=pytesseract.Output.DICT)
        texts = [t for t in data['text'] if t.strip() != ""]
        text = " ".join(texts).strip() if texts else ""
        confs = [int(c) for c in data['conf'] if c != '-1' and c != '']
        conf = float(np.mean(confs))/100.0 if confs else 0.0
    except Exception:
        text, conf = "", 0.0
    return text, conf

# ---------- Processing pipeline ----------
# Optional YOLO model (loaded on demand)
yolo_model = None

def load_yolo(weights_path: str):
    global yolo_model
    if not HAS_ULTRALYTICS:
        st.warning("Ultralytics is not installed; cannot load YOLO.")
        return False
    try:
        yolo_model = YOLO(weights_path)
        return True
    except Exception as e:
        st.error(f"Failed to load YOLO weights: {e}")
        yolo_model = None
        return False

def process_image_pipeline(img_bgr) -> Dict[str,Any]:
    """
    Accepts BGR image (numpy), returns result dict matching UI expectations.
    Uses yolo_model if available; otherwise template crop + pytesseract fallback.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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

    # If YOLO available, run detection first
    used_yolo = False
    if yolo_model is not None:
        try:
            preds = yolo_model.predict(source=img_bgr, imgsz=640, conf=0.3, verbose=False)[0]
            used_yolo = True
            for box in preds.boxes:
                cls_id = int(box.cls[0].item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                crop = img_bgr[y1:y2, x1:x2]
                # mapping depends on how you trained the model; here we assume:
                # 0: DISPLAY,1: SERIAL,2:KWH,3:KVAH,4:MD_KW,5:DEMAND_KVA or similar
                # attempt to use `yolo_model.names` if present
                cls_name = yolo_model.names.get(cls_id, str(cls_id)) if hasattr(yolo_model, "names") else str(cls_id)
                # heuristic mapping:
                if "serial" in cls_name.lower() or cls_id == 1:
                    txt, conf = ocr_serial_crop(crop)
                    result["serial"] = {"text": txt, "conf": conf}
                elif "kwh" in cls_name.lower() or cls_id == 2:
                    txt, conf = ocr_numeric_crop(crop)
                    result["kWh"] = {"text": txt, "conf": conf}
                elif "kvah" in cls_name.lower() or cls_id == 3:
                    txt, conf = ocr_numeric_crop(crop)
                    result["kVAh"] = {"text": txt, "conf": conf}
                elif "md" in cls_name.lower() or cls_id == 4:
                    txt, conf = ocr_numeric_crop(crop)
                    result["md_kW"] = {"text": txt, "conf": conf}
                elif "demand" in cls_name.lower() or cls_id == 5:
                    txt, conf = ocr_numeric_crop(crop)
                    result["demand_kVA"] = {"text": txt, "conf": conf}
        except Exception as e:
            st.warning(f"YOLO inference error: {e}")
            # fall back to template below

    # Template crop fallback
    if yolo_model is None or not used_yolo:
        for cls_id, (name, frac) in REGIONS.items():
            x1,y1,x2,y2 = rect_from_frac_w_h(frac, w, h)
            crop = img_bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            if name == "SERIAL_REGION":
                txt, conf = ocr_serial_crop(crop)
                result["serial"] = {"text": txt, "conf": conf}
            elif name == "KWH_REGION" or "KWH" in name:
                txt, conf = ocr_numeric_crop(crop)
                result["kWh"] = {"text": txt, "conf": conf}
            elif name == "MD_KW_REGION":
                txt, conf = ocr_numeric_crop(crop)
                result["md_kW"] = {"text": txt, "conf": conf}
            # display region not directly OCR'd; KVAh & demand not present in template

        # decimal detection on crops
        for parent_cls, params in DECIMAL_PARAMS.items():
            if parent_cls not in REGIONS:
                continue
            frac = REGIONS[parent_cls][1]
            px1,py1,px2,py2 = rect_from_frac_w_h(frac, w, h)
            dec_x1,dec_y1,dec_x2,dec_y2 = decimal_box_from_parent((px1,py1,px2,py2), params, img_bgr.shape)
            dec_crop = img_bgr[dec_y1:dec_y2, dec_x1:dec_x2]
            if dec_crop is None or dec_crop.size == 0:
                continue
            dec_txt, dec_conf = ocr_numeric_crop(dec_crop)
            if parent_cls == 2:
                result["kWh"]["decimal_detected"] = bool(dec_txt.strip())
                result["kWh"]["decimal_conf"] = dec_conf
            if parent_cls == 3:
                result["md_kW"]["decimal_detected"] = bool(dec_txt.strip())
                result["md_kW"]["decimal_conf"] = dec_conf

    # simple pass logic
    ok_fields = 0
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

# ---------- Visualization helpers ----------
def draw_overlays(img_bgr, results=None, show_template=True, show_decimal=True):
    """Return a PIL image with boxes drawn."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    # draw template regions
    if show_template:
        for cls_id, (name, frac) in REGIONS.items():
            x1,y1,x2,y2 = rect_from_frac_w_h(frac, w, h)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, name, (x1+4, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # decimal boxes
    if show_decimal:
        for parent_cls, params in DECIMAL_PARAMS.items():
            if parent_cls not in REGIONS: continue
            frac = REGIONS[parent_cls][1]
            px1,py1,px2,py2 = rect_from_frac_w_h(frac, w, h)
            x1,y1,x2,y2 = decimal_box_from_parent((px1,py1,px2,py2), params, img_bgr.shape)
            cv2.rectangle(vis, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(vis, f"DEC_{parent_cls}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    # add text overlay of results (if present)
    if results:
        y = 20
        for k in ["serial","kWh","md_kW"]:
            text = f"{k}: {results.get(k,{}).get('text','')}"
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
            y += 22
    return cv2_to_pil(vis)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="MeterVision QC", layout="wide")
st.title("MeterVision QC — Streamlit Demo")

# Sidebar controls
st.sidebar.header("Model & Settings")
use_yolo = st.sidebar.checkbox("Attempt to use YOLO model (if installed)", value=False)
yolo_weights_input = st.sidebar.text_input("YOLO weights path (optional)", value=str(BASE_DIR/"runs/detect/meter_field_detector/weights/best.pt"))
if use_yolo and yolo_weights_input:
    if st.sidebar.button("Load YOLO model"):
        ok = load_yolo(yolo_weights_input)
        st.sidebar.success("Loaded YOLO" if ok else "Load failed")

st.sidebar.write("Tesseract available:" , "Yes" if HAS_TESSERACT else "No")
st.sidebar.write("Ultralytics available:", "Yes" if HAS_ULTRALYTICS else "No")
st.sidebar.markdown("---")
st.sidebar.write("Thresholds")
CONF_THRESH = st.sidebar.slider("OCR confidence threshold (pass)", 0.0, 1.0, float(CONF_THRESH), 0.01)
BLUR_VAR_THRESH = st.sidebar.slider("Blur variance threshold (lower=more blur)", 10.0, 500.0, float(BLUR_VAR_THRESH), 1.0)

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Image Workspace")
    # Controls row
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Load sample images"):
            # simply list sample images
            samples = sorted([p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
            st.session_state["samples"] = samples
            st.success(f"Found {len(samples)} sample images")
    with c2:
        uploaded_file = st.file_uploader("Upload image file", type=["jpg","jpeg","png"])
        if uploaded_file is not None:
            # save file
            fname = f"{uuid.uuid4().hex}.jpg"
            path = UPLOADS_DIR / fname
            img = Image.open(uploaded_file)
            img.save(path)
            st.session_state.setdefault("samples", [])
            st.session_state["samples"].insert(0, fname)
            st.success("Uploaded and saved")
    with c3:
        cam = st.camera_input("Capture from camera (single frame)")
        if cam is not None:
            img = Image.open(cam)
            fname = f"{uuid.uuid4().hex}.jpg"
            path = UPLOADS_DIR / fname
            img.save(path)
            st.session_state.setdefault("samples", [])
            st.session_state["samples"].insert(0, fname)
            st.success("Captured and saved")

    # thumbnails and selection
    samples = st.session_state.get("samples", sorted([p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]))
    st.session_state["samples"] = samples

    if not samples:
        st.info("No sample images found in dataset_100/ (use Upload or add images).")
    else:
        # display thumbnails in grid
        per_row = 4
        rows = [samples[i:i+per_row] for i in range(0, len(samples), per_row)]
        selected = st.session_state.get("selected_image", samples[0] if samples else None)
        for row in rows:
            cols = st.columns(len(row))
            for c, name in zip(cols, row):
                p = SAMPLES_DIR / name if (SAMPLES_DIR / name).exists() else UPLOADS_DIR / name
                if not p.exists(): continue
                with c:
                    try:
                        thumb = Image.open(p)
                        thumb.thumbnail((160,120))
                    except Exception:
                        thumb = None
                    if thumb:
                        if st.button(name, key=f"btn_{name}"):
                            st.session_state["selected_image"] = name
                            selected = name
                        c.image(thumb, caption=name)
                    else:
                        c.write(name)

        # show selected detail and processing
        if selected:
            st.markdown("---")
            st.write(f"**Selected:** {selected}")
            source_path = SAMPLES_DIR / selected if (SAMPLES_DIR / selected).exists() else UPLOADS_DIR / selected
            img_bgr = load_cv2(source_path)
            if img_bgr is None:
                st.error("Could not read selected image.")
            else:
                # show preview with overlay toggle
                overlay = st.checkbox("Show template overlays (boxes)", value=True)
                decimal_vis = st.checkbox("Show decimal boxes", value=True)
                # process image & display results
                if st.button("Process selected image"):
                    with st.spinner("Running pipeline..."):
                        result = process_image_pipeline(img_bgr)
                        st.session_state["last_result"] = result
                        st.success("Processing finished")
                result = st.session_state.get("last_result", None)
                pil_vis = draw_overlays(img_bgr, result, show_template=overlay, show_decimal=decimal_vis)
                st.image(pil_vis, use_column_width=True)
                st.markdown("**OCR Results**")
                if result:
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("Serial:", result["serial"].get("text") or "----")
                        st.write("kWh:", result["kWh"].get("text") or "----")
                        st.write("MD kW:", result["md_kW"].get("text") or "----")
                    with cols[1]:
                        st.write("kVAh:", result["kVAh"].get("text") or "----")
                        st.write("Demand kVA:", result["demand_kVA"].get("text") or "----")
                        st.write("Pass:", "✅" if result.get("pass") else "❌")
                        if result.get("reasons"):
                            st.write("Reasons:", ", ".join(result.get("reasons")))

with col2:
    st.subheader("QC Dashboard")
    # Basic metrics computed from session results or batch runs
    records = st.session_state.get("batch_results", [])
    # quick buttons
    if st.button("Run batch simulation (all samples)"):
        samples_local = st.session_state.get("samples", [])
        new_records = []
        if not samples_local:
            st.warning("No samples to run.")
        else:
            with st.spinner("Running batch..."):
                for i, name in enumerate(samples_local):
                    path = SAMPLES_DIR / name if (SAMPLES_DIR / name).exists() else UPLOADS_DIR / name
                    img_bgr = load_cv2(path)
                    res = process_image_pipeline(img_bgr)
                    res_row = {
                        "image": name,
                        "serial": res["serial"]["text"],
                        "serial_conf": res["serial"]["conf"],
                        "kWh": res["kWh"]["text"],
                        "kWh_conf": res["kWh"]["conf"],
                        "md_kW": res["md_kW"]["text"],
                        "md_kW_conf": res["md_kW"]["conf"],
                        "pass": res["pass"],
                        "reasons": ";".join(res.get("reasons", []))
                    }
                    new_records.append(res_row)
                # append to session state
                st.session_state.setdefault("batch_results", [])
                st.session_state["batch_results"] = new_records
                records = new_records
            st.success(f"Batch processed {len(new_records)} images")

    if records:
        df = pd.DataFrame(records)
        st.write("Recent batch results (first 200):")
        st.dataframe(df.head(200))

        # compute basic metrics
        pass_rate = df["pass"].mean()
        avg_conf = {
            "serial": df["serial_conf"].mean(),
            "kWh": df["kWh_conf"].mean(),
            "md_kW": df["md_kW_conf"].mean()
        }
        st.metric("Pass Rate", f"{pass_rate*100:.1f}%")
        st.write("Average confidences")
        st.write(pd.Series(avg_conf).apply(lambda x: f"{x:.2f}"))

        # charts
        fig = px.bar(x=["Serial","kWh","MD_kW"], y=[avg_conf["serial"], avg_conf["kWh"], avg_conf["md_kW"]],
                     labels={"x":"Field","y":"Avg Confidence"})
        st.plotly_chart(fig, use_container_width=True)

        # error breakdown pie
        reason_counts = {}
        for r in df["reasons"].astype(str).values:
            if not r or r == "nan": continue
            for part in r.split(";"):
                reason_counts[part] = reason_counts.get(part, 0) + 1
        if reason_counts:
            rc_df = pd.DataFrame({"reason": list(reason_counts.keys()), "count": list(reason_counts.values())})
            fig2 = px.pie(rc_df, names="reason", values="count", title="Error Type Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        # export CSV
        if st.button("Export batch CSV"):
            out_csv = OUTPUTS_DIR / f"batch_{int(time.time())}.csv"
            df.to_csv(out_csv, index=False)
            st.success(f"Saved CSV: {out_csv}")

    else:
        st.info("No batch results yet. Run a batch simulation.")

st.markdown("---")
st.caption("MeterVision QC — Streamlit prototype. Replace the pipeline with your trained YOLO + CRNN for production.")

