"""
frontend/streamlit_demo.py
Anti-Gravity OCR — Streamlit Demo
Run: streamlit run frontend/streamlit_demo.py
"""

import io
import time

import requests
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anti-Gravity OCR — Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Config")
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="Base URL of the FastAPI OCR backend.",
    )
    show_raw_json = st.checkbox("Show raw JSON", value=False)
    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown(
        """
1. YOLOv8 Detection  
2. Dewarp → CLAHE → Real-ESRGAN  
3. TrOCR · PaddleOCR · EasyOCR  
4. ROVER Voting  
5. LLM Correction  
6. Isotonic Calibration  
        """
    )
    st.markdown("---")
    st.caption("Anti-Gravity OCR · Instinct GPT · v1")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>⚡ Anti-Gravity OCR</h1>"
    "<p style='color:gray;margin-top:4px'>Instinct GPT Pipeline — Meter Reading Extraction</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── upload area ───────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a utility meter image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Supports JPG, PNG, WebP. Max recommended: 5 MB.",
        label_visibility="visible",
    )

with col_preview:
    if uploaded:
        st.image(uploaded, caption=f"📷 {uploaded.name}", use_column_width=True)
    else:
        st.markdown(
            "<div style='border:2px dashed #555;border-radius:12px;padding:32px;text-align:center;"
            "color:#888;font-size:0.85rem;'>No image selected</div>",
            unsafe_allow_html=True,
        )

# ── run button ────────────────────────────────────────────────────────────────
if not uploaded:
    st.info("👆 Upload an image above to get started.")
    st.stop()

if not st.button("⚡ Run OCR", type="primary", use_container_width=False):
    st.stop()

# ── call backend ──────────────────────────────────────────────────────────────
infer_url = f"{backend_url.rstrip('/')}/infer"
img_bytes = uploaded.read()

with st.spinner("Running Anti-Gravity OCR pipeline…"):
    t0 = time.time()
    try:
        resp = requests.post(
            infer_url,
            files={"file": (uploaded.name, io.BytesIO(img_bytes), uploaded.type)},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Could not connect to backend at **{infer_url}**. "
            "Is `uvicorn api.app:app` running?"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out (>120s). The server may be overloaded.")
        st.stop()
    elapsed = time.time() - t0

if not resp.ok:
    st.error(f"❌ Server returned **{resp.status_code}**: `{resp.text[:500]}`")
    st.stop()

j = resp.json()
st.success(f"✅ Done in **{elapsed:.2f}s** (server: {j.get('processing_latency_ms', '?')}ms)")

# ── QC / flag banner ──────────────────────────────────────────────────────────
if j.get("qc_flag"):
    codes = " · ".join(j.get("reason_codes") or ["FLAGGED"])
    st.warning(f"⚠️ **QC Flag** — This result was flagged for human review: `{codes}`")

st.divider()

# ── image quality ─────────────────────────────────────────────────────────────
iq = j.get("image_quality") or {}
st.markdown("### 🔍 Image Quality")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Blur",    "⚠️ Yes" if iq.get("blur")        else "✅ No")
q2.metric("Glare",   "⚠️ Yes" if iq.get("glare")       else "✅ No")
q3.metric("Legible", "✅ Yes" if not iq.get("not_legible") else "❌ No")
q4.metric("Tilt",    f"{iq.get('tilt_deg', 0):.1f}°")

st.divider()

# ── extracted fields ──────────────────────────────────────────────────────────
st.markdown("### ⚡ Extracted Fields")

FIELD_META = {
    "kwh":          ("kWh Reading",   "⚡"),
    "meter_serial": ("Meter Serial",  "🔢"),
    "kvah":         ("kVAh",          "📊"),
    "md_kw":        ("MD kW",         "📈"),
    "demand_kva":   ("Demand kVA",    "🔋"),
}

cols = st.columns(len(FIELD_META))
for col, (key, (label, icon)) in zip(cols, FIELD_META.items()):
    v = j.get(key) or {}
    val   = v.get("value", "—")
    prob  = v.get("probability", 0.0)
    delta_color = "normal" if prob >= 0.98 else "inverse"
    col.metric(
        label=f"{icon} {label}",
        value=val,
        delta=f"{prob*100:.1f}% conf",
        delta_color=delta_color,
    )

st.divider()

# ── candidates table ──────────────────────────────────────────────────────────
kwh_v = j.get("kwh") or {}
if kwh_v.get("candidates"):
    st.markdown("### 🗳️ kWh Candidates (ROVER)")
    import pandas as pd  # only import if needed

    rows = [(c.get("value", ""), round(c.get("score", 0) * 100, 2)) for c in kwh_v["candidates"]]
    df = pd.DataFrame(rows, columns=["Candidate Value", "Score (%)"])
    df = df.sort_values("Score (%)", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # debug
    dbg = kwh_v.get("debug") or {}
    if dbg.get("raw_ocr"):
        st.caption(
            f"Raw OCR text: `{dbg['raw_ocr']}` · "
            f"Decimal detector: `{dbg.get('decimal_detector_score', '?')}`"
        )
    st.divider()

# ── artifact images ───────────────────────────────────────────────────────────
art = j.get("artifacts") or {}
has_artifacts = any([art.get("crop_url"), art.get("sr_url"), art.get("color_mask_url")])
if has_artifacts:
    st.markdown("### 🖼️ Debug Artifacts")
    art_cols = st.columns(3)

    def try_show_image(col, url, caption):
        """Attempt to load and display artifact image from URL."""
        if not url:
            return
        try:
            r = requests.get(url, timeout=5)
            if r.ok and r.headers.get("Content-Type", "").startswith("image"):
                col.image(r.content, caption=caption, use_column_width=True)
            else:
                col.markdown(
                    f"[🔗 {caption}]({url})",
                    help="Click to open in browser (URL may require VPN/auth)",
                )
        except Exception:
            col.markdown(f"[🔗 {caption}]({url})", help="Could not load image directly.")

    try_show_image(art_cols[0], art.get("crop_url"),        "📐 Cropped Region")
    try_show_image(art_cols[1], art.get("sr_url"),          "🚀 Super Resolution")
    try_show_image(art_cols[2], art.get("color_mask_url"),  "🎨 HSV Mask")
    st.divider()

# ── raw JSON ──────────────────────────────────────────────────────────────────
if show_raw_json:
    st.markdown("### 📄 Raw JSON Response")
    st.json(j)
