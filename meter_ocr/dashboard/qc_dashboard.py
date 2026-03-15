"""
meter_ocr/dashboard/qc_dashboard.py
Streamlit QC Dashboard for manual review of low-confidence meter readings.

Run:
  streamlit run meter_ocr/dashboard/qc_dashboard.py

Features:
  • Show image
  • Show extracted fields
  • Highlight low-confidence values
  • Allow human correction
  • Store corrected labels (to corrections.jsonl)
"""
import os
import json
import datetime
import streamlit as st
import requests
import numpy as np

BACKEND_URL = os.environ.get("METER_OCR_BACKEND", "http://localhost:8000")
CORRECTIONS_FILE = "corrections.jsonl"
LOW_CONF_THRESHOLD = 0.95

st.set_page_config(page_title="Meter OCR QC Dashboard", layout="wide", page_icon="⚡")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .field-card { border:1px solid #333; border-radius:8px; padding:12px; margin:4px 0; }
  .low-conf   { background:#3a1a1a; border-color:#ff4444; }
  .ok-conf    { background:#1a2a1a; border-color:#44aa44; }
  .conf-badge { font-size:0.78rem; padding:2px 8px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Anti-Gravity OCR — QC Dashboard")
st.caption("Upload a meter image to extract and review all readings. Correct any errors and save.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    low_thresh  = st.slider("Low-confidence threshold", 0.0, 1.0, LOW_CONF_THRESHOLD, 0.01)
    st.divider()
    # Health check
    try:
        h = requests.get(f"{backend_url}/health", timeout=3)
        status = "🟢 Online" if h.ok else "🔴 Error"
    except Exception:
        status = "🔴 Offline"
    st.metric("API Status", status)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload meter image", type=["png","jpg","jpeg","bmp","tiff"])

FIELDS = [
    ("kwh",         "kWh Reading",    "⚡"),
    ("kvah",        "kVAh Reading",   "🔋"),
    ("md_kw",       "MD kW",          "📊"),
    ("demand_kva",  "Demand kVA",     "📈"),
    ("meter_serial","Meter Serial",   "🔢"),
]

if uploaded:
    col_img, col_res = st.columns([1, 1.6])

    with col_img:
        st.image(uploaded, caption=f"{uploaded.name} ({uploaded.size//1024} KB)")

    with col_res:
        with st.spinner("Running OCR pipeline…"):
            try:
                uploaded.seek(0)
                resp = requests.post(
                    f"{backend_url}/infer",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                    timeout=60,
                )
                if not resp.ok:
                    st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
                    st.stop()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to backend at {backend_url}. Start with:\n"
                         f"`uvicorn meter_ocr.api.main:app --reload`")
                st.stop()

        # ── Image quality ──────────────────────────────────────────────────
        iq = result.get("image_quality", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blur",    "YES" if iq.get("blur")    else "NO")
        c2.metric("Glare",   "YES" if iq.get("glare")   else "NO")
        c3.metric("Tilt °",  f"{iq.get('tilt_deg',0):.1f}")
        c4.metric("Legible", "NO" if iq.get("not_legible") else "YES")

        latency = result.get("processing_latency_ms", 0)
        qc_flag = result.get("qc_flag", False)

        if qc_flag:
            codes = result.get("reason_codes", [])
            st.warning(f"⚠️ QC Flag — {', '.join(codes) if codes else 'Low confidence'}")
        else:
            st.success("✅ All fields passed QC")

        st.caption(f"Processing time: {latency}ms")
        st.divider()

        # ── Per-field cards with correction inputs ─────────────────────────
        st.subheader("Extracted Fields")
        corrections = {}

        for field, label, icon in FIELDS:
            fd   = result.get(field, {})
            val  = fd.get("value", "—")
            prob = fd.get("probability", 0.0)
            low  = (prob < low_thresh) or (val in ("—","N/A",""))

            card_class = "low-conf" if low else "ok-conf"
            conf_pct   = f"{prob*100:.1f}%"

            st.markdown(f"""
<div class='field-card {card_class}'>
  <b>{icon} {label}</b>
  <span style='float:right' class='conf-badge'>Confidence: {conf_pct}</span>
  <br/>
  <span style='font-size:1.4rem; font-weight:bold'>{val}</span>
  {"⚠️ Low confidence" if low else ""}
</div>
""", unsafe_allow_html=True)

            corrected = st.text_input(
                f"Correct {label} (leave blank to accept)",
                key=f"corr_{field}",
                value="" if not low else val,
                placeholder="Enter corrected value…",
            )
            corrections[field] = corrected if corrected.strip() else val

        st.divider()

        # ── Save corrections ───────────────────────────────────────────────
        if st.button("💾 Save Corrections", type="primary"):
            entry = {
                "timestamp":  datetime.datetime.utcnow().isoformat(),
                "filename":   uploaded.name,
                "original":   {f: result.get(f, {}).get("value","") for f,*_ in FIELDS},
                "corrected":  corrections,
                "qc_flag":    qc_flag,
                "latency_ms": latency,
            }
            with open(CORRECTIONS_FILE, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
            st.success(f"Corrections saved to `{CORRECTIONS_FILE}`")

        # ── Raw JSON expander ──────────────────────────────────────────────
        with st.expander("📄 Raw API Response"):
            st.json(result)

    # ── Corrections log ────────────────────────────────────────────────────────
    if os.path.isfile(CORRECTIONS_FILE):
        with st.expander(f"📋 Correction Log ({CORRECTIONS_FILE})"):
            lines = open(CORRECTIONS_FILE).readlines()[-20:]
            for line in reversed(lines):
                try:
                    st.json(json.loads(line))
                except Exception:
                    st.text(line)
