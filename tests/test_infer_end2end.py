"""
tests/test_infer_end2end.py
Integration tests for the /infer and /ui/infer endpoints.
All tests use TestClient (no live server needed).
"""
import io
import struct
import zlib
import re
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_HAS_DIGIT = re.compile(r'\d')

# ── PNG factory ───────────────────────────────────────────────────────────────

def _make_png(width=8, height=8, color=(128, 128, 128)) -> bytes:
    """Create a minimal valid PNG filled with a solid colour."""
    sig  = b"\x89PNG\r\n\x1a\n"
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    row  = b"\x00" + bytes(color) * width
    idat = chunk(b"IDAT", zlib.compress(row * height))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


GRAY_PNG  = _make_png(color=(128, 128, 128))
WHITE_PNG = _make_png(color=(255, 255, 255))
TINY_PNG  = _make_png(width=1, height=1)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "healthy"


# ── /infer — schema validation ────────────────────────────────────────────────

def test_infer_returns_200():
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"


def test_infer_schema_keys():
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    j = r.json()
    for key in ("image_id", "kwh", "kvah", "md_kw", "demand_kva",
                "meter_serial", "image_quality", "qc_flag",
                "processing_latency_ms", "artifacts"):
        assert key in j, f"Missing key '{key}'  keys={list(j)}"


def test_infer_kwh_field_structure():
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    kwh = r.json().get("kwh", {})
    assert "value"       in kwh, "kwh.value missing"
    assert "probability" in kwh, "kwh.probability missing"


def test_infer_all_fields_have_value():
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    j = r.json()
    for field in ("kwh", "kvah", "md_kw", "demand_kva", "meter_serial"):
        fd = j.get(field, {})
        assert fd.get("value") is not None, f"{field}.value is None"


def test_infer_image_quality_has_all_keys():
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    iq = r.json().get("image_quality", {})
    for key in ("blur", "glare", "tilt_deg", "not_legible"):
        assert key in iq, f"image_quality.{key} missing"


def test_kwh_value_is_never_model_text():
    """The pipeline must never return non-numeric text like 'Int', 'kWh', etc."""
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    val = r.json().get("kwh", {}).get("value", "")
    # Accept numeric, "—", "N/A" or empty; never accept pure alpha like "Int"
    if val not in ("", "—", "N/A"):
        assert _HAS_DIGIT.search(val) or val in ("—", "N/A"), \
            f"kwh.value='{val}' looks non-numeric!"


def test_no_hardcoded_serial():
    """meter_serial must not be the old hardcoded '12345678' mock."""
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    serial = r.json().get("meter_serial", {}).get("value", "")
    assert serial != "12345678", "meter_serial is still the hardcoded mock!"


def test_latency_ms_recorded():
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    lat = r.json().get("processing_latency_ms")
    assert isinstance(lat, (int, float)) and lat >= 0


def test_infer_empty_file_returns_400():
    r = client.post("/infer", files={"file": ("empty.png", io.BytesIO(b""), "image/png")})
    assert r.status_code in (400, 422), f"Expected 4xx, got {r.status_code}"


def test_infer_non_image_returns_error():
    r = client.post("/infer", files={"file": ("text.txt", io.BytesIO(b"not an image"), "text/plain")})
    assert r.status_code in (400, 422), f"Expected 4xx, got {r.status_code}"


# ── Post-processor enrichment fields ──────────────────────────────────────────

def test_infer_has_flags_field():
    """kwh response should contain 'flags' key (list or null)."""
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    kwh = r.json().get("kwh", {})
    # flags should exist as key (may be null or list)
    assert "flags" in kwh, f"kwh.flags missing  keys={list(kwh)}"
    if kwh["flags"] is not None:
        assert isinstance(kwh["flags"], list)


def test_infer_has_decimal_confidence():
    """Numeric fields should contain 'decimal_confidence' key (float or null)."""
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    kwh = r.json().get("kwh", {})
    assert "decimal_confidence" in kwh, f"kwh.decimal_confidence missing  keys={list(kwh)}"


def test_infer_has_reason():
    """Fields should contain 'reason' key from post-processor."""
    r = client.post("/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    kwh = r.json().get("kwh", {})
    assert "reason" in kwh, f"kwh.reason missing  keys={list(kwh)}"


# ── /ui/infer ─────────────────────────────────────────────────────────────────

def test_ui_infer_200():
    r = client.post("/ui/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200


def test_ui_infer_same_schema():
    r = client.post("/ui/infer", files={"file": ("test.png", io.BytesIO(GRAY_PNG), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert "kwh" in j and "image_quality" in j


# ── /ui static ────────────────────────────────────────────────────────────────

def test_ui_static_serves():
    r = client.get("/ui/")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "text/html" in r.headers.get("content-type", "")
