"""
tests/test_ui_proxy.py
Unit tests for the /ui/infer proxy route and /ui static mount.

Run (from repo root):
    .venv\\Scripts\\activate
    pytest tests/test_ui_proxy.py -v
"""

import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_minimal_png(width: int = 4, height: int = 4) -> bytes:
    """
    Build a syntactically valid, minimal PNG file (8-bit RGB, white pixels).
    Does NOT require PIL / Pillow so the test has zero extra dependencies.
    """

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # Raw scanline: filter byte (0) + RGB pixels (white = 0xFF,0xFF,0xFF)
    scanline = b"\x00" + b"\xFF\xFF\xFF" * width
    raw = scanline * height
    idat = _chunk(b"IDAT", zlib.compress(raw))

    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


MINIMAL_PNG = _make_minimal_png()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUIInferProxy:
    """Tests for POST /ui/infer logging proxy."""

    def test_valid_png_returns_200(self):
        """A valid PNG upload should complete with HTTP 200."""
        resp = client.post(
            "/ui/infer",
            files={"file": ("test_meter.png", io.BytesIO(MINIMAL_PNG), "image/png")},
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        )

    def test_valid_png_response_has_required_keys(self):
        """Response JSON must contain the core OCR schema keys."""
        resp = client.post(
            "/ui/infer",
            files={"file": ("test_meter.png", io.BytesIO(MINIMAL_PNG), "image/png")},
        )
        assert resp.status_code == 200
        j = resp.json()
        for key in ("image_id", "kwh", "image_quality", "qc_flag", "processing_latency_ms"):
            assert key in j, f"Missing key '{key}' in response: {list(j.keys())}"

    def test_valid_png_kwh_field_structure(self):
        """kwh field must contain value and probability sub-keys."""
        resp = client.post(
            "/ui/infer",
            files={"file": ("test_meter.png", io.BytesIO(MINIMAL_PNG), "image/png")},
        )
        assert resp.status_code == 200
        kwh = resp.json().get("kwh", {})
        assert "value" in kwh, "kwh.value missing"
        assert "probability" in kwh, "kwh.probability missing"

    def test_invalid_file_type_is_handled(self):
        """Posting a plain-text file should return 400 (bad image) not 500."""
        resp = client.post(
            "/ui/infer",
            files={"file": ("not_an_image.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        # Backend should return 400 (Invalid image) rather than crash
        assert resp.status_code in (400, 422), (
            f"Expected 400 or 422 for bad file, got {resp.status_code}"
        )

    def test_empty_file_is_handled(self):
        """Posting zero-byte content should return 400 (or 500 at worst), not silently succeed."""
        resp = client.post(
            "/ui/infer",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert resp.status_code in (400, 422, 500), (
            f"Expected error status for empty file, got {resp.status_code}"
        )
        # If it returns 200 with valid JSON that's a bug — it should not succeed
        assert resp.status_code != 200, "Empty file should not return 200"


    def test_processing_latency_ms_is_positive(self):
        """processing_latency_ms must be a non-negative integer."""
        resp = client.post(
            "/ui/infer",
            files={"file": ("test_meter.png", io.BytesIO(MINIMAL_PNG), "image/png")},
        )
        assert resp.status_code == 200
        lat = resp.json().get("processing_latency_ms")
        assert isinstance(lat, (int, float)) and lat >= 0, (
            f"processing_latency_ms should be >= 0, got {lat!r}"
        )


class TestStaticMount:
    """Tests for GET /ui static file serving."""

    def test_ui_root_returns_html(self):
        """GET /ui should serve the index.html (200 + text/html)."""
        resp = client.get("/ui/")
        # 200 if frontend/ exists, 404 otherwise (CI without frontend/)
        assert resp.status_code in (200, 404), (
            f"Unexpected status {resp.status_code}"
        )
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            assert "text/html" in ct, f"Expected text/html, got {ct!r}"

    def test_health_endpoint_still_works(self):
        """Ensure /health route is unaffected by the static mount."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
