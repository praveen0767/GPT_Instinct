"""
ocr_pipeline/easyocr_adapter.py
Strict numeric-only EasyOCR adapter.
Returns ONLY results that look like meter readings (digits + optional decimal).
"""
import re
import numpy as np

# Pattern: optional leading spaces/zeros, 1–10 digits, optional decimal, 0–4 more digits
_NUMERIC_RE = re.compile(r'^\s*[\d\s,]+[\.\,]?\d{0,4}\s*$')
_HAS_DIGIT  = re.compile(r'\d')

def _is_numeric(text: str) -> bool:
    """Return True if text looks like a meter reading (digits ± one decimal)."""
    t = text.strip().replace(" ", "").replace(",", ".")
    if not _HAS_DIGIT.search(t):
        return False
    # after one optional decimal point, rest must be digits
    parts = t.split(".")
    if len(parts) > 2:
        return False
    return all(p.isdigit() for p in parts if p)

def _clean(text: str) -> str:
    """Normalise OCR text to canonical digit form."""
    t = text.strip()
    # Common OCR substitutions
    for frm, to in [("O","0"),("o","0"),("I","1"),("l","1"),("S","5"),("B","8"),("Z","2")]:
        t = t.replace(frm, to)
    # Remove everything except digits and decimal
    t = re.sub(r'[^0-9.]', '', t)
    return t


class EasyOCRAdapter:
    """Adapter for EasyOCR — returns only numeric meter readings."""

    def __init__(self, use_gpu: bool = False):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            self.mock_mode = False
        except Exception as e:
            print(f"Warning: Could not load EasyOCR ({e}). Running in mock mode.")
            self.mock_mode = True

    # ------------------------------------------------------------------
    def recognize(self, image: np.ndarray) -> dict:
        """
        Returns the best numeric reading found, or empty dict if nothing numeric.
        Output schema: {text, confidence, tokens, token_scores}
        """
        if self.mock_mode:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        try:
            results = self.reader.readtext(image, detail=1, paragraph=False)
        except Exception as e:
            print(f"EasyOCR inference error: {e}")
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        if not results:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        # ── Step 1: collect all blocks that contain at least one digit ──
        digit_blocks = []
        for (bbox, text, conf) in results:
            if _HAS_DIGIT.search(text):
                cleaned = _clean(text)
                if cleaned:  # only keep if cleaning leaves something
                    digit_blocks.append((cleaned, float(conf), len(cleaned)))

        if not digit_blocks:
            # No numeric text at all in this image crop — return empty
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        # ── Step 2: score each block ─────────────────────────────────────
        # Prefer: long strings of digits, high confidence, presence of decimal
        def score(item):
            t, c, length = item
            decimal_bonus = 1.5 if '.' in t else 1.0
            return c * length * decimal_bonus

        best_text, best_conf, _ = max(digit_blocks, key=score)

        # ── Step 3: final numeric validation ─────────────────────────────
        if not _is_numeric(best_text):
            # Clean up character by character
            best_text = re.sub(r'[^0-9.]', '', best_text)

        if not best_text or not _HAS_DIGIT.search(best_text):
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        return {
            "text": best_text,
            "confidence": best_conf,
            "tokens": list(best_text),
            "token_scores": [best_conf] * len(best_text),
        }
