"""
ocr_pipeline/paddle_adapter.py
Strict numeric-only PaddleOCR adapter.
"""
import re
import numpy as np

_HAS_DIGIT = re.compile(r'\d')

def _clean(text: str) -> str:
    t = text.strip()
    for frm, to in [("O","0"),("o","0"),("I","1"),("l","1"),("S","5"),("B","8"),("Z","2")]:
        t = t.replace(frm, to)
    t = re.sub(r'[^0-9.]', '', t)
    return t

def _is_valid_reading(text: str) -> bool:
    t = text.strip()
    if not _HAS_DIGIT.search(t):
        return False
    parts = t.split(".")
    if len(parts) > 2:
        return False
    return all(p.isdigit() for p in parts if p)


class PaddleAdapter:
    """Adapter for PaddleOCR — returns only numeric meter readings."""

    def __init__(self, use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                enable_mkldnn=False,
                show_log=False,
            )
            self.mock_mode = False
        except Exception as e:
            print(f"Warning: Could not load PaddleOCR ({e}). Running in mock mode.")
            self.mock_mode = True

    # ------------------------------------------------------------------
    def recognize(self, image: np.ndarray) -> dict:
        if self.mock_mode:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        try:
            results = self.ocr.ocr(image)
        except Exception as e:
            print(f"PaddleOCR inference error: {e}")
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        if not results or not results[0]:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        # ── Collect all numeric-looking lines ───────────────────────────
        candidates = []
        for line in results[0]:
            raw_text = line[1][0]
            conf     = float(line[1][1])
            if not _HAS_DIGIT.search(raw_text):
                continue
            cleaned = _clean(raw_text)
            if cleaned and _is_valid_reading(cleaned):
                # score: favour long strings, high confidence, decimal presence
                decimal_bonus = 1.5 if '.' in cleaned else 1.0
                score = conf * len(cleaned) * decimal_bonus
                candidates.append((cleaned, conf, score))

        if not candidates:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        best_text, best_conf, _ = max(candidates, key=lambda x: x[2])

        return {
            "text": best_text,
            "confidence": best_conf,
            "tokens": list(best_text),
            "token_scores": [best_conf] * len(best_text),
        }
