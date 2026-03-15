"""
meter_ocr/ocr/paddle_reader.py
PaddleOCR reader — numeric display extraction only.
Weight in ensemble: 0.3
"""
import re
import numpy as np

_SUBS = {'O':'0','o':'0','I':'1','l':'1','S':'5','B':'8','Z':'2','G':'6'}
_DIGIT_RE = re.compile(r'\d')


def _clean(text: str) -> str:
    t = text.strip()
    for f, r in _SUBS.items():
        t = t.replace(f, r)
    parts = t.split('.')
    if len(parts) == 1:
        return re.sub(r'[^0-9]', '', parts[0])
    return re.sub(r'[^0-9]', '', parts[0]) + '.' + re.sub(r'[^0-9]', '', ''.join(parts[1:]))


class PaddleReader:
    """
    PaddleOCR reader configured for numeric meter displays.
    Ensemble weight = 0.3
    """

    def __init__(self, use_gpu: bool = False):
        self._ok = False
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                enable_mkldnn=False,
            )
            self._ok = True
            print("PaddleReader ready.")
        except Exception as e:
            print(f"PaddleReader load failed ({e}). Disabled.")

    # ------------------------------------------------------------------
    def read(self, image: np.ndarray) -> dict:
        """image must be a CROPPED display region (BGR)."""
        if not self._ok:
            return {"text": "", "confidence": 0.0, "weight": 0.3}
        try:
            results = self.ocr.ocr(image, cls=True)
            if not results or not results[0]:
                return {"text": "", "confidence": 0.0, "weight": 0.3}
            candidates = []
            for line in results[0]:
                raw  = line[1][0]
                conf = float(line[1][1])
                if not _DIGIT_RE.search(raw):
                    continue
                cleaned = _clean(raw)
                if cleaned and _DIGIT_RE.search(cleaned):
                    # Score: confidence × length × decimal bonus
                    dec   = 1.5 if '.' in cleaned else 1.0
                    score = conf * len(cleaned) * dec
                    candidates.append((cleaned, conf, score))
            if not candidates:
                return {"text": "", "confidence": 0.0, "weight": 0.3}
            best_text, best_conf, _ = max(candidates, key=lambda x: x[2])
            return {"text": best_text, "confidence": best_conf, "weight": 0.3}
        except Exception as e:
            print(f"PaddleReader inference error: {e}")
            return {"text": "", "confidence": 0.0, "weight": 0.3}
