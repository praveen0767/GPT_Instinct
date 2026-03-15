"""
meter_ocr/ocr/ensemble.py
Weighted OCR ensemble.

Weights (per spec):
  TrOCR     → 0.5
  PaddleOCR → 0.3
  EasyOCR   → 0.2

Confidence boost if all 3 engines agree on the same value.
"""
import re
import numpy as np
from typing import List, Dict

_DIGIT_RE = re.compile(r'\d')
_SUBS = {'O':'0','o':'0','I':'1','l':'1','S':'5','B':'8','Z':'2','G':'6'}


def _clean(text: str) -> str:
    t = text.strip()
    for f, r in _SUBS.items():
        t = t.replace(f, r)
    parts = t.split('.')
    if len(parts) == 1:
        return re.sub(r'[^0-9]', '', parts[0])
    return re.sub(r'[^0-9]', '', parts[0]) + '.' + re.sub(r'[^0-9]', '', ''.join(parts[1:]))


class OCREnsemble:
    """
    Aggregates TrOCR, PaddleOCR, EasyOCR results using weighted voting.
    """

    CONSENSUS_BOOST = 0.10     # confidence boost when all 3 agree

    def __init__(self):
        from meter_ocr.ocr.trocr_reader  import TrOCRReader
        from meter_ocr.ocr.paddle_reader import PaddleReader
        self.trocr  = TrOCRReader()
        self.paddle = PaddleReader()
        self.easy   = self._init_easy()

    # ------------------------------------------------------------------
    @staticmethod
    def _init_easy():
        """Load EasyOCR; return a stub on failure."""
        try:
            import easyocr, re, numpy as np
            reader = easyocr.Reader(['en'], gpu=False)

            class _EasyWrapper:
                weight = 0.2
                def __init__(self, r): self._r = r
                def read(self, image):
                    results = self._r.readtext(image, detail=1, paragraph=False)
                    candidates = []
                    for (_, text, conf) in results:
                        if not re.search(r'\d', text): continue
                        t = text.strip()
                        for f, rp in {'O':'0','I':'1','S':'5','B':'8','Z':'2'}.items():
                            t = t.replace(f, rp)
                        c = re.sub(r'[^0-9.]', '', t)
                        if c and re.search(r'\d', c):
                            dec = 1.5 if '.' in c else 1.0
                            candidates.append((c, float(conf), conf * len(c) * dec))
                    if not candidates: return {"text":"","confidence":0.0,"weight":0.2}
                    best, bc, _ = max(candidates, key=lambda x: x[2])
                    return {"text": best, "confidence": bc, "weight": 0.2}

            return _EasyWrapper(reader)
        except Exception as e:
            print(f"EasyOCR load failed ({e}). Disabled.")

            class _Stub:
                weight = 0.2
                def read(self, image): return {"text":"","confidence":0.0,"weight":0.2}
            return _Stub()

    # ------------------------------------------------------------------
    def run(self, display_crop: np.ndarray) -> Dict:
        """
        Run all 3 engines on the already-cropped display region and combine.

        Returns
        -------
        {
          "text"      : str   — best reading,
          "confidence": float — calibrated weighted confidence,
          "candidates": list  — per-engine results,
          "all_agree" : bool,
        }
        """
        results = [
            self.trocr.read(display_crop),
            self.paddle.read(display_crop),
            self.easy.read(display_crop),
        ]

        # ── Filter empty results ─────────────────────────────────────────
        valid = [r for r in results if r["text"] and _DIGIT_RE.search(r["text"])]
        if not valid:
            return {"text": "", "confidence": 0.0, "candidates": results, "all_agree": False}

        # ── Weighted vote ────────────────────────────────────────────────
        score_map: Dict[str, float] = {}
        for r in valid:
            t = _clean(r["text"])
            w = r["weight"] * r["confidence"]
            score_map[t] = score_map.get(t, 0.0) + w

        best_text = max(score_map, key=score_map.get)
        total_w   = sum(score_map.values())
        raw_conf  = score_map[best_text] / total_w if total_w > 0 else 0.0

        # ── Consensus boost ──────────────────────────────────────────────
        texts    = [_clean(r["text"]) for r in valid if r["text"]]
        all_agree= len(set(texts)) == 1 and len(texts) == 3
        confidence= min(1.0, raw_conf + (self.CONSENSUS_BOOST if all_agree else 0.0))

        return {
            "text":       best_text,
            "confidence": round(confidence, 4),
            "candidates": [{"engine": ["trocr","paddle","easyocr"][i],
                            "text": r["text"], "confidence": r["confidence"]}
                           for i, r in enumerate(results)],
            "all_agree":  all_agree,
        }
