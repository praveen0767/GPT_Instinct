"""
ocr_pipeline/ensemble_rover.py
DecimalAwareRover — numeric-only token-alignment voting.
Heavily boosts pure-digit candidates; discards non-numeric ones.
"""
import re
from typing import List, Dict

_HAS_DIGIT   = re.compile(r'\d')
_PURE_NUMERIC = re.compile(r'^\d+\.?\d*$')


def _is_numeric(text: str) -> bool:
    return bool(_HAS_DIGIT.search(text.strip()))


class DecimalAwareRover:
    """
    Weighted consensus voting across multiple OCR hypotheses.

    Scoring weights:
      - Base:          1.0
      - Has digit:    ×10   (filter non-numeric candidates)
      - Pure numeric: ×3    (e.g. "12345" vs "12A45")
      - Has decimal:  ×2    (decimals are physically meaningful)
      - Length bonus: ×√len (longer readings are more specific)
    """

    def __init__(self, decimal_penalty: float = 2.0):
        self.decimal_penalty = decimal_penalty  # kept for backward compat

    # ------------------------------------------------------------------
    def align_and_vote(self, ocr_results: List[Dict]) -> Dict:
        """
        Parameters
        ----------
        ocr_results : list of dicts, each ``{text, confidence, ...}``

        Returns
        -------
        dict  ``{text, confidence, candidates}``
        """
        if not ocr_results:
            return {"text": "", "confidence": 0.0, "candidates": []}

        # ── Step 1: filter out empty and non-numeric results ────────────
        valid = [r for r in ocr_results if r.get("text") and _is_numeric(r["text"])]

        if not valid:
            return {"text": "", "confidence": 0.0, "candidates": []}

        # ── Step 2: score each candidate ────────────────────────────────
        text_scores: Dict[str, float] = {}
        for r in valid:
            t    = r["text"].strip()
            conf = float(r.get("confidence", 0.0))

            # Base weight
            w = 1.0

            # Strongly prefer text that is purely numeric
            if _PURE_NUMERIC.match(t):
                w *= 3.0

            # Decimal bonus
            if '.' in t:
                w *= self.decimal_penalty   # default 2.0

            # Length bonus — prefer longer digit strings
            import math
            w *= max(1.0, math.sqrt(len(t)))

            text_scores[t] = text_scores.get(t, 0.0) + w * conf

        # ── Step 3: pick best ───────────────────────────────────────────
        best_text  = max(text_scores, key=text_scores.get)
        total      = sum(text_scores.values())

        candidates = [
            {"value": t, "score": round(s / total, 4) if total > 0 else 0.0}
            for t, s in sorted(text_scores.items(), key=lambda x: -x[1])
        ]

        # Average confidence of engines that agreed on best_text
        voters     = [r["confidence"] for r in valid if r["text"].strip() == best_text]
        all_voters = [r["confidence"] for r in valid]
        # Weighted: voters get higher weight
        final_conf = (sum(voters) * 2 + sum(all_voters)) / (2 * len(voters) + len(all_voters)) \
                     if voters else sum(all_voters) / len(all_voters)

        return {
            "text": best_text,
            "confidence": float(final_conf),
            "candidates": candidates,
        }
