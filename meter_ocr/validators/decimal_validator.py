"""
meter_ocr/validators/decimal_validator.py
Decimal placement validator.

Spec requirements:
  - Detects decimal pixel using OCR output + morphological CV
  - Validates decimal position using regex pattern: \\d{3,7}\\.\\d
  - Example: 12345.6
"""
import re
import cv2
import numpy as np
from typing import Optional, Dict


# Spec-defined regex
_DECIMAL_PATTERN = re.compile(r'^\d{3,7}\.\d$')

# Energy reading pattern (more flexible)
_READING_PATTERN = re.compile(r'^\d{1,7}(\.\d{0,4})?$')


def _count_decimal_candidates(image: np.ndarray) -> int:
    """
    Count small dot-like blobs in lower half of image.
    These are likely decimal points.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    roi  = gray[int(h * 0.4):, :]          # lower 60%
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 2 < area < 80:
            x, y, bw, bh = cv2.boundingRect(c)
            asp = bw / (bh + 1e-6)
            if 0.3 < asp < 2.5:
                dots += 1
    return dots


class DecimalValidator:
    """
    Validates and corrects decimal placement for meter readings.

    Rules per spec:
      Expected format: \\d{3,7}\\.\\d  (e.g. 12345.6)
    """

    def validate(
        self,
        text: str,
        field: str = "kwh",
        display_crop: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        text         : raw OCR string (already cleaned of non-numeric chars)
        field        : field name for domain-specific decimal rules
        display_crop : optional image for CV-based decimal confirmation

        Returns
        -------
        {
          "value"        : corrected string,
          "has_decimal"  : bool,
          "decimal_conf" : float,  # confidence in decimal placement
          "was_corrected": bool,
        }
        """
        t        = text.strip()
        original = t
        was_corrected = False

        # ── CV dot count (optional) ──────────────────────────────────────
        dot_count = 0
        if display_crop is not None:
            try:
                dot_count = _count_decimal_candidates(display_crop)
            except Exception:
                pass
        cv_has_decimal = dot_count > 0

        # ── Check if digit string already has decimal ────────────────────
        has_decimal = '.' in t

        # ── Case: no decimal in OCR but CV says there should be one ─────
        if not has_decimal and cv_has_decimal and t.isdigit():
            # For energy fields (kWh, kVAh): insert decimal 1 position from right
            if field in ("kwh", "kvah") and len(t) >= 2:
                t = t[:-1] + '.' + t[-1]
                has_decimal = True
                was_corrected = True
            elif field in ("md_kw", "demand_kva") and len(t) >= 2:
                # MD kW / Demand kVA: one decimal from right
                t = t[:-1] + '.' + t[-1]
                has_decimal = True
                was_corrected = True

        # ── Validate against spec pattern for kWh/kVAh ──────────────────
        matches_spec = bool(_DECIMAL_PATTERN.match(t)) if field in("kwh","kvah") else True

        # ── Decimal confidence ───────────────────────────────────────────
        decimal_conf = 0.0
        if has_decimal:
            decimal_conf = 0.92 if matches_spec else 0.70
            if cv_has_decimal:
                decimal_conf = min(1.0, decimal_conf + 0.07)
        else:
            decimal_conf = 0.50  # uncertain

        return {
            "value":         t,
            "has_decimal":   has_decimal,
            "decimal_conf":  round(decimal_conf, 4),
            "was_corrected": was_corrected,
            "matches_spec":  matches_spec,
        }
