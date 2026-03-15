"""
ocr_pipeline/multi_field_ocr.py
MultiFieldOCR — runs the full OCR ensemble on each named field crop,
then passes aggregated results through the structured post-processor.
Returns structured dict compatible with api/app.py OCRResponseSchema.
"""
import re
import numpy as np
import cv2
from typing import Dict, Any, Optional

from ocr_pipeline.trocr_adapter       import TrOCRAdapter
from ocr_pipeline.paddle_adapter      import PaddleAdapter
from ocr_pipeline.easyocr_adapter     import EasyOCRAdapter
from ocr_pipeline.ensemble_rover      import DecimalAwareRover
from ocr_pipeline.llm_post_processor  import OCRPostProcessor

_HAS_DIGIT = re.compile(r'\d')

# Per-field domain: expected decimal places
_FIELD_DECIMALS = {
    "kwh": 1, "kvah": 1, "md_kw": 2, "demand_kva": 2, "meter_serial": 0,
}


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Strong preprocessing on any sub-crop before OCR to handle glare & low res."""
    if crop is None or crop.size == 0:
        return crop

    # 1. Glare reduction (inpainting bright specular highlights)
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh_val = min(240, int(gray.mean() + 60))
        _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 5)
        if cv2.countNonZero(th) > 0:
            crop = cv2.inpaint(crop, th, 5, cv2.INPAINT_TELEA)
    except Exception as e:
        print(f"Glare reduction failed: {e}")

    # 2. CLAHE for contrast enhancement
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"CLAHE failed: {e}")

    # 3. Super-res (fast bicubic upscale to help small digits)
    h, w = crop.shape[:2]
    # If the image is very small, scale it up significantly (e.g., 2x or to a min width of 200)
    if w < 200:
        scale = max(2.0, 200 / w)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif w < 800:
        # standard 2x upscale for readability
        crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 4. Cap width to avoid massive images slowing down OCR
    h, w = crop.shape[:2]
    if w > 1000:
        new_h = int(h * 1000 / w)
        crop = cv2.resize(crop, (1000, new_h), interpolation=cv2.INTER_AREA)

    # 5. Fallback Bridge: Dot Matrix Morphology & Horizontal Squeezing
    # Converts dot-matrix digits into solid fonts and closes character gaps
    # so text-based sequence transformers (like TrOCR) sequence it correctly.
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        solid = cv2.dilate(thresh, kernel_connect, iterations=1)
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel_smooth)
        clean_inv = cv2.bitwise_not(clean)
        clean_rgb = cv2.cvtColor(clean_inv, cv2.COLOR_GRAY2BGR)
        # Squeeze by 0.5 horizontally
        crop = cv2.resize(clean_rgb, None, fx=0.5, fy=1.0, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Dot matrix morph failed: {e}")

    return crop


class MultiFieldOCR:
    """
    Runs the OCR ensemble on any set of named field crops and returns
    structured per-field results compatible with OCRResponseSchema.

    Pipeline: 3-engine OCR → ROVER voting → PostProcessor (calibration,
    decimal analysis, domain validation, flag generation).

    Usage
    -----
        mf = MultiFieldOCR()
        results = mf.run({"kwh": crop_arr}, image_quality={...})
        # results["kwh"] = {"value": "12345.6", "probability": 0.97, ...}
    """

    def __init__(self):
        self.trocr    = TrOCRAdapter()
        self.paddle   = PaddleAdapter()
        self.easyocr  = EasyOCRAdapter()
        self.rover    = DecimalAwareRover(decimal_penalty=2.0)
        self.post_processor = OCRPostProcessor()

    # ------------------------------------------------------------------
    def _ocr_one_crop(self, crop: np.ndarray, field: str) -> Dict[str, Any]:
        """Run full ensemble on a single crop and return raw ROVER result."""
        proc = _preprocess_for_ocr(crop)

        # Run 3 engines
        trocr_r  = self.trocr.recognize(proc)
        paddle_r = self.paddle.recognize(proc)
        easy_r   = self.easyocr.recognize(proc)

        # ROVER vote
        roved = self.rover.align_and_vote([trocr_r, paddle_r, easy_r])
        raw_text = roved["text"]
        raw_conf = roved["confidence"]

        # If ALL engines returned empty, return empty placeholder
        if not raw_text or not _HAS_DIGIT.search(raw_text):
            return self._empty_field(field)

        return {
            "value":       raw_text,
            "probability": float(raw_conf),
            "sources":     ["trocr", "paddleocr", "easyocr"],
            "decimals":    raw_text.count('.'),
            "candidates":  roved.get("candidates", []),
            "debug": {
                "raw_ocr":  raw_text,
                "trocr":    trocr_r.get("text", ""),
                "paddle":   paddle_r.get("text", ""),
                "easyocr":  easy_r.get("text", ""),
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_field(field: str) -> Dict[str, Any]:
        return {
            "value":       "—",
            "probability": 0.0,
            "sources":     [],
            "decimals":    0,
            "candidates":  [],
            "debug":       {"raw_ocr": "", "reason": "no_numeric_text_found"},
        }

    # ------------------------------------------------------------------
    def run(
        self,
        field_crops: Dict[str, np.ndarray],
        image_quality: Optional[dict] = None,
        dec_conf_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parameters
        ----------
        field_crops : mapping of field_name → BGR numpy array
        image_quality : dict from analyze_image_quality() — passed to post-processor
        dec_conf_map : optional override decimal confidence per field

        Returns
        -------
        Dict[field_name, structured_result]
        """
        # Step 1: Run OCR ensemble on each field crop
        raw_results = {}
        for field, crop in field_crops.items():
            if crop is None or crop.size == 0:
                raw_results[field] = self._empty_field(field)
                continue
            raw_results[field] = self._ocr_one_crop(crop, field)

        # Step 2: Run post-processor on aggregated results
        pp_output = self.post_processor.process(
            image_id="current",
            raw_ocr_fields=raw_results,
            image_quality=image_quality,
        )

        # Step 3: Merge post-processor results back into raw_results
        pp_results_by_field = {}
        for r in pp_output.get("results", []):
            # Map spec field names back to pipeline field names
            spec_field = r.get("field", "")
            from ocr_pipeline.llm_post_processor import _FIELD_MAP_REV
            pipeline_field = _FIELD_MAP_REV.get(spec_field, spec_field)
            pp_results_by_field[pipeline_field] = r

        enriched = {}
        for field, raw in raw_results.items():
            pp = pp_results_by_field.get(field, {})
            enriched[field] = {
                "value":              pp.get("corrected") or raw.get("value", "—"),
                "probability":        pp.get("confidence") if pp.get("confidence") is not None else raw.get("probability", 0.0),
                "sources":            raw.get("sources", []),
                "decimals":           raw.get("decimals", 0),
                "candidates":         raw.get("candidates", []),
                "debug":              raw.get("debug", {}),
                # New post-processor fields
                "decimal_confidence": pp.get("decimal_confidence"),
                "decimal_position":   pp.get("decimal_position"),
                "flags":              pp.get("flags", []),
                "reason":             pp.get("reason"),
            }
            # Inject post-processor debug into existing debug dict
            if pp.get("debug_notes"):
                enriched[field]["debug"]["post_processor_notes"] = pp["debug_notes"]

        # Store overall QC info for api/app.py to use
        enriched["_post_processor_meta"] = {
            "overall_pass": pp_output.get("overall_pass"),
            "qc_reasons":   pp_output.get("qc_reasons", []),
        }

        return enriched
