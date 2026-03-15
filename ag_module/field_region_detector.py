"""
ag_module/field_region_detector.py
Heuristic field-region splitter for multi-reading utility meters.

Strategy:
  1. Try YOLO display detection first (reuses existing YOLOv8 adapter).
  2. If YOLO fails or only one box found, use aspect-ratio heuristics to
     split the crop into labelled sub-regions.
  3. All regions returned as { field_name: np.ndarray crop }.
"""
import cv2
import numpy as np
from typing import Dict, Optional


# ── Relative row/column positions for a typical single-phase kWh meter ──────
# These are fractions of (height, width) of the DISPLAY crop.
# Adjust when training a proper multi-field YOLO model.
_FIELD_LAYOUT_MONO = {
    # name: (y_frac_start, y_frac_end, x_frac_start, x_frac_end)
    "kwh":        (0.10, 0.55, 0.00, 1.00),
    "kvah":       (0.55, 0.80, 0.00, 0.50),
    "md_kw":      (0.55, 0.80, 0.50, 1.00),
    "demand_kva": (0.80, 1.00, 0.00, 0.50),
}

_SERIAL_LAYOUT = {
    # Serial number nameplate — typically bottom 20% of full meter image
    "meter_serial": (0.80, 1.00, 0.10, 0.90),
}


def _crop(image: np.ndarray, y0f, y1f, x0f, x1f) -> np.ndarray:
    h, w = image.shape[:2]
    y0, y1 = int(h * y0f), int(h * y1f)
    x0, x1 = int(w * x0f), int(w * x1f)
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y1), min(w, x1)
    if y1 <= y0 or x1 <= x0:
        return image
    return image[y0:y1, x0:x1]


class FieldRegionDetector:
    """
    Splits a meter image (or display crop) into per-field crops.

    Parameters
    ----------
    yolo_model_path : str or None
        Path to YOLOv8 multi-class model trained with field classes.
        If None or not found, falls back to heuristic layout.
    """

    def __init__(self, yolo_model_path: Optional[str] = None):
        self._yolo = None
        if yolo_model_path:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(yolo_model_path)
            except Exception as e:
                print(f"FieldRegionDetector: YOLO load failed ({e}), using heuristics.")

    # ------------------------------------------------------------------
    def detect(
        self,
        display_crop: np.ndarray,
        full_image: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns dict mapping field name → image crop.

        Parameters
        ----------
        display_crop : the already-cropped display region (from YOLOv8 detection)
        full_image   : full meter image (used for serial detection)
        """
        regions: Dict[str, np.ndarray] = {}

        # ── Try YOLO multi-field detection ─────────────────────────────
        if self._yolo is not None:
            try:
                regions = self._yolo_detect(display_crop)
            except Exception as e:
                print(f"FieldRegionDetector YOLO error ({e}), falling back.")
                regions = {}

        # ── Heuristic layout fallback ───────────────────────────────────
        if not regions:
            for name, (y0f, y1f, x0f, x1f) in _FIELD_LAYOUT_MONO.items():
                crop = _crop(display_crop, y0f, y1f, x0f, x1f)
                if crop.size > 0:
                    regions[name] = crop

        # ── Serial number from full image ───────────────────────────────
        src = full_image if full_image is not None else display_crop
        for name, (y0f, y1f, x0f, x1f) in _SERIAL_LAYOUT.items():
            crop = _crop(src, y0f, y1f, x0f, x1f)
            if crop.size > 0:
                regions[name] = crop

        return regions

    # ------------------------------------------------------------------
    def _yolo_detect(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        results = self._yolo(image, conf=0.4, verbose=False)
        regions: Dict[str, np.ndarray] = {}
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
                crop = image[max(0,y1):y2, max(0,x1):x2]
                if crop.size > 0:
                    regions[cls_name] = crop
        return regions
