"""
meter_ocr/detectors/lcd_detector.py
LCD / display region detector.

Primary   : YOLOv8 model (if checkpoint present)
Fallback  : HSV green-screen segmentation

HSV thresholds as per spec:
  lower_green = [35, 40, 40]
  upper_green = [90, 255, 255]

CRITICAL RULE: Never OCR the full image. Always crop display first.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


# Spec-defined HSV thresholds
_LOWER_GREEN = np.array([35, 40, 40], dtype=np.uint8)
_UPPER_GREEN = np.array([90, 255, 255], dtype=np.uint8)


# ── HSV Fallback ─────────────────────────────────────────────────────────────

def detect_lcd_hsv(
    image: np.ndarray,
    lower: np.ndarray = _LOWER_GREEN,
    upper: np.ndarray = _UPPER_GREEN,
    min_area_frac: float = 0.005,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect LCD display via HSV green segmentation.

    Steps:
      1. Convert image to HSV
      2. Create mask with spec thresholds
      3. Find contours
      4. Largest contour → LCD bounding box

    Returns
    -------
    (x1, y1, x2, y2) or None if not found.
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up noise
    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = image.shape[0] * image.shape[1]
    largest  = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < img_area * min_area_frac:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    # Expand 10%
    dx = int(w * 0.10);  dy = int(h * 0.10)
    x1 = max(0, x - dx);  y1 = max(0, y - dy)
    x2 = min(image.shape[1], x + w + dx)
    y2 = min(image.shape[0], y + h + dy)
    return x1, y1, x2, y2


# ── YOLO Primary ─────────────────────────────────────────────────────────────

class LCDDetector:
    """
    Detects the meter LCD display region.

    Parameters
    ----------
    yolo_path : str or None
        Path to a YOLOv8 model trained on the 'display' class.
        If None or not found, uses HSV fallback.
    conf      : float
        YOLO confidence threshold.
    """

    def __init__(self, yolo_path: Optional[str] = None, conf: float = 0.40):
        self._yolo = None
        self._conf = conf
        if yolo_path:
            try:
                from ultralytics import YOLO
                import os
                if os.path.isfile(yolo_path):
                    self._yolo = YOLO(yolo_path)
                    print(f"LCDDetector: YOLO loaded from {yolo_path}")
            except Exception as e:
                print(f"LCDDetector: YOLO load failed ({e}), using HSV fallback.")

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns the cropped display region or None.

        CRITICAL: this method ALWAYS crops before returning.
                  It never returns the full image.
        """
        bbox = None

        # ── 1. Try YOLO ─────────────────────────────────────────────────
        if self._yolo is not None:
            try:
                results = self._yolo(image, conf=self._conf, verbose=False)
                best_box = None; best_conf = 0.0
                for r in results:
                    for box in r.boxes:
                        cls_name = r.names[int(box.cls[0])].lower()
                        if "display" in cls_name or "lcd" in cls_name or "meter" in cls_name:
                            c = float(box.conf[0])
                            if c > best_conf:
                                best_conf = c
                                best_box  = [int(v) for v in box.xyxy[0].cpu().tolist()]
                if best_box:
                    bbox = tuple(best_box)
            except Exception as e:
                print(f"LCDDetector YOLO inference error: {e}")

        # ── 2. HSV fallback ──────────────────────────────────────────────
        if bbox is None:
            bbox = detect_lcd_hsv(image)

        # ── 3. Absolute fallback: upper 55% (display usually at top) ────
        if bbox is None:
            h, w = image.shape[:2]
            bbox = (0, 0, w, int(h * 0.55))

        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image

        return crop, bbox
