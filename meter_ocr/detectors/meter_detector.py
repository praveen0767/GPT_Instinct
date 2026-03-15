"""
meter_ocr/detectors/meter_detector.py
Full meter body detector — finds the meter bounding box in the photo.
Used upstream of lcd_detector to isolate the meter from surroundings.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class MeterDetector:
    """
    Detects the entire meter body in an input photo.

    Primary  : YOLOv8 (class 'meter')
    Fallback : largest rectangular contour (edge-based)
    """

    def __init__(self, yolo_path: Optional[str] = None, conf: float = 0.35):
        self._yolo = None
        if yolo_path:
            try:
                import os
                from ultralytics import YOLO
                if os.path.isfile(yolo_path):
                    self._yolo = YOLO(yolo_path)
                    print(f"MeterDetector: YOLO loaded from {yolo_path}")
            except Exception as e:
                print(f"MeterDetector: YOLO load failed ({e}), using edge fallback.")
        self._conf = conf

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[tuple]]:
        """
        Returns (cropped_meter, bbox) or (original_image, None).
        """
        bbox = None

        if self._yolo is not None:
            try:
                results = self._yolo(image, conf=self._conf, verbose=False)
                best_box = None; best_c = 0
                for r in results:
                    for box in r.boxes:
                        c = float(box.conf[0])
                        if c > best_c:
                            best_c   = c
                            best_box = [int(v) for v in box.xyxy[0].cpu().tolist()]
                if best_box:
                    bbox = tuple(best_box)
            except Exception as e:
                print(f"MeterDetector inference error: {e}")

        if bbox is None:
            bbox = self._edge_fallback(image)

        if bbox is None:
            return image, None

        x1, y1, x2, y2 = bbox
        crop = image[max(0,y1):y2, max(0,x1):x2]
        return (crop if crop.size > 0 else image), bbox

    def _edge_fallback(self, image: np.ndarray) -> Optional[tuple]:
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        edged  = cv2.Canny(blur, 50, 200)
        kernel = np.ones((3, 3), np.uint8)
        edged  = cv2.dilate(edged, kernel, iterations=1)
        cnts, _= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            peri  = cv2.arcLength(c, True)
            approx= cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                return x, y, x + w, y + h
        # Just return largest contour bbox
        x, y, w, h = cv2.boundingRect(cnts[0])
        return x, y, x + w, y + h
