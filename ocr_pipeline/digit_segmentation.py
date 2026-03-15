"""
ocr_pipeline/digit_segmentation.py
Step 7: Digit Segmentation Module

Isolates and sorts numeric digit bounding boxes from the YOLOv8 detections.
Outputs clean `(H, W, 3)` chips representing single digits, enforcing left-to-right
structural ordering required to fight truncation.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple

def sort_digits_ltr(boxes: List[Dict]) -> List[Dict]:
    """Sort digit bounding boxes strictly Left-to-Right."""
    return sorted(boxes, key=lambda b: b["bbox"][0])

def segment_digits(image: np.ndarray, detections: List[Dict]) -> List[Dict]:
    """
    Given an image and YOLO detections ranging from 2-11 (digits 0-9),
    validates overlap and extracts padded image chips per digit slot.
    """
    digit_dets = [d for d in detections if d.get("class") in [str(i) for i in range(2, 13)]]
    
    # 1. Sort geographically L-to-R
    sorted_digits = sort_digits_ltr(digit_dets)
    
    # 2. Extract crops
    chips = []
    h, w = image.shape[:2]
    
    for d in sorted_digits:
        bx, by, bw, bh = d["bbox"]
        
        # Add slight padding (~5%)
        px = int(bw * 0.05)
        py = int(bh * 0.05)
        
        x1 = max(0, bx - px)
        y1 = max(0, by - py)
        x2 = min(w, bx + bw + px)
        y2 = min(h, by + bh + py)
        
        chip = image[y1:y2, x1:x2]
        
        if chip.size > 0:
            cls_str = d["class"]
            if cls_str == "12":
                actual_digit_val = "."
            else:
                actual_digit_val = str(int(cls_str) - 2)
            
            chips.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": d["confidence"],
                "yolo_class": actual_digit_val,
                "chip": chip
            })
            
    return chips
