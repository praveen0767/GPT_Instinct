"""
ocr_pipeline/digit_recognizer.py
Step 8: Digit Recognition Model

Implements a standalone Digit Recognizer wrapper that utilizes the
YOLOv8 classification capabilities or a lightweight CNN specifically built for 0-9 digits.
Currently acts as a wrapper for YOLO class confidence outputs if we use one unified model.
"""
from typing import List, Dict

class DigitRecognizer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        # If we train a separate MobileNet/ResNet for just the crops, load it here.
        # For now, if the segmentation phase provides the YOLO class, we wrap it.
        pass

    def recognize(self, chips: List[Dict]) -> List[Dict]:
        """
        Takes a list of segmented chips from `segment_digits`.
        In a purely hierarchical YOLO setup where digits are already classed 0-9,
        we just format the output probability.
        """
        results = []
        for c in chips:
            # We assume yolo_class (0-9 actual integer mappings) is already computed
            pred_digit = str(c.get("yolo_class", ""))
            prob = float(c.get("confidence", 0.0))
            
            results.append({
                "digit": pred_digit,
                "prob": prob,
                "bbox": c.get("bbox")
            })
            
        return results
