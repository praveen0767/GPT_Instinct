"""
ag_module/decimal_detector.py
Decimal detector with CV-based rule fallback.
Works without a trained model — uses morphological analysis to detect
small dots in the digit region.

When a trained model checkpoint is present at `model_path`, it loads
the MobileNetV3-Small CNN for higher accuracy.
"""
import cv2
import numpy as np
import os
from typing import Optional


# ── CV-based decimal detection (no model needed) ─────────────────────────────

def _cv_decimal_confidence(crop: np.ndarray) -> float:
    """
    Estimate probability that a decimal point exists in the digit crop.

    Strategy:
    1. Convert to grayscale + threshold.
    2. Search the lower-half of the image for small isolated blobs
       (area 3–50 px, roughly circular) that could be decimal dots.
    3. Return confidence in [0, 1] based on blob count and size.
    """
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    h, w = gray.shape

    # Focus on lower 60% of the crop where decimal usually lives
    roi = gray[int(h * 0.4):, :]

    # Threshold to get dark dots on light background OR invert for LCD
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to clean noise
    kernel = np.ones((2, 2), np.uint8)
    clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2 or area > 80:              # size filter
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-6)
        if 0.4 < aspect < 2.5:                  # roughly square → dot
            # Prefer dots in horizontal centre of width
            cx = x + bw / 2
            if w * 0.05 < cx < w * 0.95:
                candidates.append(area)

    if not candidates:
        return 0.0

    # Confidence heuristic: 1 dot → 0.75, 2+ dots → 0.90
    n = len(candidates)
    conf = min(0.90, 0.55 + 0.15 * n)
    return float(conf)


# ── CNN-based detection (optional, requires trained checkpoint) ───────────────

class _CNNDecimalDetector:
    """Lightweight MobileNetV3-Small binary classifier."""

    def __init__(self, model_path: str):
        import torch
        import torchvision.models as models

        self.device = "cpu"
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, 2
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self.model = model
        self._torch = torch

    def predict(self, crop: np.ndarray) -> float:
        import torchvision.transforms as T
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((64, 64))
        transform = T.Compose([T.ToTensor(),
                                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = transform(pil).unsqueeze(0)
        with self._torch.no_grad():
            logits = self.model(x)
            prob = self._torch.softmax(logits, dim=1)[0, 1].item()
        return float(prob)


# ── Public interface ──────────────────────────────────────────────────────────

class DecimalDetectorConfig:
    """
    Unified decimal detector:
    - If model_path exists → CNN inference.
    - Else → pure CV heuristic.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._cnn: Optional[_CNNDecimalDetector] = None
        if model_path and os.path.isfile(model_path):
            try:
                self._cnn = _CNNDecimalDetector(model_path)
                print(f"DecimalDetector: CNN loaded from {model_path}")
            except Exception as e:
                print(f"DecimalDetector: CNN load failed ({e}), using CV fallback.")

    def detect(self, crop: np.ndarray) -> float:
        """Return confidence ∈ [0, 1] that the crop contains a decimal point."""
        if self._cnn is not None:
            try:
                return self._cnn.predict(crop)
            except Exception as e:
                print(f"DecimalDetector CNN inference error ({e}), using CV.")
        return _cv_decimal_confidence(crop)
