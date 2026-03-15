"""
meter_ocr/preprocessing/glare_removal.py
Glare detection and removal from meter display crops.
"""
import cv2
import numpy as np


def detect_glare_mask(image: np.ndarray, sat_min: int = 30, val_min: int = 220) -> np.ndarray:
    """
    Detect glare using HSV saturation + value thresholds.
    Low saturation + very high value → specular highlight (glare).
    Returns binary mask (255 = glare pixel).
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    glare_mask = np.uint8(((s < sat_min) & (v > val_min)) * 255)
    kernel = np.ones((5, 5), np.uint8)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
    return glare_mask


def remove_glare(image: np.ndarray) -> np.ndarray:
    """
    Inpaint specular glare from meter images.
    Falls back gracefully if OpenCV inpaint not available.
    """
    mask = detect_glare_mask(image)
    glare_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    if glare_ratio < 0.005:           # < 0.5% glare — nothing to do
        return image
    if glare_ratio > 0.40:            # too much glare, inpainting would distort
        # Fall back to CLAHE only
        return _clahe_enhance(image)
    try:
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    except Exception:
        return _clahe_enhance(image)


def _clahe_enhance(image: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return image
