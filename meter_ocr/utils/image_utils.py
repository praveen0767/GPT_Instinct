"""
meter_ocr/utils/image_utils.py
Common image utilities shared across the pipeline.
"""
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load BGR image, raise if not found."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def resize_keep_aspect(image: np.ndarray, max_w: int = 1280, max_h: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def apply_clahe(image: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return image


def upscale_small(image: np.ndarray, min_w: int = 200, factor: float = 2.0) -> np.ndarray:
    """Upscale a crop that is too small for reliable OCR."""
    h, w = image.shape[:2]
    if w < min_w:
        scale = min_w / w
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def detect_glare(image: np.ndarray, threshold_ratio: float = 0.05) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(mask) / (image.shape[0] * image.shape[1]) > threshold_ratio


def detect_tilt(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def analyze_image_quality(image: np.ndarray) -> dict:
    """Full quality analysis; returns flags dict."""
    blur     = detect_blur(image)
    glare    = detect_glare(image)
    tilt_deg = detect_tilt(image)
    not_leg  = blur or abs(tilt_deg) > 40.0
    return {
        "blur": bool(blur),
        "glare": bool(glare),
        "tilt_deg": round(tilt_deg, 2),
        "not_legible": bool(not_leg),
    }


def draw_boxes(image: np.ndarray, boxes: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out
